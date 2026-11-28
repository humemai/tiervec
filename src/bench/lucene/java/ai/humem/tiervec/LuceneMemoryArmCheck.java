package ai.humem.tiervec;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * The checks of the Lucene arm of the agent-memory benchmark that a replay does not make (issue #160): a write is seen after a refresh and not
 * before, a filter that matches nothing, a collection large enough for graphs, searches beside batches on a second thread, and vectors of 1536
 * dimensions. Each runs on float32 and on the 8-bit format. The CI job "bench smoke" compiles it with the arm and runs it. Exit code 1 when a
 * check failed.
 */
public final class LuceneMemoryArmCheck {
    private static final int DIM = 32;

    private static int checks, failures;

    private static synchronized void check(boolean ok, String what) {
        checks++;
        if (!ok && failures++ < 20) System.out.println("SELFCHECK-FAILURE " + what);
    }

    private static float[] unit(SplittableRandom r, int dim) {
        float[] v = new float[dim];
        double n = 0;
        for (int i = 0; i < dim; i++) {
            v[i] = (float) (r.nextDouble() - 0.5);
            n += v[i] * v[i];
        }
        for (int i = 0; i < dim; i++) v[i] /= (float) Math.sqrt(n);
        return v;
    }

    public static void main(String[] args) throws Exception {
        for (String quant : new String[] {"none", "int8"}) {
            refreshAndNothing(quant);
            graphsAndDeletes(quant);
            batchesBesideSearches(quant);
            wideVectors(quant);
        }
        System.out.println("SELFCHECK-SUMMARY lucene checks=" + checks + " failures=" + failures);
        System.exit(failures == 0 && checks > 0 ? 0 : 1);
    }

    private static boolean exact(Hit hit, float[] q, float[] v) { return v != null && Math.abs(hit.score() - LiveRows.cosine(q, v)) < 1e-4; }

    private static void refreshAndNothing(String quant) {
        SplittableRandom r = new SplittableRandom(1);
        try (LuceneMemoryArm arm = new LuceneMemoryArm(DIM, Map.of("quant", quant))) {
            float[] v = unit(r, DIM), q = unit(r, DIM);
            check(arm.search(q, 10, null).length == 0, quant + ": an empty index has no hit");
            arm.put(1, v);
            check(arm.search(q, 10, null).length == 0, quant + ": a write is not seen before a refresh");
            arm.refresh();
            Hit[] hits = arm.search(q, 10, null);
            check(hits.length == 1 && hits[0].id() == 1 && exact(hits[0], q, v), quant + ": a write is seen after a refresh, with its exact score");
            check(arm.search(q, 1, Violations.only(new long[] {2})).length == 0, quant + ": a filter that matches nothing has no hit");
            System.out.println("lucene rescore of a query that matches nothing, quant=" + quant + ": " + arm.rescoreOfNothing(q));
            arm.delete(1);
            arm.refresh();
            check(arm.search(q, 10, null).length == 0, quant + ": a deleted row is gone");
        }
    }

    /** 2,000 rows in one flush, which is above the size from which Lucene gives a segment a graph */
    private static void graphsAndDeletes(String quant) {
        SplittableRandom r = new SplittableRandom(2);
        try (LuceneMemoryArm arm = new LuceneMemoryArm(DIM, Map.of("quant", quant))) {
            Map<Long, float[]> live = new HashMap<>();
            for (long id = 0; id < 2000; id++) {
                float[] v = unit(r, DIM);
                arm.put(id, v);
                live.put(id, v);
            }
            arm.refresh();
            check(arm.counters().get("graphs") > 0 && arm.rowsBuilt() >= 2000, quant + ": the segment got a graph: " + arm.counters() + " " + arm.rowsBuilt());
            recall(arm, live, r, quant + " after the load");
            for (long id = 0; id < 2000; id += 2) {
                if (id % 4 == 0) {
                    arm.delete(id);
                    live.remove(id);
                } else {
                    float[] v = unit(r, DIM);
                    arm.put(id, v);
                    live.put(id, v);
                }
            }
            arm.refresh();
            recall(arm, live, r, quant + " after deletes and updates");
            System.out.println("lucene graph check quant=" + quant + " " + arm.counters() + " rowsBuilt=" + arm.rowsBuilt() + " bytesHeld=" + arm.bytesHeld());
        }
    }

    private static void recall(LuceneMemoryArm arm, Map<Long, float[]> live, SplittableRandom r, String what) {
        int found = 0;
        for (int i = 0; i < 100; i++) {
            float[] q = unit(r, DIM);
            List<Long> best = new ArrayList<>(live.keySet());
            best.sort(Comparator.comparingDouble((Long id) -> -LiveRows.cosine(q, live.get(id))));
            Hit[] hits = arm.search(q, 10, null);
            check(hits.length == 10, what + ": ten hits, not " + hits.length);
            for (Hit h : hits) {
                check(exact(h, q, live.get(h.id())), what + ": a live id with its exact score: " + h);
                if (best.subList(0, 10).contains(h.id())) found++;
            }
        }
        check(found >= 900, what + ": recall@10 of 100 queries is at least 0.9, found " + found + " of 1000");
    }

    /** every batch gives all ids one vector, so a search that sees a batch in part finds two scores, or fewer than all ids */
    private static void batchesBesideSearches(String quant) throws InterruptedException {
        SplittableRandom r = new SplittableRandom(3);
        long[] ids = new long[20];
        for (int i = 0; i < ids.length; i++) ids[i] = i;
        float[] q = unit(r, DIM);
        try (LuceneMemoryArm arm = new LuceneMemoryArm(DIM, Map.of("quant", quant))) {
            float[] first = unit(r, DIM);
            for (long id : ids) arm.put(id, first);
            arm.refresh();
            AtomicBoolean done = new AtomicBoolean();
            AtomicReference<String> wrong = new AtomicReference<>();
            Thread watcher = new Thread(() -> {
                try {
                    while (!done.get()) {
                        arm.refresh();
                        Hit[] hits = arm.search(q, ids.length, Violations.only(ids));
                        if (hits.length != ids.length) wrong.set(hits.length + " hits");
                        else if (Math.abs(hits[0].score() - hits[hits.length - 1].score()) > 1e-5) wrong.set("two versions: " + hits[0] + " " + hits[hits.length - 1]);
                    }
                } catch (Throwable t) {
                    wrong.set(t.toString());
                }
            });
            watcher.start();
            float[] last = first;
            try {
                for (int b = 0; b < 300; b++) {
                    last = unit(r, DIM);
                    List<Runbook.Op> writes = new ArrayList<>();
                    for (long id : ids) writes.add(Runbook.Op.write(Runbook.Kind.UPDATE, id, last));
                    arm.applyBatch(writes);
                }
            } finally {
                done.set(true);
                watcher.join(60_000);
            }
            check(!watcher.isAlive() && wrong.get() == null, quant + ": no search beside the batches saw one in part: " + wrong.get());
            arm.refresh();
            Hit[] hits = arm.search(q, ids.length, Violations.only(ids));
            check(hits.length == ids.length && exact(hits[0], q, last), quant + ": the last batch is seen");
            List<Runbook.Op> deletes = new ArrayList<>();
            for (long id : ids) deletes.add(Runbook.Op.write(Runbook.Kind.DELETE, id, null));
            arm.applyBatch(deletes);
            arm.refresh();
            check(arm.search(q, ids.length, null).length == 0, quant + ": a batch of deletes alone is applied");
        }
    }

    /** the benchmark definition has 1,536 dimensions, above the cap of 1,024 of Lucene's formats */
    private static void wideVectors(String quant) {
        SplittableRandom r = new SplittableRandom(4);
        try (LuceneMemoryArm arm = new LuceneMemoryArm(1536, Map.of("quant", quant))) {
            Map<Long, float[]> live = new HashMap<>();
            for (long id = 0; id < 5; id++) {
                float[] v = unit(r, 1536);
                arm.put(id, v);
                live.put(id, v);
            }
            arm.refresh();
            float[] q = unit(r, 1536);
            Hit[] hits = arm.search(q, 10, null);
            check(hits.length == 5, quant + ": five hits of 1,536 dimensions, not " + hits.length);
            for (Hit h : hits) check(exact(h, q, live.get(h.id())), quant + ": the exact score at 1,536 dimensions: " + h);
        }
    }
}
