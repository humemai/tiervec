package ai.humem.tiervec;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Rule F2 threaded test: W writer threads (add, replace, delete on their own id ranges) and R reader threads for a fixed time.
 * Pass = zero exceptions, no reader ever sees a deleted id (deleted before the search started), results sorted and duplicate free, every
 * writer's own insert is found right after the insert (read-your-writes, checked every 25 ops), every writer finds its own last 100
 * inserts at the end, and (exact path only) final size and 100 random queries equal brute force over the writers' models.
 * Args: seconds (default 60) quant (NONE, SQ8, BINARY; default NONE). The index never exceeds MAXN vectors (the test caps itself).
 */
public final class FlatConcurrencyTest {
    static final int DIM = 100; // not a multiple of 8
    static final int MAXN = 60000;
    static final int W = 8, R = 8;

    public static void main(String[] args) throws Exception {
        int seconds = args.length > 0 ? Integer.parseInt(args[0]) : 60;
        Quant quant = args.length > 1 ? Quant.valueOf(args[1]) : Quant.NONE;
        FlatIndex idx = new FlatIndex(DIM, Metric.COSINE, quant);
        idx.setCompactionPolicy(0.3, 1024, 2.0);
        idx.startAutoCompaction(100); // built-in daemon, besides the explicit compactor thread below
        ConcurrentLinkedQueue<String> errors = new ConcurrentLinkedQueue<>();
        AtomicBoolean stop = new AtomicBoolean();
        long deadline = System.nanoTime() + seconds * 1_000_000_000L;
        ConcurrentHashMap<Long, Integer> deletedAt = new ConcurrentHashMap<>();
        AtomicInteger dseq = new AtomicInteger();
        AtomicLong writes = new AtomicLong(), reads = new AtomicLong();
        @SuppressWarnings("unchecked")
        Map<Long, float[]>[] models = new Map[W];
        @SuppressWarnings("unchecked")
        ArrayDeque<Long>[] recents = new ArrayDeque[W];

        AtomicInteger compactions = new AtomicInteger();
        Thread compactor = new Thread(() -> { // background compaction while writers and readers run
            try {
                while (!stop.get() && System.nanoTime() < deadline) {
                    Thread.sleep(500);
                    if (idx.deadFraction() > 0.5 && idx.slotsUsed() > 20000) {
                        idx.compact();
                        compactions.incrementAndGet();
                    }
                }
            } catch (Throwable t) {
                errors.add("compactor: " + t);
                stop.set(true);
            }
        }, "compactor");
        Thread[] ts = new Thread[W + R + 1];
        ts[W + R] = compactor;
        for (int w = 0; w < W; w++) {
            final int wi = w;
            models[w] = new HashMap<>();
            recents[w] = new ArrayDeque<>();
            ts[w] = new Thread(() -> {
                try {
                    Random r = new Random(1000 + wi);
                    Map<Long, float[]> model = models[wi];
                    ArrayDeque<Long> recent = recents[wi];
                    ArrayList<Long> liveIds = new ArrayList<>();
                    long base = (long) (wi + 1) * 1_000_000_000L, next = 1;
                    long ops = 0;
                    while (!stop.get() && System.nanoTime() < deadline) {
                        boolean full = idx.size() >= MAXN;
                        int x = r.nextInt(100);
                        int kind; // 0 add, 1 replace, 2 delete
                        if (full) kind = x < 65 ? 2 : 1;
                        else kind = (x < 70 || liveIds.isEmpty()) ? 0 : (x < 85 ? 1 : 2);
                        if (liveIds.isEmpty()) kind = 0;
                        long id;
                        float[] v;
                        if (kind == 0) {
                            id = base + next++;
                            v = FlatCorrectnessTest.unit(r, DIM);
                            idx.add(id, v);
                            model.put(id, v);
                            liveIds.add(id);
                            recent.addLast(id);
                            if (recent.size() > 100) recent.removeFirst();
                        } else if (kind == 1) {
                            id = liveIds.get(r.nextInt(liveIds.size()));
                            v = FlatCorrectnessTest.unit(r, DIM);
                            idx.add(id, v);
                            model.put(id, v);
                        } else {
                            int pos = r.nextInt(liveIds.size());
                            id = liveIds.get(pos);
                            if (recent.contains(id)) continue;
                            if (!idx.delete(id)) errors.add("writer " + wi + ": delete of own live id returned false");
                            model.remove(id);
                            liveIds.set(pos, liveIds.get(liveIds.size() - 1));
                            liveIds.remove(liveIds.size() - 1);
                            deletedAt.put(id, dseq.getAndIncrement());
                            writes.incrementAndGet();
                            ops++;
                            continue;
                        }
                        writes.incrementAndGet();
                        if (++ops % 25 == 0) { // read-your-writes
                            Hit[] h = idx.search(v, 1, null);
                            if (h.length != 1 || h[0].id() != id) errors.add("writer " + wi + ": read-your-writes failed for id " + id + " got " + (h.length == 0 ? "none" : h[0].id()));
                        }
                    }
                } catch (Throwable t) {
                    errors.add("writer " + wi + ": " + t);
                    stop.set(true);
                }
            }, "writer-" + w);
        }
        for (int rd = 0; rd < R; rd++) {
            final int ri = rd;
            ts[W + rd] = new Thread(() -> {
                try {
                    Random r = new Random(5000 + ri);
                    while (!stop.get() && System.nanoTime() < deadline) {
                        float[] q = FlatCorrectnessTest.unit(r, DIM);
                        int c = dseq.get();
                        int sizeBefore = idx.size(); // the short-result check below needs the index to be large for the whole search, not only after it
                        Hit[] h = idx.search(q, 10, null);
                        reads.incrementAndGet();
                        Set<Long> seen = new HashSet<>();
                        for (int i = 0; i < h.length; i++) {
                            if (!seen.add(h[i].id())) errors.add("reader " + ri + ": duplicate id " + h[i].id());
                            if (i > 0 && h[i - 1].score() < h[i].score()) errors.add("reader " + ri + ": not sorted");
                            Integer d = deletedAt.get(h[i].id());
                            if (d != null && d < c) errors.add("reader " + ri + ": returned id " + h[i].id() + " deleted before the search began");
                        }
                        // At startup the writers fill an empty index: a search that began with 4 vectors correctly returns 4 hits, and size() read after the
                        // search can already be above 100. A short result is a violation only when the index held more than 100 vectors both before and after.
                        int sizeAfter = idx.size();
                        if (sizeBefore > 100 && sizeAfter > 100 && h.length < 10) {
                            errors.add("reader " + ri + ": short result " + h.length + " with size " + sizeBefore + " before and " + sizeAfter + " after the search");
                        }
                    }
                } catch (Throwable t) {
                    errors.add("reader " + ri + ": " + t);
                    stop.set(true);
                }
            }, "reader-" + rd);
        }
        long t0 = System.nanoTime();
        for (Thread t : ts) {
            t.setDaemon(true);
            t.start();
        }
        for (Thread t : ts) t.join((seconds + 60) * 1000L);
        boolean allDone = true;
        for (Thread t : ts) if (t.isAlive()) allDone = false;
        T.check(allDone, "all threads finished in time");
        if (!allDone) {
            T.done("FlatConcurrencyTest");
            if (T.exitOnFailure) System.exit(2);
            return;
        }
        double secs = (System.nanoTime() - t0) / 1e9;
        System.out.println("  quant=" + quant + " ran " + String.format("%.1f", secs) + " s: " + writes.get() + " writes (" + (long) (writes.get() / secs) + "/s), "
                + reads.get() + " reads (" + (long) (reads.get() / secs) + "/s), size " + idx.size() + ", deletes " + dseq.get() + ", explicit compactions " + compactions.get() + ", total compactions " + idx.compactions() + ", daemon error " + idx.lastCompactionError() + ", bytesHeld " + idx.bytesHeld());
        int shown = 0;
        for (String e : errors) {
            if (shown++ < 10) System.out.println("  error: " + e);
        }
        T.check(errors.isEmpty(), "zero exceptions or violations during the run (" + errors.size() + ")");

        // quiescent verification
        int total = 0;
        for (Map<Long, float[]> m : models) total += m.size();
        T.check(idx.size() == total, "final size " + idx.size() + " == model " + total);
        int found = 0, wanted = 0;
        for (int w = 0; w < W; w++) {
            for (long id : recents[w]) {
                float[] v = models[w].get(id);
                if (v == null) continue; // cannot happen: recent ids are never deleted
                wanted++;
                Hit[] h = idx.search(v, 1, null);
                if (h.length == 1 && h[0].id() == id) found++;
            }
        }
        T.check(wanted >= W * 90 && found == wanted, "every writer finds its own last 100 inserts: " + found + "/" + wanted);
        Map<Long, float[]> all = new HashMap<>();
        for (Map<Long, float[]> m : models) all.putAll(m);
        Random r = new Random(7);
        boolean eq = true, noDel = true;
        int recallHit = 0;
        for (int i = 0; i < 100; i++) {
            float[] q = FlatCorrectnessTest.unit(r, DIM);
            Hit[] g = idx.search(q, 10, null), b = FlatCorrectnessTest.brute(all, Metric.COSINE, q, 10, null);
            if (quant == Quant.NONE) eq &= FlatCorrectnessTest.same(g, b);
            Set<Long> bs = new HashSet<>();
            for (Hit h : b) bs.add(h.id());
            for (Hit h : g) {
                noDel &= all.containsKey(h.id());
                if (bs.contains(h.id())) recallHit++;
            }
        }
        if (quant == Quant.NONE) T.check(eq, "100 queries equal brute force after the run");
        else System.out.println("  recall@10 vs brute force after the run (" + quant + ", random data): " + recallHit / 1000.0);
        T.check(noDel, "only live ids returned after the run");
        idx.close();
        T.check(idx.lastCompactionError() == null, "built-in compaction daemon raised no error");
        T.done("FlatConcurrencyTest " + quant);
    }
}
