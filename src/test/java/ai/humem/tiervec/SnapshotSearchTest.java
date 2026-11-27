package ai.humem.tiervec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.AtomicReferenceArray;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/**
 * A search reads one version of the tier (issue #163): it answers from the state when it started, whatever is written, sealed, merged, or rebuilt while
 * it scans, and it never waits for a write.
 *
 * <p>Every search here asks for every id and passes a filter that says it allows few ({@link SizedIdFilter}), so the tails and the segments are scanned
 * exactly and a complete answer holds each live id once, with the score of its vector: a missing id, a deleted one, or an old vector is a wrong answer and
 * not a recall miss. A model holds the state of the tier, and the vector of an id is named by the number of the write that gave it.
 */
@Timeout(value = 180, unit = TimeUnit.SECONDS)
class SnapshotSearchTest {
    static final int DIM = 16;
    static final int EF = 1 << 20;
    /** the rows that the random writes of a test start from */
    static final int ROWS = 300;

    static float[] vec(long id, long write) {
        return TierT.vec(write, id, DIM);
    }

    /** a filter that allows every id, says that it allows few, and runs {@code each} on every call */
    private static SizedIdFilter all(Runnable each) {
        return new SizedIdFilter() {
            @Override
            public boolean test(long id) {
                each.run();
                return true;
            }

            @Override
            public long allowedCount() {
                return 1;
            }
        };
    }

    private static Hit[] everything(TieredIndex ix, float[] q, SizedIdFilter filter) {
        return ix.search(q, 100_000, filter, EF);
    }

    /** null when the hits are the ids of the model, each once and with the score of its vector, else what differs */
    static String difference(float[] q, Hit[] hits, Map<Long, Long> model) {
        Map<Long, Float> got = new HashMap<>();
        for (Hit h : hits) if (got.put(h.id(), h.score()) != null) return "id " + h.id() + " twice";
        for (Map.Entry<Long, Long> e : model.entrySet()) {
            Float score = got.remove(e.getKey());
            if (score == null) return "id " + e.getKey() + " is missing";
            float[] v = vec(e.getKey(), e.getValue());
            double dot = 0;
            for (int i = 0; i < DIM; i++) dot += (double) q[i] * v[i];
            if (Math.abs(dot - score) > 1e-4) return "id " + e.getKey() + " scored " + score + ", its vector then scores " + dot;
        }
        return got.isEmpty() ? null : "ids that were not held: " + got.keySet();
    }

    /** only sealNow and mergeNow seal and merge, nothing is throttled, and the hard tail limit is out of reach */
    private static TieredIndex.Config manual(Quant tail, TieredIndex.MergeMode mode) {
        TieredIndex.Config d = TieredIndex.Config.defaults();
        return new TieredIndex.Config(1 << 28, 0L, 0, 1000, 4, 1.01, 0, 4, tail, d.graph(), 1).withMergeMode(mode);
    }

    /** a search on its own thread that stops inside its {@code holdAt}-th call of the filter, with the state of the tier when it started */
    private static final class Held {
        final Map<Long, Long> then;
        final float[] q;
        final CountDownLatch inside = new CountDownLatch(1);
        final CountDownLatch resume = new CountDownLatch(1);
        final AtomicReference<Hit[]> result = new AtomicReference<>();
        final AtomicReference<Throwable> error = new AtomicReference<>();
        final String where;
        final Thread thread;

        Held(TieredIndex ix, Map<Long, Long> model, float[] q, int holdAt, String where) {
            this.then = new HashMap<>(model);
            this.q = q;
            this.where = where + ", held at call " + holdAt + " of the filter";
            AtomicInteger calls = new AtomicInteger();
            SizedIdFilter filter = all(() -> {
                if (calls.incrementAndGet() == holdAt) {
                    inside.countDown();
                    try {
                        resume.await(120, TimeUnit.SECONDS);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }
            });
            thread = new Thread(() -> {
                try {
                    result.set(everything(ix, q, filter));
                } catch (Throwable t) {
                    error.set(t);
                } finally {
                    inside.countDown(); // a search with fewer calls than holdAt finishes without stopping
                }
            }, "held-snapshot-search");
            thread.setDaemon(true);
            thread.start();
            try {
                assertTrue(inside.await(60, TimeUnit.SECONDS), "the search neither stopped nor finished: " + this.where);
            } catch (InterruptedException e) {
                throw new IllegalStateException(e);
            }
        }

        void check() {
            resume.countDown();
            try {
                thread.join(60_000);
            } catch (InterruptedException e) {
                throw new IllegalStateException(e);
            }
            assertFalse(thread.isAlive(), "the search did not finish: " + where);
            assertNull(error.get(), () -> where + ": " + error.get());
            String wrong = difference(q, result.get(), then);
            assertNull(wrong, () -> where + ": the answer is not the state at the start of the search: " + wrong);
        }
    }

    /** the tier and its model, written by the test thread only */
    private static final class Writes {
        final TieredIndex ix;
        final Map<Long, Long> model = new HashMap<>();
        final SplittableRandom r;
        long write, nextId;

        Writes(TieredIndex ix, long seed) {
            this.ix = ix;
            this.r = new SplittableRandom(seed);
        }

        private long held() {
            return model.keySet().stream().skip(r.nextInt(model.size())).findFirst().orElseThrow();
        }

        /** an id to add: a held one (a replace) or a new one; mostly new ones while the tier holds few */
        private long toAdd() {
            return model.size() < ROWS / 2 || r.nextInt(4) == 0 ? nextId++ : held();
        }

        /** an id to delete: a held one while the tier holds enough rows, else one that is not there */
        private long toDelete() {
            return model.size() > ROWS / 2 && r.nextInt(16) != 0 ? held() : Long.MAX_VALUE - r.nextInt(8);
        }

        /** adds {@link #ROWS} new ids one by one */
        void fill() {
            for (int i = 0; i < ROWS; i++) add(nextId++);
        }

        void add(long id) {
            model.put(id, ++write);
            ix.add(id, vec(id, write));
        }

        void delete(long id) {
            assertEquals(model.remove(id) != null, ix.delete(id));
        }

        /** deletes every third held id and replaces every third one, half of each by single writes and half in one batch */
        void everyThird() {
            List<Long> ids = new ArrayList<>(model.keySet());
            ids.sort(null);
            List<TieredIndex.IdVector> adds = new ArrayList<>();
            List<Long> deletes = new ArrayList<>();
            for (int i = 0; i < ids.size(); i++) {
                long id = ids.get(i);
                boolean batched = (i / 3 & 1) == 0;
                if (i % 3 == 0 && !batched) {
                    delete(id);
                } else if (i % 3 == 0) {
                    model.remove(id);
                    deletes.add(id);
                } else if (i % 3 == 1 && !batched) {
                    add(id);
                } else if (i % 3 == 1) {
                    model.put(id, ++write);
                    adds.add(new TieredIndex.IdVector(id, vec(id, write)));
                }
            }
            ix.apply(new TieredIndex.Batch(adds, deletes));
        }

        /** one add or replace, delete, or batch; the tier keeps a few hundred rows */
        void step() {
            int kind = r.nextInt(10);
            if (kind < 4) {
                add(toAdd());
            } else if (kind < 7) {
                delete(toDelete());
            } else {
                List<TieredIndex.IdVector> adds = new ArrayList<>();
                List<Long> deletes = new ArrayList<>();
                for (int i = r.nextInt(6); i >= 0; i--) {
                    long id = toAdd();
                    model.put(id, ++write);
                    adds.add(new TieredIndex.IdVector(id, vec(id, write)));
                }
                for (int i = r.nextInt(6); i > 0; i--) {
                    long id = toDelete();
                    model.remove(id);
                    deletes.add(id);
                }
                ix.apply(new TieredIndex.Batch(adds, deletes));
            }
        }

        List<TieredIndex.IdVector> records() {
            List<TieredIndex.IdVector> out = new ArrayList<>();
            model.forEach((id, w) -> out.add(new TieredIndex.IdVector(id, vec(id, w))));
            return out;
        }
    }

    @Test
    void heldSearchesAnswerFromTheStateAtTheirStartAcrossWritesSealsMergesAndRebuilds() {
        for (Quant tail : new Quant[] {Quant.NONE, Quant.SQ8}) {
            for (TieredIndex.MergeMode mode : TieredIndex.MergeMode.values()) {
                for (long seed = 1; seed <= 3; seed++) run(tail, mode, seed);
            }
        }
    }

    private static void run(Quant tail, TieredIndex.MergeMode mode, long seed) {
        try (TieredIndex ix = new TieredIndex(DIM, Metric.DOT, manual(tail, mode))) {
            Writes w = new Writes(ix, seed);
            List<Held> held = new ArrayList<>();
            int[] checked = new int[1];
            // searches that start while a seal is between its build and its publication, and writes that the seal has to carry over
            int[] inSeal = new int[1];
            ix.setSealHooks(() -> {
                for (int i = 0; i < 4; i++) w.step();
                held.add(new Held(ix, w.model, TierT.vec(9, held.size(), DIM), 1 + w.r.nextInt(Math.max(1, w.model.size())),
                        "tail " + tail + ", " + mode + ", seed " + seed + ", started inside a seal"));
                inSeal[0]++;
                for (int i = 0; i < 4; i++) w.step();
            }, null);
            w.fill();
            ix.sealNow();
            for (int step = 0; step < 220; step++) {
                String where = "tail " + tail + ", " + mode + ", seed " + seed + ", step " + step;
                int kind = w.r.nextInt(100);
                if (kind < 70) w.step();
                else if (kind < 78) ix.sealNow();
                else if (kind < 82) ix.mergeNow();
                else if (kind < 84) ix.rebuildFrom(w.records());
                else if (kind < 85) ix.rebuildFrom(() -> {
                    List<TieredIndex.IdVector> snapshot = w.records();
                    for (int i = 0; i < 5; i++) w.step(); // writes during the build: recorded, and applied before the new state is published
                    return new TieredIndex.Snapshot(-1, snapshot);
                }, 7);
                else if (held.size() < 5) held.add(new Held(ix, w.model, TierT.vec(9, step, DIM), 1 + w.r.nextInt(Math.max(1, w.model.size())), where));
                while (held.size() >= 5 || !held.isEmpty() && w.r.nextInt(12) == 0) {
                    held.remove(w.r.nextInt(held.size())).check();
                    checked[0]++;
                }
                if (step % 20 == 0) {
                    float[] q = TierT.vec(8, step, DIM);
                    String wrong = difference(q, everything(ix, q, all(() -> { })), w.model);
                    assertNull(wrong, () -> where + ": a search of the present: " + wrong);
                }
            }
            assertTrue(w.model.size() >= ROWS / 2, "the writes left too few rows: " + w.model.size());
            for (Held h : held) {
                h.check();
                checked[0]++;
            }
            assertTrue(checked[0] >= 10 && inSeal[0] >= 3, "too few held searches: " + checked[0] + ", " + inSeal[0] + " of them started inside a seal");
        }
    }

    /**
     * One search held in each source of a view: the active tail, a frozen tail whose seal is running, and a new and an old segment. Then every third id
     * of every source is deleted and every third one replaced, the seal publishes, a merge and a rebuild follow, and only then the searches go on.
     */
    @Test
    void aSearchHeldInAnySourceAnswersFromItsStart() {
        for (Quant tail : new Quant[] {Quant.NONE, Quant.SQ8}) {
            try (TieredIndex ix = new TieredIndex(DIM, Metric.DOT, manual(tail, TieredIndex.MergeMode.REBUILD))) {
                Writes w = new Writes(ix, 21);
                w.fill();
                ix.sealNow();
                w.fill();
                ix.sealNow();
                for (long id = 0; id < 2 * ROWS; id += 7) w.delete(id); // both segments have hidden rows before the searches start
                w.fill(); // the tail that the next seal freezes
                List<Held> held = new ArrayList<>();
                ix.setSealHooks(() -> {
                    for (int i = 0; i < 100; i++) w.add(w.nextId++); // the new active tail
                    int rows = w.model.size();
                    // the sources are read newest first: 100 rows of the active tail, then the frozen tail, then the two segments
                    for (int holdAt : new int[] {1, 50, 150, 100 + ROWS + 20, rows - 20}) {
                        held.add(new Held(ix, w.model, TierT.vec(9, holdAt, DIM), holdAt, "tail " + tail));
                    }
                    w.everyThird();
                }, null);
                ix.sealNow();
                ix.setSealHooks(null, null);
                assertEquals(5, held.size(), "the seal did not run");
                held.add(new Held(ix, w.model, TierT.vec(9, 1000, DIM), 1, "tail " + tail + ", after the seal"));
                w.everyThird();
                ix.mergeNow();
                held.add(new Held(ix, w.model, TierT.vec(9, 1001, DIM), w.model.size() / 2, "tail " + tail + ", after the merge"));
                w.everyThird();
                ix.rebuildFrom(w.records());
                for (Held h : held) h.check();
                float[] q = TierT.vec(8, 0, DIM);
                assertNull(difference(q, everything(ix, q, all(() -> { })), w.model), "a search of the present");
            }
        }
    }

    @Test
    void aSearchThatStartsInsideABatchSeesNoneOfItAndDoesNotWait() {
        try (TieredIndex ix = new TieredIndex(DIM, Metric.DOT, manual(Quant.NONE, TieredIndex.MergeMode.REBUILD))) {
            Writes w = new Writes(ix, 4);
            w.fill();
            ix.sealNow();
            for (int i = 0; i < 100; i++) w.step();
            Map<Long, Long> before = new HashMap<>(w.model);
            assertTrue(before.size() > 100, "too few rows: " + before.size());
            List<TieredIndex.IdVector> adds = new ArrayList<>();
            List<Long> deletes = new ArrayList<>();
            for (long id : before.keySet()) {
                if (adds.size() < 20) adds.add(new TieredIndex.IdVector(id, vec(id, 100_000))); // replaced
                else if (deletes.size() < 20) deletes.add(id);
            }
            float[] q = TierT.vec(8, 1, DIM);
            AtomicInteger ran = new AtomicInteger();
            // the hook holds the write lock, between the adds and the deletes of the batch: a search from here is as deep inside a batch as can be
            ix.setBatchHook(() -> {
                String wrong = difference(q, everything(ix, q, all(() -> { })), before);
                assertNull(wrong, () -> "a search inside a batch must see none of it: " + wrong);
                ran.incrementAndGet();
            });
            ix.apply(new TieredIndex.Batch(adds, deletes));
            ix.setBatchHook(null);
            assertEquals(1, ran.get());
            Map<Long, Long> after = new HashMap<>(before);
            adds.forEach(a -> after.put(a.id(), 100_000L));
            deletes.forEach(after::remove);
            assertNull(difference(q, everything(ix, q, all(() -> { })), after), "a search after apply returned sees all of the batch");
        }
    }

    /**
     * The acceptance test of issue #163: one thread applies batches back to back while another searches a collection whose scan is longer than a batch,
     * which the filter makes sure of: no search goes past its first candidate before two more batches have been applied. The bound of every search is one
     * scan, counted in calls of the filter: as many as the collection holds rows. A search that scans again goes past it, and the context stops it.
     */
    @Test
    void batchesBackToBackCannotStarveASearch() throws Exception {
        int rows = 1500, searches = 25;
        TieredIndex.Config d = TieredIndex.Config.defaults();
        try (TieredIndex ix = new TieredIndex(DIM, Metric.DOT, new TieredIndex.Config(200, 0L, 100, 4, 3, 0.3, 0, 4, Quant.NONE, d.graph(), 1))) {
            List<TieredIndex.IdVector> init = new ArrayList<>();
            for (long id = 0; id < rows; id++) init.add(new TieredIndex.IdVector(id, vec(id, 0)));
            ix.apply(new TieredIndex.Batch(init, List.of()));
            AtomicBoolean stop = new AtomicBoolean();
            AtomicLong batches = new AtomicLong();
            AtomicReference<Throwable> error = new AtomicReference<>();
            // each batch replaces 4 rows by 4 new ids, so every version of the collection holds the same number of rows
            Thread writer = new Thread(() -> {
                try {
                    for (long next = rows; !stop.get(); next += 4) {
                        List<TieredIndex.IdVector> adds = new ArrayList<>();
                        List<Long> deletes = new ArrayList<>();
                        for (long id = next; id < next + 4; id++) {
                            adds.add(new TieredIndex.IdVector(id, vec(id, 0)));
                            deletes.add(id - rows);
                        }
                        assertTrue(ix.awaitCapacity(System.nanoTime() + 60_000_000_000L), "no room in the tail");
                        ix.apply(new TieredIndex.Batch(adds, deletes));
                        batches.incrementAndGet();
                    }
                } catch (Throwable t) {
                    error.set(t);
                }
            }, "back-to-back-batches");
            writer.setDaemon(true);
            writer.start();
            try {
                for (int i = 0; i < searches; i++) {
                    AtomicLong calls = new AtomicLong();
                    long deadline = System.nanoTime() + 30_000_000_000L; // only so that a broken tier fails the test instead of hanging it
                    SizedIdFilter filter = all(() -> {
                        if (calls.incrementAndGet() == 1) {
                            long target = batches.get() + 2;
                            while (batches.get() < target && writer.isAlive()) Thread.onSpinWait();
                        }
                    });
                    SearchContext oneScan = () -> calls.get() > rows || System.nanoTime() - deadline >= 0;
                    Hit[] hits;
                    try {
                        hits = ix.search(TierT.vec(8, i, DIM), rows + 10, filter, EF, oneScan);
                    } catch (SearchCancelledException e) {
                        fail("search " + i + " did not finish within one scan: " + calls.get() + " calls of the filter for " + rows + " rows, " + batches.get()
                                + " batches applied");
                        return;
                    }
                    assertEquals(rows, calls.get(), "search " + i + " looked at every row of its version once");
                    assertEquals(rows, hits.length, "search " + i);
                    long newest = 0, oldest = Long.MAX_VALUE;
                    for (Hit h : hits) {
                        newest = Math.max(newest, h.id());
                        oldest = Math.min(oldest, h.id());
                    }
                    assertEquals(rows - 1, newest - oldest, "search " + i + ": the ids of one version are consecutive, a mixture of two leaves a gap");
                }
            } finally {
                stop.set(true);
                writer.join(60_000);
            }
            assertFalse(writer.isAlive(), "the writer did not stop");
            assertNull(error.get(), () -> "the writer failed: " + error.get());
            assertTrue(batches.get() >= 2L * searches, "only " + batches.get() + " batches");
        }
    }

    /**
     * One writer, several searchers, and a worker that seals and merges all the time. The writer records the state after every write. A search must answer
     * with exactly one recorded state, and not with one older than the last write that had returned when it started: a part of a batch matches no state,
     * and a lost write or a stale view gives a state that is too old.
     */
    @Test
    void everyAnswerIsOneWholeStateAndNotAnOldOne() throws Exception {
        int writes = 1200;
        TieredIndex.Config d = TieredIndex.Config.defaults();
        for (Quant tail : new Quant[] {Quant.NONE, Quant.SQ8}) {
            try (TieredIndex ix = new TieredIndex(DIM, Metric.DOT, new TieredIndex.Config(40, 0L, 20, 3, 2, 0.3, 0, 4, tail, d.graph(), 1))) {
                Writes w = new Writes(ix, 11);
                w.fill();
                AtomicReferenceArray<Map<Long, Long>> states = new AtomicReferenceArray<>(writes + 1);
                states.set(0, Map.copyOf(w.model));
                AtomicInteger begun = new AtomicInteger(), done = new AtomicInteger();
                AtomicBoolean stop = new AtomicBoolean();
                AtomicLong searches = new AtomicLong();
                AtomicReference<String> bad = new AtomicReference<>();
                AtomicReference<Throwable> error = new AtomicReference<>();
                List<Thread> readers = new ArrayList<>();
                for (int t = 0; t < 3; t++) {
                    final int ri = t;
                    Thread reader = new Thread(() -> {
                        try {
                            for (int i = 0; !stop.get(); i++) {
                                float[] q = TierT.vec(8, ri * 100_000L + i, DIM);
                                int oldest = done.get();
                                Hit[] hits = everything(ix, q, all(() -> { }));
                                int newest = begun.get();
                                String wrong = "no state to compare with";
                                for (int s = oldest; s <= newest && wrong != null; s++) {
                                    Map<Long, Long> state;
                                    while ((state = states.get(s)) == null && !stop.get()) Thread.onSpinWait(); // the writer records it right after the write
                                    if (state == null) return;
                                    wrong = difference(q, hits, state);
                                }
                                if (wrong != null) {
                                    bad.compareAndSet(null, "the answer matches no state from write " + oldest + " to write " + newest + "; against the last: " + wrong);
                                    stop.set(true);
                                }
                                searches.incrementAndGet();
                            }
                        } catch (Throwable e) {
                            error.compareAndSet(null, e);
                            stop.set(true);
                        }
                    }, "snapshot-reader-" + t);
                    reader.setDaemon(true);
                    reader.start();
                    readers.add(reader);
                }
                for (int i = 1; i <= writes && !stop.get(); i++) {
                    begun.set(i);
                    w.step();
                    states.set(i, Map.copyOf(w.model));
                    done.set(i);
                    if (i % 300 == 0) ix.rebuildFrom(w.records()); // the same state, built anew
                    if (i % 20 == 0) { // proof of overlap without timing: the writer goes on only while the searches keep finishing
                        long target = i / 20;
                        while (!stop.get() && searches.get() < target) Thread.onSpinWait();
                    }
                }
                stop.set(true);
                for (Thread t : readers) {
                    t.join(60_000);
                    assertFalse(t.isAlive(), "a reader did not stop");
                }
                assertNull(error.get(), () -> "a reader failed: " + error.get());
                assertNull(bad.get(), "tail " + tail + ": " + bad.get());
                assertTrue(searches.get() >= writes / 20, "only " + searches.get() + " searches");
                assertEquals(0L, ix.stats().failures(), () -> "background failure: " + ix.lastError());
            }
        }
    }
}
