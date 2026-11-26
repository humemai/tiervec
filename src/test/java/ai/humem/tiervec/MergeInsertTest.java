package ai.humem.tiervec;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * Merge by insertion into the largest segment ({@link TieredIndex.MergeMode#INSERT}, issue #16).
 *
 * <p>The recall bound, written before the first run: over ten generations that each add a tenth of the first size, with and without deletes
 * and replacements in the large segment, the recall at 10 of the inserting tier at a beam of {@link #EF} is at most {@link #BOUND} below the
 * recall of a tier that rebuilds at every merge, at every generation, on the same writes. This is a check of the graph at a small size; it
 * says nothing about large collections, and nothing here is a measurement of time or memory.
 *
 * <p>The class normally takes about a minute. A test that runs for five minutes is stuck (issue #144): it fails, and the log gets a thread
 * dump (pom.xml, {@code junit.jupiter.execution.timeout.threaddump.enabled}).
 */
@Timeout(value = 5, unit = TimeUnit.MINUTES, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
class MergeInsertTest {
    private static final int DIM = 32;
    private static final int BASE = 3000;
    private static final int PER_GENERATION = BASE / 10;
    private static final int GENERATIONS = 10;
    private static final int QUERIES = 200;
    private static final int K = 10;
    private static final int EF = 40;
    private static final double BOUND = 0.02;
    private static final long SEED = 16;
    /** a bound on the rows of the concurrent test: its check searches once per id, so its cost grows with the square of the rows */
    private static final int WRITES = 40_000;

    @TempDir
    Path dir;

    private final List<TieredIndex> opened = new ArrayList<>();

    @AfterEach
    void closeAll() {
        for (TieredIndex t : opened) t.close();
    }

    private TieredIndex track(TieredIndex t) {
        opened.add(t);
        return t;
    }

    /** seals and merges happen when a test asks for them; one build thread, so a build is repeatable */
    private static TieredIndex.Config cfg(TieredIndex.MergeMode mode, boolean sq8) {
        GraphIndex.Params d = GraphIndex.Params.defaults();
        GraphIndex.Params p = new GraphIndex.Params(d.M(), 100, d.efSearch(), d.alpha(), sq8, d.seed());
        return new TieredIndex.Config(1_000_000, 0L, 10, 8, 4, 0.3, 0, 4, Quant.NONE, p, 1).withMergeMode(mode);
    }

    private static float[] vec(long serial) {
        return TierT.vec(SEED, serial, DIM);
    }

    private static double recall(TieredIndex t, Map<Long, float[]> live) {
        long found = 0;
        for (int q = 0; q < QUERIES; q++) {
            float[] query = vec(5_000_000L + q);
            double[] best = new double[live.size()];
            int i = 0;
            for (float[] v : live.values()) best[i++] = TierT.score(Metric.COSINE, query, v);
            Arrays.sort(best);
            double kth = best[best.length - K];
            for (Hit h : t.search(query, K, null, EF)) {
                if (TierT.score(Metric.COSINE, query, live.get(h.id())) >= kth - 1e-9) found++;
            }
        }
        return (double) found / (QUERIES * K);
    }

    /** every live id answers with its own vector, and no other id of {@code gone} answers at all */
    private static void assertHolds(TieredIndex t, Map<Long, float[]> live, Map<Long, float[]> gone, String what) {
        assertEquals(live.size(), t.size(), what + ": size");
        for (Map.Entry<Long, float[]> e : live.entrySet()) {
            final long id = e.getKey();
            Hit[] r = t.search(e.getValue(), 1, x -> x == id);
            assertEquals(1, r.length, what + ": live id " + id);
            assertEquals(1f, r[0].score(), 1e-3f, what + ": id " + id + " holds another vector");
        }
        for (Map.Entry<Long, float[]> e : gone.entrySet()) {
            final long id = e.getKey();
            assertEquals(0, t.search(e.getValue(), 1, x -> x == id).length, what + ": deleted id " + id + " is returned");
        }
    }

    @ParameterizedTest(name = "deletes per generation {0}, sq8 {1}")
    @CsvSource({"0, false", "150, false", "150, true"})
    void recallStaysWithinTheBoundOfARebuildOverGenerations(int deletes, boolean sq8) throws IOException {
        TieredIndex insert = track(new TieredIndex(DIM, Metric.COSINE, cfg(TieredIndex.MergeMode.INSERT, sq8)));
        TieredIndex rebuild = track(new TieredIndex(DIM, Metric.COSINE, cfg(TieredIndex.MergeMode.REBUILD, sq8)));
        List<TieredIndex> both = List.of(insert, rebuild);
        Map<Long, float[]> live = new TreeMap<>();
        Map<Long, float[]> gone = new TreeMap<>();
        long nextId = 0;
        long serial = 0;
        for (int i = 0; i < BASE; i++) {
            long id = nextId++;
            float[] v = vec(serial++);
            live.put(id, v);
            for (TieredIndex t : both) t.add(id, v);
        }
        for (TieredIndex t : both) t.sealNow();
        Random r = new Random(SEED);
        for (int g = 1; g <= GENERATIONS; g++) {
            int written = 0;
            if (deletes > 0) {
                List<Long> ids = new ArrayList<>(live.keySet());
                java.util.Collections.shuffle(ids, r);
                for (int i = 0; i < deletes; i++) {
                    long id = ids.get(i);
                    gone.put(id, live.remove(id));
                    for (TieredIndex t : both) assertTrue(t.delete(id));
                }
                for (int i = deletes; i < deletes + deletes / 5; i++) { // replacements: the new vector is in another input of the merge
                    long id = ids.get(i);
                    float[] v = vec(serial++);
                    live.put(id, v);
                    written++;
                    for (TieredIndex t : both) t.add(id, v);
                }
            }
            for (int i = 0; i < PER_GENERATION; i++) {
                long id = nextId++;
                float[] v = vec(serial++);
                live.put(id, v);
                written++;
                for (TieredIndex t : both) t.add(id, v);
            }
            long insertedBefore = insert.stats().mergedVectors();
            long rebuiltBefore = rebuild.stats().mergedVectors();
            for (TieredIndex t : both) {
                t.sealNow();
                t.mergeNow();
                assertTrue(t.awaitIdle(60_000));
                assertEquals(1, t.segmentCount());
                assertEquals(live.size(), t.segmentSizes()[0]);
                assertEquals(0, t.stats().floorEntries());
            }
            assertEquals(written, insert.stats().mergedVectors() - insertedBefore, "generation " + g + ": only the new rows were inserted");
            assertEquals(live.size(), rebuild.stats().mergedVectors() - rebuiltBefore);
            double ri = recall(insert, live);
            double rr = recall(rebuild, live);
            System.out.printf("MERGE-INSERT deletes=%d sq8=%b generation=%d rows=%d recallInsert=%.4f recallRebuild=%.4f%n", deletes, sq8, g, live.size(), ri, rr);
            assertTrue(ri >= rr - BOUND, "generation " + g + ": recall after insertion " + ri + ", after a rebuild " + rr);
        }
        assertHolds(insert, live, gone, "after the generations");

        insert.checkpoint(dir);
        TieredIndex reopened = track(TieredIndex.open(dir, cfg(TieredIndex.MergeMode.INSERT, sq8)));
        assertNull(reopened.openFallbackCause());
        for (int q = 0; q < 50; q++) assertArrayEquals(insert.search(vec(5_000_000L + q), K, null), reopened.search(vec(5_000_000L + q), K, null));
    }

    @Test
    void aLargeSegmentThatIsTooHiddenIsRebuilt() {
        TieredIndex t = track(new TieredIndex(DIM, Metric.COSINE, cfg(TieredIndex.MergeMode.INSERT, false)));
        Map<Long, float[]> live = new TreeMap<>();
        Map<Long, float[]> gone = new TreeMap<>();
        for (long id = 0; id < BASE; id++) {
            live.put(id, vec(id));
            t.add(id, vec(id));
        }
        t.sealNow();
        long before = t.stats().mergedVectors();
        for (long id = 0; id < BASE; id += 2) { // half of it, above the dead fraction of 0.3
            gone.put(id, live.remove(id));
            assertTrue(t.delete(id));
        }
        final int left = live.size();
        for (long id = BASE; id < BASE + 100; id++) {
            live.put(id, vec(id));
            t.add(id, vec(id));
        }
        t.sealNow();
        t.mergeNow();
        assertTrue(t.awaitIdle(60_000));
        // the worker may have compacted the large segment first; either way its surviving rows were built again
        assertTrue(t.stats().mergedVectors() - before >= left, "rows built " + (t.stats().mergedVectors() - before) + ", at least " + left);
        assertEquals(1, t.segmentCount());
        assertHolds(t, live, gone, "after the rebuild");
    }

    @Test
    void writesDuringInsertingMergesAreNeitherLostNorRevived() throws Exception {
        GraphIndex.Params p = GraphIndex.Params.defaults().withEfConstruction(64);
        TieredIndex.Config c = new TieredIndex.Config(150, 0L, 10, 2, 2, 0.3, 0, 8, Quant.NONE, p, 2).withMergeMode(TieredIndex.MergeMode.INSERT);
        TieredIndex t = track(new TieredIndex(DIM, Metric.COSINE, c));
        Map<Long, float[]> live = new TreeMap<>();
        Map<Long, float[]> gone = new TreeMap<>();
        for (long id = 0; id < 1000; id++) {
            live.put(id, vec(id));
            t.add(id, vec(id));
        }
        t.sealNow();
        AtomicBoolean stop = new AtomicBoolean();
        AtomicReference<Throwable> failed = new AtomicReference<>();
        Thread writer = new Thread(() -> {
            try {
                Random r = new Random(1);
                long next = 1000;
                long serial = 1000;
                while (!stop.get() && serial < WRITES) {
                    int what = r.nextInt(10);
                    long old = r.nextInt((int) next);
                    if (what < 2) {
                        if (live.containsKey(old)) {
                            gone.put(old, live.remove(old));
                            if (!t.delete(old)) throw new AssertionError("delete of the live id " + old);
                        }
                    } else {
                        long id = what < 4 ? old : next++; // a replace or a re-add, or a new id
                        float[] v = vec(serial++);
                        live.put(id, v);
                        gone.remove(id);
                        t.add(id, v);
                    }
                    if ((serial & 7) == 0) Thread.sleep(1); // spreads the writes over the five seconds
                }
            } catch (Throwable e) {
                failed.set(e);
            }
        });
        writer.start();
        long end = System.nanoTime() + 5_000_000_000L;
        while (System.nanoTime() < end) {
            t.sealNow();
            t.mergeNow();
        }
        stop.set(true);
        writer.join(60_000);
        assertNull(failed.get());
        t.sealNow();
        t.mergeNow();
        assertTrue(t.awaitIdle(60_000));
        assertEquals(0, t.stats().failures(), String.valueOf(t.lastError()));
        assertHolds(t, live, gone, "after the concurrent writes");
    }
}
