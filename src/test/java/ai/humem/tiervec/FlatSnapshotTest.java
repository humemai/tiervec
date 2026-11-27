package ai.humem.tiervec;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.LongPredicate;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/**
 * {@code FlatIndex.searchAt} (issue #163): a search as of a version answers like a search at that version, whatever was added, replaced or deleted
 * since. The writes go through the versioned forms of {@code add} and {@code delete}, one at a time and with ascending versions.
 */
@Timeout(120)
class FlatSnapshotTest {
    static final int DIM = 40;
    static final int IDS = 300;

    /** the vector that id {@code id} got at version {@code ver} */
    static float[] vec(long id, long ver) {
        return TierT.vec(ver, id, DIM);
    }

    /** what the index held at a version: the slots written, and per live id the version of its vector */
    record Point(long ver, int slots, Map<Long, Long> rows) {}

    /** one random write at version {@code ver}: mostly adds and replaces, some deletes */
    private static void write(FlatIndex ix, Map<Long, Long> model, SplittableRandom r, long ver) {
        long id = r.nextInt(IDS);
        if (model.containsKey(id) && r.nextInt(3) == 0) {
            assertTrue(ix.delete(id, ver));
            model.remove(id);
        } else {
            ix.add(id, vec(id, ver), ver);
            model.put(id, ver);
        }
    }

    /** the rows of a point in a fresh index: what the index that kept versions has to answer as */
    private static FlatIndex reference(Metric metric, Map<Long, Long> rows) {
        FlatIndex ref = new FlatIndex(DIM, metric);
        for (Map.Entry<Long, Long> e : rows.entrySet()) ref.add(e.getKey(), vec(e.getKey(), e.getValue()));
        return ref;
    }

    @Test
    void aSearchAsOfAVersionAnswersLikeASearchAtThatVersion() {
        for (Metric metric : Metric.values()) {
            check(metric, Quant.NONE, Sq8Kernel.DEFAULT);
            for (Sq8Kernel kernel : Sq8Kernel.values()) check(metric, Quant.SQ8, kernel);
        }
    }

    private static void check(Metric metric, Quant quant, Sq8Kernel kernel) {
        String what = metric + " " + quant + " " + kernel;
        FlatIndex ix = new FlatIndex(DIM, metric, quant, 0, kernel);
        Map<Long, Long> model = new HashMap<>();
        List<Point> points = new ArrayList<>();
        SplittableRandom r = new SplittableRandom(17);
        points.add(new Point(0, 0, Map.of()));
        for (long ver = 1; ver <= 1500; ver++) {
            write(ix, model, r, ver);
            if (ver % 131 == 0 || ver == 1500) points.add(new Point(ver, ix.slotsUsed(), Map.copyOf(model)));
        }
        assertTrue(ix.slotsUsed() > 448, "the writes must fill more than three chunks, they filled " + ix.slotsUsed() + " slots");
        LongPredicate third = id -> id % 3 == 0;
        for (Point p : points) {
            FlatIndex ref = reference(metric, p.rows());
            for (int i = 0; i < 5; i++) {
                float[] q = TierT.vec(1234, i, DIM);
                String where = what + ", version " + p.ver() + ", query " + i;
                // k at least the rows: the exact path, which must give every row of the version and no other, with the score of its vector then
                Hit[] all = ix.searchAt(q, IDS, null, null, p.slots(), p.rows().size(), p.ver());
                assertArrayEquals(ref.search(q, IDS, null), all, where);
                assertArrayEquals(ref.search(q, 7, third), ix.searchAt(q, 7, third, null, p.slots(), p.rows().size(), p.ver()), where + ", filtered");
                // a small k: with SQ8 the codes pick the candidates, so the ids may differ, never the version they come from
                Hit[] few = ix.searchAt(q, 3, null, null, p.slots(), p.rows().size(), p.ver());
                assertEquals(Math.min(3, p.rows().size()), few.length, where);
                for (Hit h : few) {
                    assertTrue(p.rows().containsKey(h.id()), where + ": id " + h.id() + " was not held at this version");
                    Hit[] same = ref.search(q, 1, id -> id == h.id());
                    assertEquals(same[0].score(), h.score(), where + ": the score of id " + h.id());
                }
            }
        }
        // a search of the present is what it was
        assertArrayEquals(reference(metric, model).search(TierT.vec(1234, 0, DIM), IDS, null), ix.search(TierT.vec(1234, 0, DIM), IDS, null), what);
    }

    @Test
    void anUnversionedIndexKeepsNoVersions() {
        FlatIndex ix = new FlatIndex(DIM, Metric.COSINE);
        for (long id = 0; id < 100; id++) ix.add(id, vec(id, 1));
        long before = ix.bytesHeld();
        for (long id = 0; id < 50; id++) assertTrue(ix.delete(id));
        assertEquals(before, ix.bytesHeld(), "a delete without a version allocates nothing for versions");
        assertTrue(ix.delete(60, 5));
        assertTrue(ix.bytesHeld() >= before + 8L * 64, "the first versioned delete of a chunk allocates its versions");
    }

    /** the writer publishes a point after every write, as a tier publishes a view; the readers search as of the points they pick up */
    @Test
    void searchesAsOfPublishedVersionsWhileAWriterRuns() throws Exception {
        for (Quant quant : new Quant[] {Quant.NONE, Quant.SQ8}) {
            FlatIndex ix = new FlatIndex(DIM, Metric.DOT, quant);
            AtomicReference<Point> published = new AtomicReference<>(new Point(0, 0, Map.of()));
            AtomicBoolean stop = new AtomicBoolean();
            AtomicLong searches = new AtomicLong();
            AtomicReference<String> bad = new AtomicReference<>();
            AtomicReference<Throwable> error = new AtomicReference<>();
            Thread[] readers = new Thread[3];
            for (int t = 0; t < readers.length; t++) {
                final int ri = t;
                readers[t] = new Thread(() -> {
                    try {
                        SplittableRandom r = new SplittableRandom(100 + ri);
                        while (!stop.get()) {
                            Point p = published.get();
                            // one reader keeps its point while the writer moves on, so that its searches read a version far behind. The others
                            // never wait: the writer waits for finished searches, so some reader always has to be free to finish one
                            if (ri == 0) {
                                long until = p.ver() + 200;
                                while (!stop.get() && published.get().ver() < until) Thread.onSpinWait();
                            }
                            float[] q = TierT.vec(4321, r.nextInt(50), DIM);
                            Hit[] hits = ix.searchAt(q, IDS, null, null, p.slots(), p.rows().size(), p.ver());
                            String wrong = compare(q, hits, p);
                            if (wrong != null) {
                                bad.compareAndSet(null, wrong);
                                stop.set(true);
                            }
                            searches.incrementAndGet();
                        }
                    } catch (Throwable e) {
                        error.compareAndSet(null, e);
                        stop.set(true);
                    }
                }, "flat-snapshot-reader-" + t);
                readers[t].setDaemon(true);
                readers[t].start();
            }
            Map<Long, Long> model = new HashMap<>();
            SplittableRandom r = new SplittableRandom(5);
            for (long ver = 1; ver <= 6000 && !stop.get(); ver++) {
                write(ix, model, r, ver);
                published.set(new Point(ver, ix.slotsUsed(), Map.copyOf(model)));
                // proof of overlap without timing: the writer goes on only while the readers keep finishing searches
                if (ver % 100 == 0) {
                    long target = ver / 100 * 3;
                    while (!stop.get() && searches.get() < target) Thread.onSpinWait();
                }
            }
            stop.set(true);
            for (Thread t : readers) {
                t.join(60_000);
                assertFalse(t.isAlive(), "a reader did not stop");
            }
            assertNull(error.get(), () -> "a reader failed: " + error.get());
            assertNull(bad.get(), bad.get());
            assertTrue(searches.get() >= 180, "only " + searches.get() + " searches ran beside the writer");
        }
    }

    /** null when the hits are exactly the rows of the point, each with the score of the vector it had then */
    private static String compare(float[] q, Hit[] hits, Point p) {
        if (hits.length != p.rows().size()) return "version " + p.ver() + ": " + hits.length + " hits for " + p.rows().size() + " rows";
        for (Hit h : hits) {
            Long v = p.rows().get(h.id());
            if (v == null) return "version " + p.ver() + ": id " + h.id() + " was not held";
            float[] x = vec(h.id(), v);
            double dot = 0;
            for (int i = 0; i < DIM; i++) dot += (double) q[i] * x[i];
            if (Math.abs(dot - h.score()) > 1e-4) return "version " + p.ver() + ": id " + h.id() + " scored " + h.score() + ", its vector then scores " + dot;
        }
        for (int i = 1; i < hits.length; i++) if (hits[i].id() == hits[i - 1].id()) return "version " + p.ver() + ": id " + hits[i].id() + " twice";
        return null;
    }
}
