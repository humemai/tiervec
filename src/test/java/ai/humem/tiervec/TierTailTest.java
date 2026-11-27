package ai.humem.tiervec;

import static ai.humem.tiervec.SegmentTestSupport.vectors;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/**
 * A tier whose tail is not empty, written with checkpoint and read back (#158). The scenario has two sealed segments and a tail that holds new
 * ids, deletes of sealed ids, a replaced id, and ids that were deleted and then added again. The segments are smaller than the beam of a
 * search, so every search is exact and the answers can be compared whether a row is in a segment or in the tail.
 */
class TierTailTest {
    static final int DIM = TierReopenTest.DIM;
    static final long EPOCH = 7;
    /** rows of the two segments of the scenario: ids 1 to 40 and 41 to 72 */
    static final int FIRST = 40, SECOND = 32;
    /** the largest id of the scenario, which is in the tail; the largest id of the segments is {@code FIRST + SECOND} */
    static final long LAST = 96;
    static final float[][] DATA = vectors(400, DIM, 1581);
    static final float[][] QUERIES = vectors(40, DIM, 1582);

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

    static TieredIndex.Config cfg() {
        return TierReopenTest.cfg();
    }

    /** a host: the batches it committed, the records that they leave, and what it says it recovered */
    static final class Host implements RecordSource {
        final List<TieredIndex.Batch> batches = new ArrayList<>();
        final TreeMap<Long, float[]> records = new TreeMap<>();
        final List<Long> askedAbove = new ArrayList<>();
        long recovered = Manifest.NONE;
        private int nextVector;

        void commit(TieredIndex t, long[] adds, long... deletes) {
            List<TieredIndex.IdVector> a = new ArrayList<>();
            for (long id : adds) a.add(new TieredIndex.IdVector(id, DATA[nextVector++]));
            List<Long> d = new ArrayList<>();
            for (long id : deletes) d.add(id);
            TieredIndex.Batch b = new TieredIndex.Batch(a, d);
            batches.add(b);
            for (TieredIndex.IdVector iv : a) {
                records.put(iv.id(), iv.v());
                recovered = Math.max(recovered, iv.id());
            }
            d.forEach(records::remove);
            t.apply(b);
        }

        /** forgets every record above {@code max}, as after a crash that lost the end of the log */
        void forgetAbove(long max) {
            records.tailMap(max, false).clear();
            recovered = max;
        }

        @Override
        public long epoch() {
            return EPOCH;
        }

        @Override
        public Iterable<TieredIndex.IdVector> allLive() {
            throw new AssertionError("the index went cold and asked for every record");
        }

        @Override
        public Iterable<TieredIndex.IdVector> liveAbove(long id) {
            askedAbove.add(id);
            return records.tailMap(id, false).entrySet().stream().map(e -> new TieredIndex.IdVector(e.getKey(), e.getValue())).toList();
        }

        @Override
        public boolean isLive(long id) {
            return records.containsKey(id);
        }

        @Override
        public long recoveredMax() {
            return recovered;
        }
    }

    private static long[] ids(long from, long to) {
        long[] r = new long[(int) (to - from + 1)];
        for (int i = 0; i < r.length; i++) r[i] = from + i;
        return r;
    }

    /** the scenario: two segments of ids 1 to 72, then writes that stay in the tail, up to id 96 */
    private TieredIndex scenario(Host h) {
        TieredIndex t = track(new TieredIndex(DIM, Metric.L2, cfg()));
        t.rebuildFrom(List.<TieredIndex.IdVector>of(), EPOCH, 0); // the epoch of the host
        h.commit(t, ids(1, FIRST));
        t.sealNow();
        h.commit(t, ids(FIRST + 1, FIRST + SECOND));
        t.sealNow();
        assertTrue(t.awaitIdle(60_000));
        h.commit(t, ids(73, 84));                 // new ids
        h.commit(t, new long[0], 3, 41, 42);      // deletes of sealed ids
        h.commit(t, new long[] {5, 75});          // a sealed id and a tail id replaced
        h.commit(t, new long[0], 7, 78);          // a sealed id and a tail id deleted ...
        h.commit(t, new long[] {7, 78});          // ... and added again
        h.commit(t, ids(85, LAST), 80);           // more new ids, and a tail id deleted for good
        assertEquals(2, t.segmentCount());
        assertTrue(t.pendingTail() > 0);
        return t;
    }

    private static Hit[][] answers(TieredIndex t) {
        Hit[][] out = new Hit[QUERIES.length][];
        for (int i = 0; i < QUERIES.length; i++) out[i] = t.search(QUERIES[i], 10, null);
        return out;
    }

    private static void assertSameBits(Hit[][] want, TieredIndex got, String what) {
        for (int i = 0; i < QUERIES.length; i++) assertArrayEquals(want[i], got.search(QUERIES[i], 10, null), what + ", query " + i);
    }

    /** the tier holds exactly these records: no id more, none less, and each under its own vector (a stale one has another score) */
    private static void assertHolds(TieredIndex t, Map<Long, float[]> records, String what) {
        assertEquals(records.size(), t.size(), what + ": size");
        for (long id = 1; id <= LAST + 40; id++) {
            final long target = id;
            float[] v = records.get(id);
            Hit[] r = t.search(v != null ? v : QUERIES[0], 1, x -> x == target);
            if (v == null) {
                assertEquals(0, r.length, what + ": id " + id + " is not live and is returned");
            } else {
                assertEquals(1, r.length, what + ": id " + id + " is not found");
                assertEquals(TierT.score(Metric.L2, v, v), r[0].score(), 1e-3f, what + ": id " + id + " holds another vector");
            }
        }
    }

    private static Set<String> names(Path d) throws IOException {
        try (Stream<Path> s = Files.list(d)) {
            return s.map(p -> p.getFileName().toString()).filter(n -> !n.equals(DirectoryLock.FILE)).collect(Collectors.toCollection(TreeSet::new));
        }
    }

    private static Set<String> segments(Path d) throws IOException {
        return names(d).stream().filter(n -> n.startsWith("seg-")).collect(Collectors.toCollection(TreeSet::new));
    }

    private static Set<String> expected(Manifest m) {
        Set<String> s = new TreeSet<>();
        s.add(Manifest.manifestName(m.generation()));
        s.add(Manifest.tombName(m.tombGeneration()));
        if (m.tailRows() > 0) s.add(Manifest.tailName(m.generation()));
        for (Manifest.Entry e : m.entries()) s.add(Manifest.segName(e.segId()));
        return s;
    }

    @Test
    void checkpointWritesTheTailAndBuildsNoSegment() throws IOException {
        Host h = new Host();
        TieredIndex t = scenario(h);
        final int tail = t.pendingTail();
        final long seals = t.stats().seals();
        Hit[][] before = answers(t);

        t.checkpoint(dir);

        assertEquals(2, t.segmentCount());
        assertEquals(seals, t.stats().seals());
        assertEquals(tail, t.pendingTail(), "the tail stays in the tier");
        assertSameBits(before, t, "the tier that was checkpointed");
        Manifest m = Manifest.readLatest(dir);
        assertEquals(2, m.entries().size());
        assertEquals(tail, m.tailRows());
        assertEquals(LAST, m.hwm(), "the high-water id covers the tail");
        assertEquals(TailFile.HEADER_LEN + tail * (16L + 4L * DIM), m.tailLength());
        assertEquals(m.tailLength(), Files.size(dir.resolve(Manifest.tailName(m.generation()))));
        assertEquals(expected(m), names(dir));
    }

    @Test
    void aReopenBringsTheTailBack() throws IOException {
        Host h = new Host();
        TieredIndex t = scenario(h);
        Hit[][] before = answers(t);
        assertHolds(t, h.records, "before the checkpoint");
        t.checkpoint(dir);

        TieredIndex u = track(TieredIndex.open(dir, cfg()));
        assertNull(u.openFallbackCause());
        assertEquals(t.size(), u.size());
        assertEquals(t.pendingTail(), u.pendingTail());
        assertEquals(2, u.segmentCount());
        assertEquals(EPOCH, u.epoch());
        assertSameBits(before, u, "after open");
        assertHolds(u, h.records, "after open");

        // a checkpoint of the reopened tier writes the tail again and no segment, and the generation before goes with its tail file
        Set<String> segs = segments(dir);
        u.checkpoint(dir);
        Manifest m = Manifest.readLatest(dir);
        assertEquals(2, m.generation());
        assertEquals(segs, segments(dir));
        assertEquals(expected(m), names(dir));
        assertSameBits(before, track(TieredIndex.open(dir, cfg())), "after the second open");

        // writes go on over the restored rows: a delete and a replace in the tail, then a seal that takes it
        h.commit(u, new long[] {75, 5}, 7, 85);
        assertHolds(u, h.records, "writes after the open");
        u.sealNow();
        assertTrue(u.awaitIdle(60_000));
        assertHolds(u, h.records, "after the seal of the restored tail");
        u.checkpoint(dir);
        Manifest sealed = Manifest.readLatest(dir);
        assertEquals(0, sealed.tailRows());
        assertEquals(expected(sealed), names(dir), "an empty tail has no file");
        assertHolds(track(TieredIndex.open(dir, cfg())), h.records, "after the open of the sealed tier");
    }

    @Test
    void anOpenWithTheRecordSourceReplaysNothingAndApplyingTheBatchesAgainChangesNothing() throws IOException {
        Host h = new Host();
        TieredIndex t = scenario(h);
        Hit[][] before = answers(t);
        t.checkpoint(dir);

        TieredIndex u = track(TieredIndex.open(dir, DIM, Metric.L2, cfg(), h));
        assertNull(u.openFallbackCause());
        assertEquals(List.of(LAST), h.askedAbove, "the replay starts above the largest id of the segments and the tail");
        assertEquals(t.pendingTail(), u.pendingTail());
        assertSameBits(before, u, "after open");
        assertHolds(u, h.records, "after open");

        h.batches.forEach(u::apply);
        Hit[][] once = answers(u);
        for (int i = 0; i < QUERIES.length; i++) {
            assertEquals(before[i].length, once[i].length);
            for (int j = 0; j < before[i].length; j++) {
                assertEquals(before[i][j].id(), once[i][j].id(), "query " + i + ", hit " + j);
                assertEquals(before[i][j].score(), once[i][j].score(), Math.abs(before[i][j].score()) * 1e-5f + 1e-6f, "query " + i + ", hit " + j);
            }
        }
        assertHolds(u, h.records, "after the batches were applied again");
        h.batches.forEach(u::apply);
        assertSameBits(once, u, "the second apply");
        assertHolds(u, h.records, "after the second apply");
    }

    @Test
    void idsOfTheTailAboveTheRecoveredMaximumAreDroppedForGood() throws IOException {
        Host h = new Host();
        TieredIndex t = scenario(h);
        t.checkpoint(dir);
        t.close();
        Set<String> segs = segments(dir);

        h.forgetAbove(82); // ids 83 to 96 are in the tail
        TieredIndex u = track(TieredIndex.open(dir, DIM, Metric.L2, cfg(), h));
        assertNull(u.openFallbackCause());
        assertEquals(List.of(82L), h.askedAbove);
        assertHolds(u, h.records, "after the drop");
        Manifest m = Manifest.readLatest(dir);
        assertEquals(2, m.generation(), "the drop is published before open returns");
        assertEquals(82, m.hwm());
        assertEquals(u.pendingTail(), m.tailRows());
        assertEquals(segs, segments(dir), "a drop from the tail alone builds no segment");
        assertEquals(expected(m), names(dir));
        u.close();

        // a later reopen, by a reader that knows nothing about the host
        TieredIndex again = track(TieredIndex.open(dir, cfg()));
        assertHolds(again, h.records, "a later open");
        again.close();

        // the host mints the dropped ids again, for other vectors: they are replayed, and the old vectors stay gone
        TieredIndex throwaway = track(new TieredIndex(DIM, Metric.L2, cfg()));
        h.commit(throwaway, ids(83, 90));
        assertHolds(track(TieredIndex.open(dir, DIM, Metric.L2, cfg(), h)), h.records, "after the ids were minted again");
    }

    @Test
    void aDropThatReachesTheSegmentsRebuildsThemAndDropsTheTail() throws IOException {
        Host h = new Host();
        TieredIndex t = scenario(h);
        t.checkpoint(dir);
        t.close();
        Set<String> segs = segments(dir);

        h.forgetAbove(60); // ids 61 to 72 are in the second segment, and every new id of the tail is above them
        TieredIndex u = track(TieredIndex.open(dir, DIM, Metric.L2, cfg(), h));
        assertNull(u.openFallbackCause());
        assertHolds(u, h.records, "after the drop");
        Manifest m = Manifest.readLatest(dir);
        assertTrue(m.hwm() <= 60, "high-water id " + m.hwm());
        assertTrue(!segments(dir).equals(segs), "the segment with dropped rows was rebuilt");
        u.close();
        assertHolds(track(TieredIndex.open(dir, cfg())), h.records, "a later open");
    }

    @Test
    void aTailIdThatIsNotLiveIsRemovedAtOpen() throws IOException {
        Host h = new Host();
        TieredIndex t = scenario(h);
        t.checkpoint(dir);
        t.close();

        h.records.remove(86L); // deleted after the checkpoint
        TieredIndex u = track(TieredIndex.open(dir, DIM, Metric.L2, cfg(), h));
        assertHolds(u, h.records, "after open");
        assertEquals(1, Manifest.readLatest(dir).generation(), "a liveness check alone writes nothing");
        u.checkpoint(dir);
        u.close();
        assertHolds(track(TieredIndex.open(dir, cfg())), h.records, "after the next checkpoint");
    }

    @ParameterizedTest
    @EnumSource(Metric.class)
    void aTierThatIsAllTailRoundTripsWithEveryMetricAndTailKernel(Metric metric) throws IOException {
        for (Quant q : new Quant[] {Quant.NONE, Quant.SQ8}) {
            TieredIndex.Config c = cfg().withTailQuant(q);
            Path d = dir.resolve(q.name());
            TieredIndex t = track(new TieredIndex(DIM, metric, c));
            for (int i = 0; i < 300; i++) t.add(i, DATA[i]);
            for (int i = 0; i < 300; i += 7) t.delete(i);
            for (int i = 1; i < 300; i += 11) t.add(i, DATA[(i + 50) % 300]);
            Hit[][] before = answers(t);
            t.checkpoint(d);
            assertEquals(0, t.segmentCount());
            TieredIndex u = track(TieredIndex.open(d, c));
            assertEquals(0, u.segmentCount());
            assertEquals(t.size(), u.size());
            assertSameBits(before, u, metric + " " + q);
        }
    }

    @Test
    void aTailThatAFailedSealLeftFrozenIsWrittenWithTheActiveOne() throws IOException {
        Host h = new Host();
        TieredIndex t = scenario(h);
        t.setSealHooks(() -> {
            throw new IllegalStateException("the seal fails before it publishes");
        }, null);
        assertThrows(IllegalStateException.class, t::sealNow);
        // the hook stays: the worker tries the frozen tail again, and every try has to fail for the tail to stay frozen until the test ends
        // one more row in the new active tail, and a delete and a replace of rows of the frozen one
        h.commit(t, new long[] {97, 90}, 91);
        assertEquals(2, t.segmentCount());
        Hit[][] before = answers(t);
        t.checkpoint(dir);
        assertEquals(t.pendingTail(), Manifest.readLatest(dir).tailRows());
        TieredIndex u = track(TieredIndex.open(dir, cfg()));
        assertEquals(t.pendingTail(), u.pendingTail());
        assertSameBits(before, u, "after open");
        assertHolds(u, h.records, "after open");
    }

    @Test
    void theTailFileHoldsTheRowsInTheOrderInWhichTheyArrived() throws IOException {
        TieredIndex t = track(new TieredIndex(DIM, Metric.COSINE, cfg()));
        t.add(30, DATA[0]);
        t.add(10, DATA[1]);
        t.add(20, DATA[2]);
        t.add(30, DATA[3]); // replaced: it arrives again, last
        t.checkpoint(dir);
        ByteBuffer b = ByteBuffer.wrap(Files.readAllBytes(dir.resolve(Manifest.tailName(1)))).order(ByteOrder.LITTLE_ENDIAN);
        int p = TailFile.HEADER_LEN;
        assertEquals(10, b.getLong(p));
        assertEquals(20, b.getLong(p + 8));
        assertEquals(30, b.getLong(p + 16));
        assertEquals(2, b.getLong(p + 24));
        assertEquals(3, b.getLong(p + 32));
        assertEquals(4, b.getLong(p + 40));
        // the vectors are the ones that were passed in, not the normalized copies that a cosine scan reads
        for (int j = 0; j < DIM; j++) assertEquals(Float.floatToRawIntBits(DATA[1][j]), b.getInt(p + 48 + 4 * j));
    }

    @Test
    void aManifestOfMajor1OpensAndTheNextCheckpointWritesMajor2() throws IOException {
        // a directory written before the tail was persisted: no tail, rows hidden by a delete and by a replace
        Host h = new Host();
        TieredIndex t = scenario(h);
        t.sealNow();
        assertTrue(t.awaitIdle(60_000));
        Hit[][] before = answers(t);
        t.checkpoint(dir);
        t.close();
        Manifest m = Manifest.readLatest(dir);
        assertEquals(0, m.tailRows());
        Path file = dir.resolve(Manifest.manifestName(m.generation()));
        Files.write(file, m.encode(1, 0));
        assertEquals(m.encode(2, 0).length - 24, (int) Files.size(file), "major 1 has no tail fields");

        TieredIndex u = track(TieredIndex.open(dir, cfg()));
        assertNull(u.openFallbackCause());
        assertSameBits(before, u, "a manifest of major 1");
        assertHolds(u, h.records, "a manifest of major 1");
        h.commit(u, new long[] {98}, 73);
        u.checkpoint(dir);
        Manifest next = Manifest.readLatest(dir);
        byte[] raw = Files.readAllBytes(dir.resolve(Manifest.manifestName(next.generation())));
        assertEquals(2, raw[8]);
        assertEquals(1, next.tailRows());
        assertHolds(track(TieredIndex.open(dir, cfg())), h.records, "after the next checkpoint");
    }
}
