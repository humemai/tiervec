package ai.humem.tiervec;

import static ai.humem.tiervec.SegmentTestSupport.idOf;
import static ai.humem.tiervec.SegmentTestSupport.vectors;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.humem.tiervec.SegmentFormat.CorruptException;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/**
 * Checkpoint and open of a tier with deleted, replaced and re-added ids (issue #27): the tombstone file marks the hidden rows, open rebuilds the
 * floor map from them, and the reopened tier answers as the tier in memory did, also after a crash at each step of the publish.
 */
class TierTombstoneTest {
    private static final int DIM = 16;
    private static final int ROWS = 600;
    private static final int SEGMENTS = 3;
    private static final long EPOCH = 7;

    @TempDir
    Path dir;

    private final List<TieredIndex> opened = new ArrayList<>();

    /** what the tier must hold: the live ids with their current vectors, and for each id that lost a vector one of the old ones */
    private static final class Model {
        final Map<Long, float[]> live = new TreeMap<>();
        final Map<Long, float[]> stale = new TreeMap<>();
    }

    @AfterEach
    void closeAll() {
        for (TieredIndex t : opened) t.close();
    }

    private TieredIndex track(TieredIndex t) {
        opened.add(t);
        return t;
    }

    /** no automatic seal and, with a dead fraction above 1, no compaction: the hidden rows stay where they are until a test merges */
    private static TieredIndex.Config cfg(double deadFraction) {
        return new TieredIndex.Config(1_000_000, 0L, 10, 8, 4, deadFraction, 0, 4, Quant.NONE, GraphIndex.Params.defaults().withEfConstruction(64), 1);
    }

    private static TieredIndex.Config keep() {
        return cfg(2.0);
    }

    private TieredIndex build(Model m, float[][] data) {
        TieredIndex t = track(new TieredIndex(DIM, Metric.L2, keep()));
        for (int s = 0; s < SEGMENTS; s++) {
            for (int i = s * ROWS; i < (s + 1) * ROWS; i++) {
                t.add(idOf(i), data[i]);
                m.live.put(idOf(i), data[i]);
            }
            t.sealNow();
        }
        assertTrue(t.awaitIdle(60_000));
        assertEquals(SEGMENTS, t.segmentCount());
        return t;
    }

    /**
     * One round of changes over every segment: some ids deleted, some replaced by another vector, some deleted and added again, and some
     * added to the tail and deleted there. Ends with a seal, so the tail is empty.
     */
    private static void change(TieredIndex t, Model m, int round) {
        float[][] fresh = vectors(ROWS * SEGMENTS, DIM, 900 + round);
        List<Long> ids = new ArrayList<>(m.live.keySet());
        for (int i = 0; i < ids.size(); i++) {
            long id = ids.get(i);
            int what = (i + 3 * round) % 29;
            if (what == 0 || what == 1) {
                assertTrue(t.delete(id));
                m.stale.put(id, m.live.remove(id));
            } else if (what == 2) {
                t.add(id, fresh[i]);
                m.stale.put(id, m.live.put(id, fresh[i]));
            } else if (what == 3) {
                assertTrue(t.delete(id));
                m.stale.put(id, m.live.get(id));
                t.add(id, fresh[i]);
                m.live.put(id, fresh[i]);
            }
        }
        long passing = 1_000_000L + round; // lives in the tail only
        t.add(passing, fresh[0]);
        assertTrue(t.delete(passing));
        m.stale.put(passing, fresh[0]);
        t.sealNow();
        assertTrue(t.awaitIdle(60_000));
    }

    /** every live id answers with its current vector, a deleted id answers nothing, and no old vector answers at the score of a match */
    private static void assertModel(TieredIndex t, Model m, String what) {
        assertEquals(m.live.size(), t.size(), what + ": size");
        for (Map.Entry<Long, float[]> e : m.live.entrySet()) {
            final long id = e.getKey();
            Hit[] r = t.search(e.getValue(), 1, x -> x == id);
            assertEquals(1, r.length, what + ": live id " + id);
            assertEquals(0f, r[0].score(), 1e-5f, what + ": id " + id + " holds another vector");
        }
        for (Map.Entry<Long, float[]> e : m.stale.entrySet()) {
            final long id = e.getKey();
            Hit[] r = t.search(e.getValue(), 5, x -> x == id);
            if (!m.live.containsKey(id)) {
                assertEquals(0, r.length, what + ": deleted id " + id + " is returned");
            } else {
                assertEquals(1, r.length, what + ": id " + id + " once");
                assertTrue(r[0].score() < -1e-5f, what + ": id " + id + " answers with an old vector");
            }
        }
        for (Hit h : t.search(vectors(1, DIM, 5)[0], 50, null)) assertTrue(m.live.containsKey(h.id()), what + ": a plain search returned " + h.id());
    }

    private static void assertSameTier(TieredIndex a, TieredIndex b, float[][] qs, String what) {
        assertEquals(a.size(), b.size(), what + ": size");
        assertArrayEquals(a.segmentSizes(), b.segmentSizes(), what + ": segment sizes");
        assertEquals(a.stats().floorEntries(), b.stats().floorEntries(), what + ": floor entries");
        for (float[] q : qs) {
            assertArrayEquals(a.search(q, 10, null), b.search(q, 10, null), what);
            assertArrayEquals(a.search(q, 10, id -> id % 3 == 0), b.search(q, 10, id -> id % 3 == 0), what + ": filtered");
        }
    }

    @Test
    void aTierWithDeletedAndReAddedIdsReopensWithTheSameHits() throws IOException {
        Model m = new Model();
        TieredIndex t = build(m, vectors(ROWS * SEGMENTS, DIM, 1));
        change(t, m, 1);
        assertTrue(t.stats().floorEntries() > 100, "the changes left hidden copies: " + t.stats().floorEntries());
        float[][] qs = vectors(100, DIM, 2);
        t.checkpoint(dir);

        TieredIndex u = track(TieredIndex.open(dir, keep()));
        assertNull(u.openFallbackCause());
        assertSameTier(t, u, qs, "first reopen");
        assertModel(u, m, "first reopen");

        // the reopened tier takes further changes, writes them (the old segment files stay, the bitmaps are new) and reopens again
        change(u, m, 2);
        u.checkpoint(dir);
        TieredIndex v = track(TieredIndex.open(dir, keep()));
        assertSameTier(u, v, qs, "second reopen");
        assertModel(v, m, "second reopen");

        // a merge drops the marked rows for good
        v.mergeNow();
        assertTrue(v.awaitIdle(60_000));
        assertEquals(0, v.stats().floorEntries());
        assertEquals(m.live.size(), v.segmentSizes()[0]);
        v.checkpoint(dir);
        TieredIndex w = track(TieredIndex.open(dir, keep()));
        assertSameTier(v, w, qs, "after the merge");
        assertModel(w, m, "after the merge");
    }

    @Test
    void aSegmentWhoseRowsAreAllHiddenReopensEmptyOfThem() throws IOException {
        Model m = new Model();
        TieredIndex t = build(m, vectors(ROWS * SEGMENTS, DIM, 3));
        for (int i = 0; i < ROWS; i++) { // the whole first segment
            assertTrue(t.delete(idOf(i)));
            m.stale.put(idOf(i), m.live.remove(idOf(i)));
        }
        assertTrue(t.awaitIdle(60_000));
        t.checkpoint(dir);
        TieredIndex u = track(TieredIndex.open(dir, keep()));
        assertEquals(SEGMENTS, u.segmentCount());
        assertModel(u, m, "all rows of a segment hidden");
    }

    @Test
    void theWorkerCompactsAReopenedSegmentThatIsMostlyHidden() throws IOException {
        Model m = new Model();
        TieredIndex t = build(m, vectors(ROWS * SEGMENTS, DIM, 4));
        for (int i = 0; i < ROWS / 2; i++) {
            assertTrue(t.delete(idOf(i)));
            m.stale.put(idOf(i), m.live.remove(idOf(i)));
        }
        t.checkpoint(dir);
        TieredIndex u = track(TieredIndex.open(dir, cfg(0.3)));
        assertTrue(u.awaitIdle(60_000));
        assertEquals(0, u.stats().floorEntries(), "the dead fraction is above 0.3, so the segment was rewritten");
        assertModel(u, m, "after the compaction");
    }

    @ParameterizedTest
    @EnumSource(FaultHook.Step.class)
    void aCrashAtAnyStepOfAPublishWithHiddenRowsLeavesOneGeneration(FaultHook.Step stop) throws IOException {
        Model m = new Model();
        TieredIndex t = build(m, vectors(ROWS * SEGMENTS, DIM, 6));
        change(t, m, 1);
        float[][] qs = vectors(100, DIM, 7);
        List<String> first = TierHaltTest.digest(t, qs);
        int firstSize = t.size();
        t.checkpoint(dir);

        change(t, m, 2);
        List<String> second = TierHaltTest.digest(t, qs);
        IOException cut = assertThrows(IOException.class, () -> t.checkpoint(dir, step -> {
            if (step == stop) throw new IOException("crash after " + step);
        }));
        assertEquals("crash after " + stop, cut.getMessage());

        boolean renamed = stop.ordinal() >= FaultHook.Step.MANIFEST_RENAMED.ordinal();
        TieredIndex u = track(TieredIndex.open(dir, keep()));
        assertNull(u.openFallbackCause());
        assertEquals(renamed ? 2 : 1, Manifest.readLatest(dir).generation());
        assertEquals(renamed ? t.size() : firstSize, u.size());
        assertEquals(renamed ? second : first, TierHaltTest.digest(u, qs));

        // the tier that survived publishes again, and that generation holds everything
        t.checkpoint(dir);
        TieredIndex v = track(TieredIndex.open(dir, keep()));
        assertEquals(second, TierHaltTest.digest(v, qs));
        assertModel(v, m, "after the retry");
    }

    @Test
    void openWithARecordSourceKeepsTheHiddenRowsHidden() throws IOException {
        FakeHost h = new FakeHost(DIM, 11, EPOCH);
        h.commit(300);
        h.flush();
        TieredIndex t = track(TieredIndex.open(dir, DIM, Metric.L2, keep(), h));
        FakeHost.Txn x = h.commit(5, 10, 20, 30, 300); // five new ids, and four of the first segment deleted
        t.apply(FakeHost.batch(x));
        h.flush();
        t.sealNow();
        t.checkpoint(dir);
        t.close();

        TieredIndex u = track(TieredIndex.open(dir, DIM, Metric.L2, keep(), h));
        assertNull(u.openFallbackCause());
        assertTrue(u.stats().floorEntries() >= 4, "the deleted rows came back as floor entries");
        assertEquals(0, u.pendingTail(), "nothing above the high-water id has to be replayed");
        FakeHost.assertMatches(u, h, Metric.L2, "after the reopen");
    }

    @Test
    void aHiddenRowAboveTheRecoveredMaximumIsDroppedToo() throws IOException {
        FakeHost h = new FakeHost(DIM, 12, EPOCH);
        h.commit(300);
        h.flush();
        TieredIndex t = track(TieredIndex.open(dir, DIM, Metric.L2, keep(), h));
        FakeHost.Txn lost = h.commit(40); // ids 301 to 340, not durable in the host
        t.apply(FakeHost.batch(lost));
        t.sealNow();
        for (long id = 301; id <= 340; id++) assertTrue(t.delete(id)); // every row of the second segment is hidden in this generation
        t.checkpoint(dir);
        t.close();
        h.crash();
        h.commit(40); // ids 301 to 340 again, for other vectors
        h.flush();

        TieredIndex u = track(TieredIndex.open(dir, DIM, Metric.L2, keep(), h));
        assertNull(u.openFallbackCause());
        FakeHost.assertMatches(u, h, Metric.L2, "after the drop and the replay");
        assertEquals(300L, Manifest.readLatest(dir).hwm(), "the new generation holds no row above the recovered maximum");
        TieredIndex raw = track(TieredIndex.open(dir, keep()));
        assertEquals(300, raw.size());
        int rows = 0;
        for (int n : raw.segmentSizes()) rows += n;
        assertEquals(300, rows);
    }

    @Test
    void aTombstoneFileThatHidesTheNewerCopyIsRefused() throws IOException {
        Model m = new Model();
        TieredIndex t = build(m, vectors(ROWS * SEGMENTS, DIM, 8));
        t.add(idOf(0), vectors(1, DIM, 9)[0]); // the old copy is row 0 of the first segment, the new one row 0 of the fourth
        t.sealNow();
        assertTrue(t.awaitIdle(60_000));
        t.checkpoint(dir);
        Manifest gen = Manifest.readLatest(dir);
        Path file = dir.resolve(Manifest.tombName(gen.tombGeneration()));
        byte[] raw = Files.readAllBytes(file);
        ByteBuffer b = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        int pos = 24;
        long[] words = new long[gen.entries().size()];
        int[] at = new int[gen.entries().size()];
        for (int i = 0; i < gen.entries().size(); i++) {
            at[i] = pos + 16;
            words[i] = b.getLong(at[i]);
            pos += 16 + 8 * (int) ((gen.entries().get(i).n() + 63) / 64);
        }
        assertEquals(1L, words[0], "row 0 of the first segment is marked");
        assertEquals(0L, words[words.length - 1]);
        b.putLong(at[0], 0L);
        b.putLong(at[words.length - 1], 1L); // now the newer copy is marked and the older one is not
        b.putInt(raw.length - 4, SegmentFormat.crc32c(raw, 0, raw.length - 4));
        Files.write(file, raw);
        Manifest forged = new Manifest(gen.generation(), gen.epoch(), gen.hwm(), gen.versionCounter(), gen.nextSegId(), gen.fingerprint(), gen.entries(),
                gen.tombGeneration(), raw.length, SegmentFormat.crc32c(raw, 0, raw.length), gen.tailRows(), gen.tailLength(), gen.tailCrc());
        Files.write(dir.resolve(Manifest.manifestName(gen.generation())), forged.encode(Manifest.FORMAT_MAJOR, Manifest.FORMAT_MINOR));

        CorruptStateException refused = assertThrows(CorruptStateException.class, () -> TieredIndex.open(dir, keep()));
        CorruptException e = assertInstanceOf(CorruptException.class, refused.getCause());
        assertEquals("TOMBSTONES", e.section());
    }
}
