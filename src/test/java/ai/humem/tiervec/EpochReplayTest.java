package ai.humem.tiervec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.humem.tiervec.SegmentFormat.MismatchException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/**
 * The id epoch, the snapshot position of a rebuild, and the replay after a reopen (issue #7, docs/ID-EPOCH-AND-REPLAY.md). Every test has a fake host
 * (FakeHost) that holds the records, mints increasing ids, and can forget its unflushed transactions; after each step the tier must hold exactly the
 * host's live records, each with the host's vector.
 */
class EpochReplayTest {
    static final int DIM = 16;
    static final Metric METRIC = Metric.L2;
    static final long EPOCH = 41;

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

    private TieredIndex open(FakeHost h) {
        return track(TieredIndex.open(dir, DIM, METRIC, FakeHost.cfg(), h));
    }

    /** the records are handed over lazily, and the host does something the first time the library asks for one: that is a write that arrives during the build */
    private static Iterable<TieredIndex.IdVector> whileReading(Iterable<TieredIndex.IdVector> in, Runnable duringBuild) {
        return () -> new Iterator<TieredIndex.IdVector>() {
            private final Iterator<TieredIndex.IdVector> it = in.iterator();
            private boolean fired;

            @Override
            public boolean hasNext() {
                if (!fired) {
                    fired = true;
                    duringBuild.run();
                }
                return it.hasNext();
            }

            @Override
            public TieredIndex.IdVector next() {
                return it.next();
            }
        };
    }

    @Test
    void rebuildFromTakesTheEpochAndTheSnapshotPosition() {
        FakeHost h = new FakeHost(DIM, 1, EPOCH);
        h.commit(40);
        TieredIndex t = track(new TieredIndex(DIM, METRIC, FakeHost.cfg()));
        assertEquals(Manifest.NONE, t.epoch());
        TieredIndex.Snapshot s = h.snapshot();
        t.rebuildFrom(s.records(), EPOCH, s.position());
        assertEquals(EPOCH, t.epoch());
        FakeHost.assertMatches(t, h, METRIC, "after the rebuild");
        assertThrows(IllegalArgumentException.class, () -> t.rebuildFrom(s.records(), Manifest.NONE, s.position()), "the value that stands for no epoch is not an epoch");
    }

    @Test
    void writesThatArriveDuringTheBuildAreKeptAndAppliedAfterTheSwap() {
        FakeHost h = new FakeHost(DIM, 2, EPOCH);
        h.commit(300);
        h.commit(20); // ids 301 to 320
        final TieredIndex t = track(new TieredIndex(DIM, METRIC, FakeHost.cfg()));
        final long[] position = new long[1];
        final Set<Long> inSnapshot = new HashSet<>();
        final FakeHost.Txn[] arrived = new FakeHost.Txn[2];
        Runnable during = () -> {
            // 5 adds above the snapshot position, and the deletion of ids 7 and 8, which the snapshot holds. They come from another thread, which
            // must get through: a rebuild that holds the write lock for the build would keep it waiting
            Thread writer = new Thread(() -> {
                synchronized (h) {
                    arrived[0] = h.commit(5, 7, 8);
                    t.apply(FakeHost.batch(arrived[0]));
                }
            }, "fake-host-writer");
            writer.start();
            try {
                writer.join(10_000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            assertFalse(writer.isAlive(), "apply did not return while the build ran: the build holds the write lock");
            synchronized (h) {
                long added = arrived[0].adds().get(0).id();
                // an update: 2 adds, and the deletion of an id that the first batch added and of id 9, which the snapshot holds
                arrived[1] = h.commit(2, added, 9);
                t.apply(FakeHost.batch(arrived[1]));
                // a batch that the host committed before its snapshot reaches the tier late: its ids are at or below the snapshot position
                t.apply(FakeHost.batch(h.txn(0)));
            }
            // the old state serves reads while the build runs: the write is visible now
            TieredIndex.IdVector probe = arrived[0].adds().get(1);
            Hit[] r = t.search(probe.v(), 1, null);
            assertEquals(1, r.length);
            assertEquals(probe.id(), r[0].id(), "a write that arrives during the build is visible at once");
        };
        t.rebuildFrom(() -> {
            TieredIndex.Snapshot s;
            synchronized (h) {
                s = h.snapshot();
            }
            position[0] = s.position();
            for (TieredIndex.IdVector iv : s.records()) inSnapshot.add(iv.id());
            return new TieredIndex.Snapshot(s.position(), whileReading(s.records(), during));
        }, EPOCH);

        assertEquals(EPOCH, t.epoch());
        assertEquals(320L, position[0]);
        FakeHost.assertMatches(t, h, METRIC, "after the swap");
        assertEquals(1, t.segmentCount(), "the snapshot became one segment");
        Map<Long, float[]> live = h.live();
        long above = live.keySet().stream().filter(id -> id > position[0]).count();
        assertEquals(above, t.pendingTail(), "the adds above the snapshot position are in the tail, once");
        long deletedFromSnapshot = inSnapshot.stream().filter(id -> !live.containsKey(id)).count();
        assertEquals(3L, deletedFromSnapshot);
        assertEquals((int) deletedFromSnapshot, t.stats().floorEntries(), "the deletes of ids of the snapshot were applied, and the late batch at or below the position was not applied again");
    }

    @Test
    void aWriteThatArrivesDuringTheBuildIsNotLost() {
        FakeHost h = new FakeHost(DIM, 13, EPOCH);
        h.commit(100);
        final TieredIndex t = track(new TieredIndex(DIM, METRIC, FakeHost.cfg()));
        // adds above the snapshot position and deletes of ids that the snapshot holds, applied by the thread that builds
        Runnable during = () -> {
            synchronized (h) {
                t.apply(FakeHost.batch(h.commit(5, 3, 4)));
                t.apply(FakeHost.batch(h.commit(0, 6)));
            }
        };
        t.rebuildFrom(() -> {
            TieredIndex.Snapshot s;
            synchronized (h) {
                s = h.snapshot();
            }
            return new TieredIndex.Snapshot(s.position(), whileReading(s.records(), during));
        }, EPOCH);
        FakeHost.assertMatches(t, h, METRIC, "after the swap");
    }

    @Test
    void anAbortedRebuildChangesNothing() {
        FakeHost h = new FakeHost(DIM, 3, EPOCH);
        h.commit(50);
        TieredIndex t = track(new TieredIndex(DIM, METRIC, FakeHost.cfg()));
        TieredIndex.Snapshot s = h.snapshot();
        t.rebuildFrom(s.records(), EPOCH, s.position());
        assertThrows(IllegalStateException.class, () -> t.rebuildFrom(() -> {
            throw new IllegalStateException("the host failed");
        }, EPOCH + 1));
        assertEquals(EPOCH, t.epoch(), "the epoch changes only when the swap happened");
        FakeHost.assertMatches(t, h, METRIC, "after the failed rebuild");
        FakeHost.Txn x = h.commit(3, 2);
        t.apply(FakeHost.batch(x));
        FakeHost.assertMatches(t, h, METRIC, "a write after the failed rebuild");
        t.rebuildFrom(() -> h.snapshot(), EPOCH + 2);
        assertEquals(EPOCH + 2, t.epoch());
        FakeHost.assertMatches(t, h, METRIC, "a rebuild after the failed one");
    }

    /** a cold open on an empty directory, then seal and checkpoint */
    private TieredIndex persisted(FakeHost h) throws IOException {
        TieredIndex t = open(h);
        assertNotNull(t.openFallbackCause(), "an empty directory gives a cold index");
        t.sealNow();
        t.checkpoint(dir);
        return t;
    }

    @Test
    void theManifestStoresTheEpochAndTheHighWaterId() throws IOException {
        FakeHost h = new FakeHost(DIM, 4, EPOCH);
        h.commit(60);
        h.flush();
        persisted(h);
        Manifest m = Manifest.readLatest(dir);
        assertEquals(EPOCH, m.epoch());
        assertEquals(60L, m.hwm(), "the largest id that a segment holds");
    }

    @Test
    void aManifestOfThePreviousLayoutOpensAndGivesAColdIndexWhenTheHostHasAnEpoch() throws IOException {
        // a tier that never got an epoch writes what the previous version wrote: epoch and high-water id unset
        FakeHost first = new FakeHost(DIM, 5, EPOCH);
        first.commit(50);
        TieredIndex old = track(new TieredIndex(DIM, METRIC, FakeHost.cfg()));
        for (TieredIndex.IdVector iv : first.allLive()) old.add(iv.id(), iv.v());
        old.sealNow();
        old.checkpoint(dir);
        Manifest m = Manifest.readLatest(dir);
        assertEquals(Manifest.NONE, m.epoch());
        assertEquals(Manifest.NONE, m.hwm());

        TieredIndex u = track(TieredIndex.open(dir, FakeHost.cfg()));
        assertEquals(Manifest.NONE, u.epoch());
        assertEquals(50, u.size());
        // a larger minor version still opens
        Files.write(dir.resolve(Manifest.manifestName(m.generation())), m.encode(Manifest.FORMAT_MAJOR, 9));
        TieredIndex w = track(TieredIndex.open(dir, FakeHost.cfg()));
        assertEquals(50, w.size());

        // the host has an epoch and the persisted state has none: the state is not trusted, the index is built from the host's records
        FakeHost other = new FakeHost(DIM, 6, EPOCH);
        other.commit(50);
        other.flush();
        TieredIndex c = open(other);
        assertInstanceOf(MismatchException.class, c.openFallbackCause());
        assertEquals(EPOCH, c.epoch());
        FakeHost.assertMatches(c, other, METRIC, "cold index");
    }

    @Test
    void anEpochMismatchGivesAColdIndex() throws IOException {
        FakeHost h = new FakeHost(DIM, 7, EPOCH);
        h.commit(80);
        h.flush();
        TieredIndex t = persisted(h);
        t.close();
        // the host compacts: it deletes records and reissues the rest in rank order under a new epoch, so id 1 is now the vector that was id 4
        h.commit(0, 1, 2, 3);
        h.flush();
        h.renumber(EPOCH + 1);
        TieredIndex u = open(h);
        assertInstanceOf(MismatchException.class, u.openFallbackCause());
        assertEquals(EPOCH + 1, u.epoch());
        FakeHost.assertMatches(u, h, METRIC, "after the renumbering");
    }

    @Test
    void idsAboveTheRecoveredMaximumAreDroppedBeforeTheIndexIsReady() throws IOException {
        FakeHost h = new FakeHost(DIM, 8, EPOCH);
        h.commit(300);
        h.flush();
        TieredIndex t = open(h);
        FakeHost.Txn lost = h.commit(40); // ids 301 to 340, not durable in the host
        t.apply(FakeHost.batch(lost));
        t.sealNow();
        t.checkpoint(dir); // the segment files hold ids that the host is about to forget
        t.close();
        h.crash();
        assertEquals(300L, h.recoveredMax());
        FakeHost.Txn minted = h.commit(40); // ids 301 to 340 again, for other vectors, before the index is opened: the recovered maximum stays 300
        assertEquals(300L, h.recoveredMax());
        assertEquals(301L, minted.adds().get(0).id());
        h.flush();

        TieredIndex u = open(h);
        assertNull(u.openFallbackCause(), "the persisted state is kept, only the rows above the recovered maximum go");
        FakeHost.assertMatches(u, h, METRIC, "after the drop and the replay");

        // the drop was durable before the index reported ready: a reader that knows nothing about the host finds none of the lost rows,
        // and the high-water id is lowered to the recovered maximum
        TieredIndex raw = track(TieredIndex.open(dir, FakeHost.cfg()));
        assertEquals(300, raw.size());
        assertEquals(300L, Manifest.readLatest(dir).hwm());
        for (TieredIndex.IdVector iv : lost.adds()) {
            final long target = iv.id();
            assertEquals(0, raw.search(iv.v(), 1, id -> id == target).length, "lost id " + target + " is still persisted");
        }
        raw.close();

        // dead is sticky: the host's new records under the dropped ids are persisted, and no later open brings the old vectors back
        u.sealNow();
        u.mergeNow();
        assertTrue(u.awaitIdle(60_000));
        u.checkpoint(dir);
        u.close();
        h.restart(); // a restart that loses nothing: the host's recovered maximum is 340 now
        TieredIndex v = open(h);
        assertNull(v.openFallbackCause());
        FakeHost.assertMatches(v, h, METRIC, "after a later generation");
    }

    @Test
    void theLivenessCheckHidesIdsTheHostDeleted() throws IOException {
        FakeHost h = new FakeHost(DIM, 9, EPOCH);
        h.commit(200);
        h.flush();
        TieredIndex t = persisted(h);
        t.close();
        long generation = Manifest.readLatest(dir).generation();
        h.commit(0, 5, 17, 100); // the host deleted three records, and the library never heard of it
        h.flush();

        TieredIndex u = open(h);
        assertNull(u.openFallbackCause());
        assertEquals(197, u.size());
        assertEquals(3, u.stats().floorEntries(), "the three rows are hidden, not rewritten");
        assertEquals(generation, Manifest.readLatest(dir).generation(), "a liveness check alone writes nothing");
        FakeHost.assertMatches(u, h, METRIC, "after the liveness check");
    }

    @Test
    void theHostReplaysEverythingAboveTheHighWaterId() throws IOException {
        FakeHost h = new FakeHost(DIM, 10, EPOCH);
        h.commit(100);
        h.flush();
        TieredIndex t = persisted(h);
        t.close();
        h.commit(30);
        h.commit(10, 3);
        h.flush();

        TieredIndex u = open(h);
        assertNull(u.openFallbackCause());
        assertEquals(1, u.segmentCount());
        assertEquals(40, u.pendingTail(), "ids 101 to 140 come back into the tail");
        FakeHost.assertMatches(u, h, METRIC, "after the replay");
    }

    @Test
    void replayingTheSameBatchesAfterAReopenChangesNothing() throws IOException {
        FakeHost h = new FakeHost(DIM, 11, EPOCH);
        TieredIndex t = open(h);
        FakeHost.Txn t1 = h.commit(100);
        t.apply(FakeHost.batch(t1));
        h.flush();
        t.sealNow();
        t.checkpoint(dir);
        FakeHost.Txn t2 = h.commit(20, 2, 5);
        t.apply(FakeHost.batch(t2));
        h.flush();
        t.close();

        TieredIndex u = open(h);
        FakeHost.assertMatches(u, h, METRIC, "after the reopen");
        int size = u.size();
        // a host that starts its replay too early sends the batches again, in order
        u.apply(FakeHost.batch(t1));
        u.apply(FakeHost.batch(t2));
        u.apply(FakeHost.batch(t2));
        assertEquals(size, u.size());
        FakeHost.assertMatches(u, h, METRIC, "after the batches were applied again");
        u.sealNow();
        u.mergeNow();
        assertTrue(u.awaitIdle(60_000));
        u.checkpoint(dir);
        u.close();
        FakeHost.assertMatches(open(h), h, METRIC, "after a checkpoint and another reopen");
    }

    @ParameterizedTest
    @EnumSource(FaultHook.Step.class)
    void aCrashAtEveryStepOfAPublishLeavesAnIndexThatMatchesTheHost(FaultHook.Step stop) throws IOException {
        FakeHost h = new FakeHost(DIM, 12, EPOCH);
        final TieredIndex t = open(h);
        FakeHost.Txn t1 = h.commit(150);
        t.apply(FakeHost.batch(t1));
        h.flush();
        t.sealNow();
        t.checkpoint(dir); // generation 1
        FakeHost.Txn t2 = h.commit(30, 4, 9);
        t.apply(FakeHost.batch(t2));
        h.flush();
        t.sealNow();
        t.mergeNow();
        assertTrue(t.awaitIdle(60_000));
        assertThrows(IOException.class, () -> t.checkpoint(dir, step -> {
            if (step == stop) throw new IOException("crash after " + step);
        }));
        t.close();

        boolean published = stop.ordinal() >= FaultHook.Step.MANIFEST_RENAMED.ordinal();
        Manifest m = Manifest.readLatest(dir);
        assertEquals(published ? 2L : 1L, m.generation());
        TieredIndex u = open(h);
        assertNull(u.openFallbackCause());
        FakeHost.assertMatches(u, h, METRIC, "after a crash after " + stop);
        assertEquals(EPOCH, m.epoch());
        assertEquals(published ? 180L : 150L, m.hwm());
    }
}
