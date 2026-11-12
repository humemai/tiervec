package ai.humem.tiervec;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.LongPredicate;
import org.junit.jupiter.api.Test;

/**
 * {@link TieredIndex#apply} and {@link TieredIndex#awaitCapacity} (issue #10). A host transaction is an ordered batch of adds and deletes, and an update is an add
 * of a new id plus a delete of the old id. A search must see the batch whole: both ids of an update never, and neither id never.
 *
 * <p>The atomicity tests hold a search inside its first call of the allow predicate, apply a batch on the test thread, and let the search go on, the way
 * {@code RebuildSwapRaceTest} does. A search that was running when the batch was applied may answer from the state before the batch or from the state after it, and
 * never from a mixture. The test needs no timing: the held search is stopped at a place that is known, and the batch is applied while it waits. The threaded test
 * of the same property is {@code ApplyStressTest} (slow).
 */
class ApplyBatchTest {
    static final int DIM = 32;
    static final long SEED = 3;

    static float[] vec(long id) {
        return TierT.vec(SEED, id, DIM);
    }

    static long[] range(long from, long toExclusive) {
        long[] r = new long[(int) (toExclusive - from)];
        for (int i = 0; i < r.length; i++) r[i] = from + i;
        return r;
    }

    static TieredIndex.Batch batch(long[] adds, long... deletes) {
        List<TieredIndex.IdVector> a = new ArrayList<>();
        for (long id : adds) a.add(new TieredIndex.IdVector(id, vec(id)));
        List<Long> d = new ArrayList<>();
        for (long id : deletes) d.add(id);
        return new TieredIndex.Batch(a, d);
    }

    static Set<Long> idsOf(Hit[] hits) {
        Set<Long> s = new HashSet<>();
        for (Hit h : hits) s.add(h.id());
        return s;
    }

    /** only sealNow and mergeNow seal and merge, nothing is throttled, and the hard tail limit is out of reach */
    static TieredIndex.Config manual() {
        TieredIndex.Config d = TieredIndex.Config.defaults();
        return new TieredIndex.Config(1 << 28, 0L, 1000, 1000, 4, 1.01, 0, 4, d.tailQuant(), d.graph(), 2);
    }

    /** a search that stops inside its first call of the allow predicate for an id that {@code holdOn} accepts, and goes on when told to */
    private static final class HeldSearch {
        final CountDownLatch inside = new CountDownLatch(1);
        final CountDownLatch resume = new CountDownLatch(1);
        final AtomicBoolean first = new AtomicBoolean();
        final AtomicReference<Hit[]> result = new AtomicReference<>();
        final AtomicReference<Throwable> error = new AtomicReference<>();
        final Thread thread;

        HeldSearch(TieredIndex ix, float[] q, LongPredicate holdOn, LongPredicate show, String name) {
            LongPredicate allow = id -> {
                if (holdOn.test(id) && first.compareAndSet(false, true)) {
                    inside.countDown();
                    try {
                        resume.await(60, TimeUnit.SECONDS);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }
                return show.test(id);
            };
            thread = new Thread(() -> {
                try {
                    result.set(ix.search(q, 50, allow, 200));
                } catch (Throwable t) {
                    error.set(t);
                }
            }, name);
            thread.setDaemon(true);
            thread.start();
        }

        Hit[] finish() throws InterruptedException {
            resume.countDown();
            thread.join(60_000);
            assertFalse(thread.isAlive(), "the search did not finish");
            assertNull(error.get(), "the search failed");
            assertNotNull(result.get(), "the search returned nothing");
            return result.get();
        }
    }

    private static void assertOneOf(Hit[] hits, long before, long after, String where) {
        Set<Long> ids = idsOf(hits);
        boolean old = ids.equals(Set.of(before));
        boolean updated = ids.equals(Set.of(after));
        assertTrue(old || updated, "a search that was running while a batch (add " + after + ", delete " + before + ") was applied must see the state before the batch"
                + " or the state after it, and it saw " + ids + " (" + where + ")");
    }

    @Test
    void theWholeBatchIsVisibleAfterApplyReturns() {
        TieredIndex ix = new TieredIndex(DIM, Metric.COSINE, manual());
        try {
            for (long id = 0; id < 50; id++) ix.add(id, vec(id));
            ix.apply(batch(new long[] {100, 101}, 0, 1));
            Set<Long> expect = new HashSet<>();
            for (long id = 2; id < 50; id++) expect.add(id);
            expect.add(100L);
            expect.add(101L);
            assertEquals(expect, idsOf(ix.search(vec(100), 200, null)), "the adds are visible and the deletes are applied");
            assertEquals(100L, ix.search(vec(100), 1, null)[0].id(), "an added vector is its own best hit");
            assertEquals(50, ix.size());
            assertEquals(50, ix.stats().live());

            // adds are applied before deletes: an id that is added and deleted by one batch ends deleted
            ix.apply(batch(new long[] {200}, 200));
            assertFalse(idsOf(ix.search(vec(200), 200, null)).contains(200L), "added and deleted in one batch: deleted");
            assertEquals(50, ix.size());
            assertEquals(50, ix.stats().live());

            // the delete of an absent id and an empty batch change nothing
            ix.apply(batch(new long[0], 12345));
            ix.apply(batch(new long[0]));
            assertEquals(expect, idsOf(ix.search(vec(100), 200, null)));
            assertEquals(50, ix.size());
        } finally {
            ix.close();
        }
    }

    @Test
    void aBatchWithABadVectorChangesNothing() {
        TieredIndex ix = new TieredIndex(DIM, Metric.COSINE, manual());
        try {
            for (long id = 0; id < 10; id++) ix.add(id, vec(id));
            List<TieredIndex.IdVector> adds = List.of(new TieredIndex.IdVector(100, vec(100)), new TieredIndex.IdVector(101, new float[DIM + 1]));
            TieredIndex.Batch bad = new TieredIndex.Batch(adds, List.of(0L));
            assertThrows(IllegalArgumentException.class, () -> ix.apply(bad));
            assertEquals(10, ix.size(), "a rejected batch applies none of its changes");
            Set<Long> ids = idsOf(ix.search(vec(100), 50, null));
            assertFalse(ids.contains(100L), "the add that came before the bad vector was not applied");
            assertTrue(ids.contains(0L), "the delete after the bad vector was not applied");
        } finally {
            ix.close();
        }
    }

    @Test
    void applyingTheSameBatchAgainChangesNothingVisible() {
        TieredIndex ix = new TieredIndex(DIM, Metric.COSINE, manual());
        try {
            for (long id = 0; id < 1000; id++) ix.add(id, vec(id));
            ix.sealNow();
            for (long id = 1000; id < 1020; id++) ix.add(id, vec(id));
            assertEquals(1, ix.segmentCount());

            // five adds of new ids, three deletes in the segment, two deletes in the tail
            TieredIndex.Batch b = batch(range(2000, 2005), 0, 1, 2, 1000, 1001);
            ix.apply(b);
            long[] queries = {0, 1, 5, 500, 1000, 1010, 2000, 2004};
            Hit[][] once = new Hit[queries.length][];
            for (int i = 0; i < queries.length; i++) once[i] = ix.search(vec(queries[i]), 10, null, 200);
            int size1 = ix.size(), live1 = ix.stats().live(), floor1 = ix.stats().floorEntries();
            assertEquals(1020, size1, "20 in the tail and 1000 in the segment, 5 added, 5 deleted");
            assertEquals(size1, live1);

            ix.apply(b);
            assertEquals(size1, ix.size(), "re-applying must not change size()");
            assertEquals(live1, ix.stats().live(), "re-applying must not change stats().live()");
            assertEquals(floor1, ix.stats().floorEntries(), "re-applying must not write floor entries");
            for (int i = 0; i < queries.length; i++) {
                assertArrayEquals(once[i], ix.search(vec(queries[i]), 10, null, 200), "query " + queries[i] + " answers the same after the batch is applied again");
            }

            // the same once more after a seal moved the added ids into a segment: the answer is the same set, and nothing is counted twice
            ix.sealNow();
            ix.apply(b);
            assertEquals(size1, ix.size());
            assertEquals(live1, ix.stats().live());
            Set<Long> gone = Set.of(0L, 1L, 2L, 1000L, 1001L);
            assertTrue(idsOf(ix.search(vec(0), 50, gone::contains, 200)).isEmpty(), "the deleted ids stay deleted");
            assertEquals(Set.of(2000L, 2001L, 2002L, 2003L, 2004L), idsOf(ix.search(vec(2000), 50, id -> id >= 2000 && id < 2005, 200)), "each added id once");
        } finally {
            ix.close();
        }
    }

    @Test
    void aSearchHeldInTheTailSeesTheBatchWholeOrNotAtAll() throws Exception {
        final long a = 150, b = 6000; // slot 150 lies in a later word of the live bits than slot 0, where the search is held
        TieredIndex ix = new TieredIndex(DIM, Metric.COSINE, manual());
        try {
            for (long id = 0; id < 200; id++) ix.add(id, vec(id));
            HeldSearch held = new HeldSearch(ix, vec(a), id -> true, id -> id == a || id == b, "apply-held-in-tail");
            assertTrue(held.inside.await(60, TimeUnit.SECONDS), "the search never called the allow predicate");

            ix.apply(batch(new long[] {b}, a)); // an update: the new id comes, the old id goes

            assertOneOf(held.finish(), a, b, "held in the tail scan");
            assertEquals(Set.of(b), idsOf(ix.search(vec(a), 50, id -> id == a || id == b, 200)), "a search that starts after apply returned sees the adds and not the deletes");
        } finally {
            ix.close();
        }
    }

    @Test
    void aSearchHeldInASegmentSeesTheBatchWholeOrNotAtAll() throws Exception {
        final long a = 500, b = 6000;
        TieredIndex ix = new TieredIndex(DIM, Metric.COSINE, manual());
        try {
            for (long id = 0; id < 1000; id++) ix.add(id, vec(id));
            ix.sealNow();
            assertTrue(ix.delete(7), "the segment holds a hidden copy, so searches filter it through the floor while they scan");
            for (long id = 5000; id < 5010; id++) ix.add(id, vec(id)); // a tail that is not empty, so that the search scans it before the segment
            assertEquals(1, ix.segmentCount());

            HeldSearch held = new HeldSearch(ix, vec(a), id -> id < 1000, id -> id == a || id == b, "apply-held-in-segment");
            assertTrue(held.inside.await(60, TimeUnit.SECONDS), "the search never called the allow predicate inside the segment");

            ix.apply(batch(new long[] {b}, a));

            assertOneOf(held.finish(), a, b, "held in the segment scan, after the tail scan");
            assertEquals(Set.of(b), idsOf(ix.search(vec(a), 50, id -> id == a || id == b, 200)), "a search that starts after apply returned sees the adds and not the deletes");
        } finally {
            ix.close();
        }
    }

    @Test
    void applyDoesNotWaitForTheWorker() throws Exception {
        TieredIndex.Config d = TieredIndex.Config.defaults();
        TieredIndex.Config cfg = new TieredIndex.Config(50, 0L, 10, 100, 4, 1.01, 0, 4, d.tailQuant(), d.graph(), 1);
        TieredIndex ix = new TieredIndex(DIM, Metric.COSINE, cfg);
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        try {
            ix.setSealHooks(() -> {
                entered.countDown();
                try {
                    release.await(120, TimeUnit.SECONDS);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }, null);
            ix.apply(batch(range(0, 60))); // reaches tailLimit: the worker freezes the tail, builds a segment, and stops in the hook, holding the work lock
            assertTrue(entered.await(60, TimeUnit.SECONDS), "the worker never reached the seal hook");

            AtomicReference<Throwable> error = new AtomicReference<>();
            Thread t = new Thread(() -> {
                try {
                    ix.apply(batch(range(100, 110), 0, 1, 2));
                } catch (Throwable e) {
                    error.set(e);
                }
            }, "apply-while-sealing");
            t.setDaemon(true);
            t.start();
            t.join(10_000);
            assertFalse(t.isAlive(), "apply waited for the worker while a seal was running");
            assertNull(error.get(), "apply failed");
            assertEquals(67, ix.size());
        } finally {
            release.countDown();
            ix.close();
        }
    }

    @Test
    void awaitCapacityReturnsTrueWhenThereIsRoom() {
        TieredIndex ix = new TieredIndex(DIM, Metric.COSINE, manual());
        try {
            assertTrue(ix.awaitCapacity(System.nanoTime() + 1_000_000_000L), "an empty tail has room");
            assertTrue(ix.awaitCapacity(System.nanoTime() - 1L), "a deadline that has passed still reports room when there is room");
        } finally {
            ix.close();
        }
    }

    /**
     * The behavior at the hard tail limit (hardTailFactor * max(tailLimit, smallLimit) = 40 here): awaitCapacity waits for room, and returns false at its
     * deadline; apply does not look at the limit, it never waits, and one batch may take the tail above it; awaitCapacity says true again as soon as deletes
     * or a seal have made room, also for a call that is already waiting. The limit is reached while the worker is held inside a seal, which is how a writer
     * outruns the worker.
     */
    @Test
    void behaviorAtTheHardTailLimit() throws Exception {
        TieredIndex.Config d = TieredIndex.Config.defaults();
        TieredIndex.Config cfg = new TieredIndex.Config(20, 0L, 10, 100, 4, 1.01, 0, 2, d.tailQuant(), d.graph(), 1);
        TieredIndex ix = new TieredIndex(DIM, Metric.COSINE, cfg);
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        try {
            ix.setSealHooks(() -> {
                entered.countDown();
                try {
                    release.await(120, TimeUnit.SECONDS);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }, null);
            ix.apply(batch(range(0, 25)));
            assertTrue(entered.await(60, TimeUnit.SECONDS), "the worker never reached the seal hook");
            assertTrue(ix.awaitCapacity(System.nanoTime() + 1_000_000_000L), "the new active tail is empty");

            ix.apply(batch(range(100, 145))); // 45 vectors in one batch: above the limit of 40 in one call
            long t0 = System.nanoTime();
            boolean room = ix.awaitCapacity(t0 + 50_000_000L);
            long waitedMs = (System.nanoTime() - t0) / 1_000_000L;
            assertFalse(room, "the tail holds 45 vectors, above the hard limit of 40, and the worker is held");
            assertTrue(waitedMs >= 45, "awaitCapacity must wait for its deadline before it says false, it waited " + waitedMs + " ms");

            AtomicReference<Throwable> error = new AtomicReference<>();
            Thread t = new Thread(() -> {
                try {
                    ix.apply(batch(range(200, 205)));
                } catch (Throwable e) {
                    error.set(e);
                }
            }, "apply-at-hard-limit");
            t.setDaemon(true);
            t.start();
            t.join(5_000);
            assertFalse(t.isAlive(), "apply waited at the hard tail limit");
            assertNull(error.get(), "apply failed at the hard tail limit");
            assertEquals(Set.of(200L, 201L, 202L, 203L, 204L), idsOf(ix.search(vec(200), 50, id -> id >= 200 && id < 205, 200)), "the batch above the limit is visible");
            assertEquals(75, ix.size());

            ix.apply(batch(new long[0], range(100, 120))); // 50 vectors in the active tail, 20 deleted: 30, room again below 40
            t0 = System.nanoTime();
            assertTrue(ix.awaitCapacity(t0 + 5_000_000_000L), "deletes made room");
            assertTrue((System.nanoTime() - t0) / 1_000_000L < 2_000, "a call with room returns at once");

            ix.apply(batch(range(300, 320))); // full again
            assertFalse(ix.awaitCapacity(System.nanoTime() + 20_000_000L));
            AtomicReference<Boolean> got = new AtomicReference<>();
            CountDownLatch started = new CountDownLatch(1);
            Thread waiter = new Thread(() -> {
                started.countDown();
                got.set(ix.awaitCapacity(System.nanoTime() + 60_000_000_000L));
            }, "await-capacity");
            waiter.setDaemon(true);
            waiter.start();
            assertTrue(started.await(60, TimeUnit.SECONDS));
            Thread.sleep(100);
            ix.apply(batch(new long[0], range(300, 320)));
            waiter.join(20_000);
            assertFalse(waiter.isAlive(), "a call that is waiting must return when deletes make room");
            assertEquals(Boolean.TRUE, got.get());
        } finally {
            release.countDown();
            ix.close();
        }
    }
}
