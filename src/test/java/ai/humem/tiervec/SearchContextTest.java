package ai.humem.tiervec;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Field;
import java.util.SplittableRandom;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.LongPredicate;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/**
 * {@link SearchContext} (issue #11): a search stops when its context says so, it polls the context at least once every {@link Poll#INTERVAL}
 * candidates, and while the context never expires it answers like a search without one.
 *
 * <p>The bounds are counted through the allow predicate, which the library calls once for every candidate of an exact scan, and once for every
 * node of a graph walk whose beam is as wide as the segment. No test depends on timing: the thread test waits for an event, never for a duration.
 */
@Timeout(value = 120, unit = TimeUnit.SECONDS)
class SearchContextTest {
    private static final int DIM = 24;
    private static final int INTERVAL = Poll.INTERVAL;
    private static final int NEVER = Integer.MAX_VALUE;

    /** only sealNow seals, nothing merges by itself, and the hard tail limit is out of reach */
    private static TieredIndex.Config manual(Quant tail) {
        TieredIndex.Config d = TieredIndex.Config.defaults();
        return new TieredIndex.Config(1 << 28, 0L, 1000, 1000, 4, 1.01, 0, 4, tail, d.graph(), 2);
    }

    private static void addRange(TieredIndex ix, int from, int to) {
        for (int i = from; i < to; i++) ix.add(i, TierT.vec(1, i, DIM));
    }

    private static float[] query(int i) {
        return TierT.vec(99, i, DIM);
    }

    /** expires at poll number {@code expireAt} (the first poll is 1), and counts the predicate calls between two polls */
    private static final class Counting implements SearchContext, LongPredicate {
        final int expireAt;
        final LongPredicate filter;
        int polls;
        long calls, atLastPoll, maxGap, minId = Long.MAX_VALUE;

        Counting(int expireAt) {
            this(expireAt, id -> true);
        }

        Counting(int expireAt, LongPredicate filter) {
            this.expireAt = expireAt;
            this.filter = filter;
        }

        @Override
        public boolean expired() {
            polls++;
            gap();
            atLastPoll = calls;
            return polls >= expireAt;
        }

        @Override
        public boolean test(long id) {
            calls++;
            minId = Math.min(minId, id);
            return filter.test(id);
        }

        long gap() {
            maxGap = Math.max(maxGap, calls - atLastPoll);
            return maxGap;
        }
    }

    private interface Search {
        Hit[] run(Counting c);
    }

    /** the search stops at the poll that expires, polls no more, and never looks at more than INTERVAL candidates between two polls */
    private static void assertStopsWithinTheInterval(LongPredicate filter, long candidatesAtLeast, Search search) {
        Counting full = new Counting(NEVER, filter);
        search.run(full);
        assertTrue(full.calls >= candidatesAtLeast, "the full search looked at " + full.calls + " candidates");
        assertTrue(full.gap() <= INTERVAL, full.maxGap + " candidates between two polls");
        assertTrue(full.polls >= candidatesAtLeast / INTERVAL, "the full search polled " + full.polls + " times");
        for (int expireAt : new int[] {1, 2, 3, 5, 8}) {
            Counting c = new Counting(expireAt, filter);
            assertThrows(SearchCancelledException.class, () -> search.run(c), "expiry at poll " + expireAt);
            assertEquals(expireAt, c.polls, "no poll after the one that expired");
            assertTrue(c.calls <= (long) (expireAt - 1) * INTERVAL, "expiry at poll " + expireAt + ": " + c.calls + " candidates");
            assertTrue(c.gap() <= INTERVAL, c.maxGap + " candidates between two polls");
            assertTrue(c.calls < full.calls, "the cancelled search did less work than the full one");
        }
    }

    @Test
    void theTailScanStopsWithinTheInterval() {
        for (Quant q : new Quant[] {Quant.NONE, Quant.SQ8}) {
            try (TieredIndex ix = new TieredIndex(DIM, Metric.COSINE, manual(q))) {
                addRange(ix, 0, 5000);
                assertStopsWithinTheInterval(id -> true, 5000, c -> ix.search(query(0), 10, c, c));
                // a filter that rejects almost everything: the candidates count, not only the distance computations
                assertStopsWithinTheInterval(id -> id % 1000 == 0, 5000, c -> ix.search(query(1), 10, c, c));
            }
        }
    }

    @Test
    void theGraphWalkStopsWithinTheInterval() {
        try (TieredIndex ix = new TieredIndex(DIM, Metric.COSINE, manual(Quant.NONE))) {
            addRange(ix, 0, 3000);
            ix.sealNow();
            // a beam as wide as the segment: every node that the walk reaches passes through the predicate
            assertStopsWithinTheInterval(id -> true, 2500, c -> ix.search(query(0), 10, c, 3000, c));
        }
    }

    @Test
    void theExactScanOfASegmentStopsWithinTheInterval() {
        try (TieredIndex ix = new TieredIndex(DIM, Metric.COSINE, manual(Quant.NONE))) {
            addRange(ix, 0, 3000);
            ix.sealNow();
            // few ids pass: the walk spends its visit budget and falls back to the scan
            assertStopsWithinTheInterval(id -> id % 1000 == 0, 3000, c -> ix.search(query(0), 10, c, 10, c));
            // the scan-first plan: a sample of the predicate, then the scan
            ix.filterScanBelow(1.0);
            assertStopsWithinTheInterval(id -> true, 3000, c -> ix.search(query(1), 10, c, c));
        }
    }

    @Test
    void theContextIsPolledBeforeEveryTailAndSegment() {
        try (TieredIndex ix = new TieredIndex(DIM, Metric.COSINE, manual(Quant.NONE))) {
            for (int s = 0; s < 4; s++) {
                addRange(ix, s * 100, (s + 1) * 100);
                ix.sealNow();
            }
            addRange(ix, 400, 500);
            assertEquals(4, ix.stats().segments());

            Counting full = new Counting(NEVER);
            ix.search(query(0), 10, full, full);
            assertEquals(6, full.polls, "the start, the tail, and four segments");

            Counting c = new Counting(4); // the start, the tail, the newest segment, and then it expires
            assertThrows(SearchCancelledException.class, () -> ix.search(query(0), 10, c, c));
            assertEquals(4, c.polls);
            assertTrue(c.minId >= 300, "an older segment was read: id " + c.minId);
        }
    }

    @Test
    void aContextThatNeverExpiresChangesNoAnswer() {
        for (Quant q : new Quant[] {Quant.NONE, Quant.SQ8}) {
            try (TieredIndex ix = new TieredIndex(DIM, Metric.COSINE, manual(q))) {
                for (int s = 0; s < 3; s++) {
                    addRange(ix, s * 1500, (s + 1) * 1500);
                    ix.sealNow();
                }
                addRange(ix, 4500, 5100);
                for (int id = 0; id < 5100; id += 17) ix.delete(id); // hidden copies: the segments are searched through the floor
                for (int id = 5; id < 5100; id += 101) ix.add(id, TierT.vec(2, id, DIM));

                LongPredicate[] filters = {null, id -> true, id -> id % 2 == 0, id -> id % 10 == 3, id -> id % 100 == 7, id -> false};
                SplittableRandom r = new SplittableRandom(11);
                SearchContext never = () -> false;
                for (double scanBelow : new double[] {0.0, 0.05, 1.0}) {
                    ix.filterScanBelow(scanBelow);
                    for (int i = 0; i < 25; i++) {
                        float[] qv = query(r.nextInt(1_000_000));
                        int k = 1 + r.nextInt(40);
                        int ef = 10 + r.nextInt(200);
                        for (int f = 0; f < filters.length; f++) {
                            LongPredicate p = filters[f];
                            String where = "tail " + q + ", scanBelow " + scanBelow + ", query " + i + ", filter " + f;
                            Hit[] plain = ix.search(qv, k, p);
                            Hit[] wide = ix.search(qv, k, p, ef);
                            TieredIndex.SearchResult sealed = ix.search(qv, k, p, TieredIndex.Freshness.SEALED_ONLY);
                            TieredIndex.SearchResult fresh = ix.search(qv, k, p, TieredIndex.Freshness.FRESH);

                            // a cancelled search leaves nothing behind that a later one could see
                            try {
                                ix.search(qv, k, p, ef, new Counting(1 + r.nextInt(12)));
                            } catch (SearchCancelledException expected) {
                                // most of them stop, some finish first
                            }

                            assertArrayEquals(plain, ix.search(qv, k, p, never), where);
                            assertArrayEquals(wide, ix.search(qv, k, p, ef, never), where + ", ef " + ef);
                            assertSameResult(sealed, ix.search(qv, k, p, TieredIndex.Freshness.SEALED_ONLY, never), where + ", sealed only");
                            assertSameResult(fresh, ix.search(qv, k, p, TieredIndex.Freshness.FRESH, never), where + ", fresh");
                            assertArrayEquals(plain, ix.search(qv, k, p), where + ", without a context again");
                        }
                    }
                }
            }
        }
    }

    private static void assertSameResult(TieredIndex.SearchResult a, TieredIndex.SearchResult b, String where) {
        assertArrayEquals(a.hits(), b.hits(), where);
        assertEquals(a.exact(), b.exact(), where);
        assertEquals(a.tailScanned(), b.tailScanned(), where);
        assertEquals(a.segments(), b.segments(), where);
    }

    @Test
    void aFlagSetByAnotherThreadStopsASearchThatRunsInALoop() throws Exception {
        try (TieredIndex ix = new TieredIndex(DIM, Metric.COSINE, manual(Quant.NONE))) {
            addRange(ix, 0, 2000);
            ix.sealNow();
            addRange(ix, 2000, 2500);
            AtomicBoolean cancel = new AtomicBoolean();
            AtomicLong finished = new AtomicLong();
            CountDownLatch looping = new CountDownLatch(3);
            AtomicReference<Throwable> end = new AtomicReference<>();
            Thread t = new Thread(() -> {
                try {
                    for (;;) {
                        ix.search(query(0), 10, null, cancel::get);
                        finished.incrementAndGet();
                        looping.countDown();
                    }
                } catch (Throwable e) {
                    end.set(e);
                }
            }, "search-loop");
            t.setDaemon(true);
            t.start();
            assertTrue(looping.await(30, TimeUnit.SECONDS), "the loop did not start");
            cancel.set(true);
            t.join(30_000);
            assertFalse(t.isAlive(), "the loop was not stopped after " + finished.get() + " searches");
            assertTrue(end.get() instanceof SearchCancelledException, String.valueOf(end.get()));
        }
    }

    @Test
    void aFlagSetWhileASearchIsInsideAScanStopsThatSearch() throws Exception {
        try (TieredIndex ix = new TieredIndex(DIM, Metric.COSINE, manual(Quant.NONE))) {
            addRange(ix, 0, 5000);
            AtomicBoolean cancel = new AtomicBoolean();
            AtomicLong calls = new AtomicLong();
            CountDownLatch inside = new CountDownLatch(1);
            CountDownLatch resume = new CountDownLatch(1);
            LongPredicate hold = id -> {
                if (calls.incrementAndGet() == 1) {
                    inside.countDown();
                    try {
                        resume.await(30, TimeUnit.SECONDS);
                    } catch (InterruptedException e) {
                        throw new AssertionError(e);
                    }
                }
                return true;
            };
            AtomicReference<Object> end = new AtomicReference<>();
            Thread t = new Thread(() -> {
                try {
                    end.set(ix.search(query(0), 10, hold, cancel::get));
                } catch (Throwable e) {
                    end.set(e);
                }
            }, "held-search");
            t.setDaemon(true);
            t.start();
            assertTrue(inside.await(30, TimeUnit.SECONDS), "the search did not reach the predicate");
            cancel.set(true);
            resume.countDown();
            t.join(30_000);
            assertFalse(t.isAlive());
            assertTrue(end.get() instanceof SearchCancelledException, String.valueOf(end.get()));
            assertTrue(calls.get() <= INTERVAL, calls.get() + " candidates after the flag was set");
        }
    }

    @Test
    void aSearchThatWaitsForABatchStopsToo() throws Exception {
        try (TieredIndex ix = new TieredIndex(DIM, Metric.COSINE, manual(Quant.NONE))) {
            addRange(ix, 0, 100);
            // an odd batch counter is how a search sees a batch in progress; here the batch never ends
            Field f = TieredIndex.class.getDeclaredField("batchSeq");
            f.setAccessible(true);
            long before = f.getLong(ix);
            f.setLong(ix, before + 1);
            try {
                Counting c = new Counting(4);
                assertThrows(SearchCancelledException.class, () -> ix.search(query(0), 10, c, c));
                assertEquals(4, c.polls);
                assertEquals(0, c.calls, "nothing is scanned while the batch is open");
            } finally {
                f.setLong(ix, before);
            }
            assertEquals(10, ix.search(query(0), 10, null, () -> false).length);
        }
    }

    @Test
    void aStoppedSearchLeavesTheIndexAsItWas() {
        try (TieredIndex ix = new TieredIndex(DIM, Metric.COSINE, manual(Quant.NONE))) {
            addRange(ix, 0, 1200);
            ix.sealNow();
            addRange(ix, 1200, 2400);
            ix.sealNow();
            addRange(ix, 2400, 3000);
            for (int expireAt = 1; expireAt <= 10; expireAt++) {
                Counting c = new Counting(expireAt);
                assertThrows(SearchCancelledException.class, () -> ix.search(query(0), 10, c, 1200, c));
            }
            // an exception of the context itself reaches the caller unchanged
            IllegalStateException host = new IllegalStateException("from the host");
            AtomicLong polls = new AtomicLong();
            SearchContext broken = () -> {
                if (polls.incrementAndGet() == 3) throw host;
                return false;
            };
            assertSame(host, assertThrows(IllegalStateException.class, () -> ix.search(query(0), 10, null, broken)));
            // a search that did not leave its view would keep the floor cleanup of this merge pending for ever
            for (int id = 0; id < 100; id++) ix.delete(id);
            ix.mergeNow();
            assertTrue(ix.awaitIdle(30_000), "a stopped search is still counted as a reader");
            assertEquals(0, ix.stats().floorEntries());
        }
    }

    @Test
    void argumentsAreCheckedAsWithoutAContext() {
        try (TieredIndex ix = new TieredIndex(DIM, Metric.COSINE, manual(Quant.NONE))) {
            addRange(ix, 0, 100);
            SearchContext expired = () -> true;
            assertEquals(0, ix.search(query(0), 0, null, expired).length, "k <= 0 asks for nothing");
            assertThrows(IllegalArgumentException.class, () -> ix.search(new float[DIM + 1], 10, null, expired));
            assertArrayEquals(ix.search(query(0), 10, null), ix.search(query(0), 10, null, (SearchContext) null), "null is no context");
        }
        try (TieredIndex empty = new TieredIndex(DIM, Metric.COSINE, manual(Quant.NONE))) {
            assertThrows(SearchCancelledException.class, () -> empty.search(query(0), 10, null, () -> true), "the context is polled at the start");
        }
    }
}
