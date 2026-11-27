package ai.humem.tiervec;

import static ai.humem.tiervec.SegmentTestSupport.vectors;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.LongPredicate;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/** The partial result of a stopped search, on request only (issue #134). */
@Timeout(120)
class PartialResultTest {
    private static final int DIM = 16;
    private static final int K = 10;
    private static final int TAIL_FROM = 900;
    private static final int N = 950;

    /** three segments of 300 rows, and the ids from {@code TAIL_FROM} in the tail */
    private static TieredIndex tier(float[][] data) {
        TieredIndex.Config c = new TieredIndex.Config(1_000_000, 0L, 50, 8, 4, 0.9, 0, 4, Quant.NONE,
                GraphIndex.Params.defaults().withEfConstruction(64), 1);
        TieredIndex t = new TieredIndex(DIM, Metric.L2, c);
        for (int i = 0; i < N; i++) {
            t.add(i, data[i]);
            if (i == 299 || i == 599 || i == TAIL_FROM - 1) t.sealNow();
        }
        assertEquals(3, t.segmentCount());
        return t;
    }

    private static SearchContext context(AtomicBoolean stop, boolean partial) {
        return new SearchContext() {
            @Override
            public boolean expired() {
                return stop.get();
            }

            @Override
            public boolean partialWanted() {
                return partial;
            }
        };
    }

    /** a filter that allows what {@code allow} allows and sets {@code stop} at the first id below {@code below} that it is asked about */
    private static LongPredicate stopBelow(long below, AtomicBoolean stop, LongPredicate allow) {
        return id -> {
            if (id < below) stop.set(true);
            return allow.test(id);
        };
    }

    private static SearchCancelledException stopped(TieredIndex t, float[] q, LongPredicate allow, SearchContext context) {
        return assertThrows(SearchCancelledException.class, () -> t.search(q, K, allow, context));
    }

    private static List<String> text(Hit[] hits) {
        List<String> out = new ArrayList<>();
        for (Hit h : hits) out.add(h.id() + " " + h.score());
        return out;
    }

    @Test
    void aStopAfterTheTailReturnsTheBestHitsOfTheTail() {
        float[][] data = vectors(N + 10, DIM, 31);
        try (TieredIndex t = tier(data)) {
            for (int qi = 0; qi < 10; qi++) {
                float[] q = data[N + qi];
                AtomicBoolean stop = new AtomicBoolean();
                // the tail is read first: the first call below N stops the search inside it, and the tail is read to its end before the next poll
                SearchCancelledException e = stopped(t, q, stopBelow(N, stop, id -> true), context(stop, true));
                assertEquals(text(t.search(q, K, id -> id >= TAIL_FROM, TieredIndex.Freshness.FRESH).hits()), text(e.partialHits()), "query " + qi);
                assertEquals(K, e.partialHits().length);
            }
        }
    }

    @Test
    void aStopLaterAddsTheSegmentsThatWereReadToTheEnd() {
        float[][] data = vectors(N + 10, DIM, 32);
        try (TieredIndex t = tier(data)) {
            assertTrue(t.delete(950 - 1)); // a deleted id of the tail and one of the newest segment must not come back
            assertTrue(t.delete(700));
            for (int qi = 0; qi < 10; qi++) {
                float[] q = data[N + qi];
                AtomicBoolean stop = new AtomicBoolean();
                // the first id below TAIL_FROM belongs to the newest segment, which is read to its end; the next segment is never started
                SearchCancelledException e = stopped(t, q, stopBelow(TAIL_FROM, stop, id -> true), context(stop, true));
                Hit[] partial = e.partialHits();
                assertEquals(text(t.search(q, K, id -> id >= 600)), text(partial), "query " + qi);
                for (Hit h : partial) assertTrue(h.id() >= 600 && h.id() != 700 && h.id() != 949, "id " + h.id());
            }
        }
    }

    @Test
    void withoutTheRequestAStoppedSearchCarriesNoHits() {
        float[][] data = vectors(N + 1, DIM, 33);
        try (TieredIndex t = tier(data)) {
            AtomicBoolean stop = new AtomicBoolean();
            assertEquals(0, stopped(t, data[N], stopBelow(TAIL_FROM, stop, id -> true), context(stop, false)).partialHits().length);
            stop.set(false);
            // a lambda context never asks
            assertEquals(0, stopped(t, data[N], stopBelow(TAIL_FROM, stop, id -> true), stop::get).partialHits().length);
        }
    }

    @Test
    void aStopBeforeAnySourceWasReadCarriesNoHits() {
        float[][] data = vectors(N + 1, DIM, 34);
        try (TieredIndex t = tier(data)) {
            AtomicBoolean stop = new AtomicBoolean(true);
            assertEquals(0, stopped(t, data[N], null, context(stop, true)).partialHits().length);
        }
    }

    @Test
    void aBatchDuringTheScanLeavesThePartialResultOfItsVersion() {
        float[][] data = vectors(N + 1, DIM, 35);
        try (TieredIndex t = tier(data)) {
            AtomicBoolean stop = new AtomicBoolean();
            AtomicInteger batches = new AtomicInteger();
            // the best hits of the tail and the newest segment, which are the two sources that the stopped search reads to their end
            Hit[] before = t.search(data[N], K, id -> id >= 600);
            List<Long> gone = new ArrayList<>();
            for (Hit h : before) gone.add(h.id());
            LongPredicate batchThenStop = id -> {
                if (id < TAIL_FROM) {
                    if (batches.getAndIncrement() == 0) {
                        Thread writer = new Thread(() -> t.apply(new TieredIndex.Batch(List.of(), gone))); // while the segment is scanned
                        writer.start();
                        try {
                            writer.join();
                        } catch (InterruptedException e) {
                            throw new IllegalStateException(e);
                        }
                    }
                    stop.set(true);
                }
                return true;
            };
            SearchCancelledException e = stopped(t, data[N], batchThenStop, context(stop, true));
            assertEquals(N - K, t.size(), "the batch was applied");
            assertEquals(text(before), text(e.partialHits()), "the partial result answers from the state at the start of the search");
        }
    }

    @Test
    void anExceptionOfTheContextItselfPassesUnchanged() {
        float[][] data = vectors(N + 1, DIM, 36);
        try (TieredIndex t = tier(data)) {
            SearchCancelledException own = new SearchCancelledException("mine");
            AtomicInteger polls = new AtomicInteger();
            SearchContext c = new SearchContext() {
                @Override
                public boolean expired() {
                    if (polls.incrementAndGet() > 2) throw own;
                    return false;
                }

                @Override
                public boolean partialWanted() {
                    return true;
                }
            };
            assertTrue(stopped(t, data[N], null, c) == own);
            assertEquals(0, own.partialHits().length);
        }
    }
}
