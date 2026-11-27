package ai.humem.tiervec;

import static ai.humem.tiervec.SegmentTestSupport.vectors;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.LongPredicate;
import java.util.stream.Collectors;
import java.util.stream.LongStream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/** The size hint of an allowed-id filter and the exact plan that it picks (issue #12). */
@Timeout(120)
class SizedIdFilterTest {
    private static final int DIM = 16;
    private static final int N = 1500;
    private static final int K = 10;

    /** a filter over a set of ids that reports {@code count}, right or wrong, and counts its calls */
    private static final class Filter implements SizedIdFilter {
        final LongPredicate allow;
        final long count;
        final AtomicInteger calls = new AtomicInteger();

        Filter(LongPredicate allow, long count) {
            this.allow = allow;
            this.count = count;
        }

        @Override
        public boolean test(long id) {
            calls.incrementAndGet();
            return allow.test(id);
        }

        @Override
        public long allowedCount() {
            return count;
        }
    }

    private static TieredIndex.Config cfg() {
        return new TieredIndex.Config(1_000_000, 0L, 50, 8, 4, 0.9, 0, 4, Quant.NONE, GraphIndex.Params.defaults().withEfConstruction(64), 1);
    }

    /** three segments with replaced and deleted ids, and a tail; {@code live} receives the current vector of every live id */
    private static TieredIndex tier(float[][] data, Map<Long, float[]> live) {
        TieredIndex t = new TieredIndex(DIM, Metric.L2, cfg());
        for (int i = 0; i < N; i++) {
            t.add(i, data[i]);
            live.put((long) i, data[i]);
            if (i == 499 || i == 999 || i == 1399) t.sealNow();
        }
        for (int i = 0; i < 200; i += 4) { // replaced: the old copy stays hidden in the first segment, the new one is in the tail
            t.add(i, data[N + i]);
            live.put((long) i, data[N + i]);
        }
        for (int i = 1; i < N; i += 9) {
            t.delete(i);
            live.remove((long) i);
        }
        assertEquals(3, t.segmentCount());
        return t;
    }

    /** brute force over the live ids that pass: squared L2, nearest first */
    private static List<Long> exact(Map<Long, float[]> live, float[] q, LongPredicate allow, int k) {
        Map<Long, Double> dist = new HashMap<>();
        for (Map.Entry<Long, float[]> e : live.entrySet()) {
            if (!allow.test(e.getKey())) continue;
            double d = 0;
            for (int j = 0; j < DIM; j++) {
                double x = q[j] - e.getValue()[j];
                d += x * x;
            }
            dist.put(e.getKey(), d);
        }
        return dist.keySet().stream().sorted(Comparator.comparing((Long id) -> dist.get(id))).limit(k).toList();
    }

    private static List<Long> ids(Hit[] hits) {
        List<Long> out = new ArrayList<>();
        for (Hit h : hits) out.add(h.id());
        return out;
    }

    @Test
    void aSmallAllowedSetIsScannedExactly() {
        float[][] data = vectors(2 * N + 20, DIM, 21);
        Map<Long, float[]> live = new HashMap<>();
        try (TieredIndex t = tier(data, live)) {
            assertEquals(live.size(), t.liveCount());
            Set<Long> allowed = LongStream.range(0, N).filter(i -> i % 25 == 0).boxed().collect(Collectors.toSet());
            for (int qi = 0; qi < 20; qi++) {
                float[] q = data[2 * N + qi];
                Filter f = new Filter(allowed::contains, allowed.size());
                TieredIndex.SearchResult r = t.search(q, K, f, TieredIndex.Freshness.FRESH);
                assertEquals(exact(live, q, allowed::contains, K), ids(r.hits()), "query " + qi);
                assertTrue(r.exact(), "the plan is exact");
                assertEquals(3, r.segments());
            }
        }
    }

    @Test
    void aWrongCountNeverGivesAWrongResult() {
        float[][] data = vectors(2 * N + 20, DIM, 22);
        Map<Long, float[]> live = new HashMap<>();
        try (TieredIndex t = tier(data, live)) {
            for (int qi = 0; qi < 20; qi++) {
                float[] q = data[2 * N + qi];
                // a count of 0 or 3 for a filter that allows every id: the scan finds the true top k
                for (long wrong : new long[] {0, 3}) {
                    assertEquals(exact(live, q, id -> true, K), ids(t.search(q, K, new Filter(id -> true, wrong))), "count " + wrong);
                }
                // a count that is too large for a filter that allows three ids: the walk or its fallback, never another id
                LongPredicate three = id -> id == 2 || id == 702 || id == 1202;
                for (long id : ids(t.search(q, K, new Filter(three, 1_000_000)))) assertTrue(three.test(id), "id " + id);
            }
        }
    }

    @Test
    void anEmptyAllowedSetReturnsNothingWithoutAWalk() {
        float[][] data = vectors(2 * N, DIM, 23);
        Map<Long, float[]> live = new HashMap<>();
        try (TieredIndex t = tier(data, live)) {
            Filter f = new Filter(id -> false, 0);
            TieredIndex.SearchResult r = t.search(data[0], K, f, TieredIndex.Freshness.FRESH);
            assertEquals(0, r.hits().length);
            assertTrue(r.exact());
            // a scan asks once for a row at most; the tail and the segments hold N + 50 rows, hidden ones included
            assertTrue(f.calls.get() <= N + 50, f.calls.get() + " calls");
        }
    }

    @Test
    void anUnknownOrLargeCountSearchesLikeAPlainPredicate() {
        float[][] data = vectors(2 * N + 20, DIM, 24);
        Map<Long, float[]> live = new HashMap<>();
        try (TieredIndex t = tier(data, live)) {
            LongPredicate half = id -> id % 2 == 0;
            for (int qi = 0; qi < 20; qi++) {
                float[] q = data[2 * N + qi];
                List<Long> plain = ids(t.search(q, K, half));
                assertEquals(plain, ids(t.search(q, K, new Filter(half, SizedIdFilter.UNKNOWN))));
                TieredIndex.SearchResult r = t.search(q, K, new Filter(half, N / 2), TieredIndex.Freshness.FRESH);
                assertEquals(plain, ids(r.hits()));
                assertTrue(!r.exact(), "a large set is walked");
            }
        }
    }

    @Test
    void theStandaloneGraphsScanASmallAllowedSetToo() {
        float[][] data = vectors(N + 20, DIM, 25);
        Map<Long, float[]> live = new HashMap<>();
        LongPredicate allow = id -> id % 30 == 7;
        try (GraphIndex g = new GraphIndex(DIM, Metric.L2, GraphIndex.Params.defaults().withEfConstruction(64))) {
            for (int i = 0; i < N; i++) {
                g.add(i, data[i]);
                live.put((long) i, data[i]);
            }
            assertEquals(N, g.liveCount());
            try (SealedGraph s = g.seal()) {
                for (int qi = 0; qi < 20; qi++) {
                    float[] q = data[N + qi];
                    List<Long> expected = exact(live, q, allow, K);
                    assertEquals(expected, ids(g.search(q, K, new Filter(allow, N / 30))), "graph, query " + qi);
                    assertEquals(expected, ids(s.search(q, K, new Filter(allow, N / 30))), "sealed graph, query " + qi);
                }
            }
        }
    }

    @Test
    void theTierSaysThatItPicksASelectivePlan() {
        assertTrue(TieredIndex.capabilities().contains(Capability.SELECTIVE_PLAN));
    }
}
