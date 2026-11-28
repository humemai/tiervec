package ai.humem.tiervec;

import java.util.Arrays;
import java.util.function.LongPredicate;

/**
 * Two filter-aware walks of a sealed segment, for the filter screen (docs/SCREEN-IDEAS.md, ideas S7 and S8, issue #162). Benchmark code: the
 * library does not use it and nothing here changes how a graph is built. The rules, the stop, the budget and the scan that follows a walk
 * are those that the document registered before this class was written.
 *
 * <p>The upper layers are searched as {@link SealedGraph} searches them, without the filter. On layer 0 the node reached there is a candidate
 * whether it passes or not; apart from it only rows that pass become candidates. A walk that ends with fewer than {@code ef} rows collected
 * is followed by the exact scan of the library. One instance is for one thread. It reads an fp32 graph only.
 */
final class FilterWalk {
    /** {@link #ACORN}: after Lucene's FilteredHnswGraphSearcher (S7). {@link #NAVIX}: the adaptive-local rule of NaviX (S8). */
    enum Rule { ACORN, NAVIX }

    /** the hits, and whether an exact scan answered */
    record Result(Hit[] hits, boolean scanned) {}

    /** Lucene's default: at this share of passing rows or above, the filtered walk is not used */
    private static final float ACORN_BELOW = 0.60f;
    /** Lucene: second-hop neighbors are read only when more than this share of the neighbors fail */
    private static final float ACORN_EXPAND_ABOVE = 0.10f;
    /** NaviX: at this local share or above, only the passing neighbors are scored */
    private static final float NAVIX_ONE_HOP = 0.5f;
    private static final int NAVIX_LENIENCY = 3;
    private static final double PHI = 0.6180339887498949;

    private final SealedGraph g;
    private final Rule rule;
    private final int m, d; // M of the graph, and the degree limit of layer 0
    private final float[] scratch;
    private final Heap cand = new Heap(), res = new Heap();
    // per walk, valid where the stamp equals gen
    private final int[] seen;       // ACORN: looked at. NAVIX: scored and offered
    private final int[] askedAt;    // the predicate was asked about the row
    private final boolean[] passed; // and this was its answer
    private final int[] distAt;     // NAVIX: the distance of the row was computed
    private final float[] distOf;
    private int gen;

    private float[] q;
    private LongPredicate allow;
    private int ef;
    private int computed; // distances on layer 0

    // ACORN, set per query from the sampled share
    private int multiplier, minToFind;
    private int[] toScore = new int[0], toExplore = new int[0];
    // NAVIX
    private final int[] selected, order;
    private final float[] orderD;

    FilterWalk(SealedGraph g, Rule rule) {
        if (g.codes != null && g.params.sq8()) throw new IllegalArgumentException("the filter-aware walks read fp32 graphs only");
        this.g = g;
        this.rule = rule;
        this.m = g.params.M();
        this.d = 2 * m;
        this.scratch = g.vectors.newScratch();
        seen = new int[g.n];
        askedAt = new int[g.n];
        passed = new boolean[g.n];
        distAt = new int[g.n];
        distOf = new float[g.n];
        int widest = 0;
        for (int r = 0; r < g.n; r++) widest = Math.max(widest, g.off0[r + 1] - g.off0[r]);
        if (widest > d) throw new IllegalStateException("a layer-0 list of " + widest + " entries is longer than the limit " + d);
        selected = new int[d];
        order = new int[d];
        orderD = new float[d];
    }

    Result search(float[] query, int k, LongPredicate filter, int beam) {
        if (filter == null) throw new IllegalArgumentException("a filter-aware walk needs a filter");
        if (query.length != g.dim) throw new IllegalArgumentException("dim");
        if (k <= 0 || g.n == 0) return new Result(new Hit[0], false);
        ef = Math.max(beam, k);
        if (FilterPlan.smallByHint(filter, ef)) return new Result(g.scan(query, k, filter, null), true);
        if (rule == Rule.ACORN) {
            float share = sampledShare(filter);
            if (share >= ACORN_BELOW) return shipped(query, k, filter);
            multiplier = (int) Math.round(Math.min(1 / share, m / 2.0));
            minToFind = (int) Math.round(Math.min(Math.max(0, 1.0 / share - 2.0 * m), m));
            int capacity = d * Math.max(1, multiplier);
            if (toScore.length < capacity) {
                toScore = new int[capacity];
                toExplore = new int[capacity];
            }
        }
        allow = filter;
        q = query.clone();
        if (g.metric == Metric.COSINE) normalize(q);
        if (++gen == 0) {
            Arrays.fill(seen, 0);
            Arrays.fill(askedAt, 0);
            Arrays.fill(distAt, 0);
            gen = 1;
        }
        walk();
        if (res.n < ef) return new Result(g.scan(query, k, filter, null), true);
        return new Result(collect(k), false);
    }

    /** the shipped walk of the library, and whether it scanned: only a scan asks the predicate about every row */
    private Result shipped(float[] query, int k, LongPredicate filter) {
        long[] calls = new long[1];
        Hit[] hits = g.search(query, k, id -> {
            calls[0]++;
            return filter.test(id);
        }, ef, 0.0);
        return new Result(hits, calls[0] >= g.size());
    }

    /** the share of passing rows in the sample that {@link FilterPlan} takes */
    private float sampledShare(LongPredicate filter) {
        int sample = Math.min(FilterPlan.SAMPLE, g.n);
        int live = 0, pass = 0;
        for (int j = 0; j < sample; j++) {
            int o = sample == g.n ? j : Math.min(g.n - 1, (int) (((j + 0.5) * PHI % 1.0) * g.n));
            if (g.isDead(o)) continue;
            live++;
            if (filter.test(g.ids[o])) pass++;
        }
        return live == 0 ? 0f : pass / (float) live;
    }

    private void walk() {
        int cur = g.entryRow;
        float curD = distance(cur);
        for (int l = g.maxLevel; l >= 1; l--) {
            for (boolean moved = true; moved;) {
                moved = false;
                int b = g.upIdx[cur];
                if (b < 0) break;
                for (int i = g.upOff[b + l - 1]; i < g.upOff[b + l]; i++) {
                    int x = g.upEdges[i];
                    float dx = distance(x);
                    if (dx < curD) {
                        curD = dx;
                        cur = x;
                        moved = true;
                    }
                }
            }
        }
        cand.clear();
        res.clear();
        computed = 0;
        int budget = Math.max(50 * ef, g.n / 20);
        seen[cur] = gen;
        distAt[cur] = gen;
        distOf[cur] = curD;
        cand.push(curD, cur);
        if (passes(cur)) res.push(-curD, cur);
        while (cand.n > 0) {
            float nearest = cand.topKey();
            int c = cand.topVal();
            cand.pop();
            if (res.n >= ef && nearest > -res.topKey()) break;
            if (computed > budget) break;
            if (rule == Rule.ACORN) expandAcorn(c);
            else expandNavix(c);
        }
    }

    private void expandAcorn(int c) {
        int from = g.off0[c], to = g.off0[c + 1], degree = to - from;
        if (degree == 0) return;
        int capacity = d * Math.max(1, multiplier);
        int found = 0, head = 0, tail = 0;
        for (int i = from; i < to && found < capacity; i++) {
            int x = g.edges0[i];
            if (seen[x] == gen) continue;
            seen[x] = gen;
            if (passes(x)) toScore[found++] = x;
            else toExplore[tail++] = x;
        }
        float failing = tail / (float) degree;
        int target = (int) (degree * Math.min(multiplier, 1f / (1f - failing)));
        int mostLooked = capacity - 1;
        int looked = found + tail;
        if (found < target && failing > ACORN_EXPAND_ABOVE) {
            while (head < tail && looked < mostLooked && found < target) {
                int f = toExplore[head++];
                for (int j = g.off0[f], end = g.off0[f + 1]; j < end && found < target; j++) {
                    int y = g.edges0[j];
                    if (seen[y] == gen) continue;
                    seen[y] = gen;
                    looked++;
                    if (passes(y)) toScore[found++] = y;
                    else if (looked < mostLooked && found < minToFind) toExplore[tail++] = y;
                }
            }
        }
        for (int i = 0; i < found; i++) offer(toScore[i], cached(toScore[i]));
    }

    private void expandNavix(int c) {
        int from = g.off0[c], to = g.off0[c + 1], degree = to - from;
        if (degree == 0) return;
        int passing = 0;
        for (int i = from; i < to; i++) {
            int x = g.edges0[i];
            if (passes(x)) selected[passing++] = x;
        }
        float share = passing / (float) degree;
        if (share >= NAVIX_ONE_HOP) {
            for (int i = 0; i < passing; i++) score(selected[i]);
            return;
        }
        boolean blind = share * (d + 1) * d < NAVIX_LENIENCY * d;
        for (int i = 0; i < degree; i++) order[i] = g.edges0[from + i];
        if (!blind) { // directed: the distance of every neighbor, and the neighbors in the order of it
            for (int i = 0; i < degree; i++) {
                int x = order[i];
                float dx = cached(x);
                int j = i;
                while (j > 0 && orderD[j - 1] > dx) {
                    orderD[j] = orderD[j - 1];
                    order[j] = order[j - 1];
                    j--;
                }
                orderD[j] = dx;
                order[j] = x;
            }
        }
        int scored = 0;
        for (int i = 0; i < passing; i++) if (score(selected[i])) scored++;
        for (int i = 0; i < degree && scored < d; i++) {
            int f = order[i];
            for (int j = g.off0[f], end = g.off0[f + 1]; j < end && scored < d; j++) {
                int y = g.edges0[j];
                if (seen[y] != gen && passes(y) && score(y)) scored++;
            }
        }
    }

    /** scores a passing row and offers it, unless the walk did that before */
    private boolean score(int x) {
        if (seen[x] == gen) return false;
        seen[x] = gen;
        offer(x, cached(x));
        return true;
    }

    private void offer(int x, float dx) {
        if (res.n < ef || dx < -res.topKey()) {
            cand.push(dx, x);
            res.push(-dx, x);
            if (res.n > ef) res.pop();
        }
    }

    /** the distance of a row on layer 0, computed once per walk */
    private float cached(int x) {
        if (distAt[x] != gen) {
            distAt[x] = gen;
            distOf[x] = distance(x);
            computed++;
        }
        return distOf[x];
    }

    /** whether a live row passes the filter; the predicate is asked once per row and walk */
    private boolean passes(int x) {
        if (askedAt[x] != gen) {
            askedAt[x] = gen;
            passed[x] = !g.isDead(x) && allow.test(g.ids[x]);
        }
        return passed[x];
    }

    private float distance(int r) {
        if (g.metric == Metric.L2) return g.vectors.l2(q, r, scratch);
        float dot = g.vectors.dot(q, r, scratch);
        return g.metric == Metric.COSINE ? 1f - dot : -dot;
    }

    private Hit[] collect(int k) {
        int n = res.n;
        int[] rows = new int[n];
        float[] dist = new float[n];
        for (int i = n - 1; i >= 0; i--) {
            dist[i] = -res.topKey();
            rows[i] = res.topVal();
            res.pop();
        }
        Hit[] out = new Hit[Math.min(k, n)];
        int count = 0;
        for (int i = 0; i < n && count < k; i++) {
            if (g.isDead(rows[i])) continue;
            out[count++] = new Hit(g.ids[rows[i]], g.metric == Metric.COSINE ? 1f - dist[i] : -dist[i]);
        }
        return count == out.length ? out : Arrays.copyOf(out, count);
    }

    private static void normalize(float[] v) {
        double n = 0;
        for (float x : v) n += (double) x * x;
        if (n > 0) {
            float inv = (float) (1 / Math.sqrt(n));
            for (int i = 0; i < v.length; i++) v[i] *= inv;
        }
    }
}
