package ai.humem.tiervec;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.function.LongPredicate;

/**
 * The tests of the filter-aware walks of the filter screen (issue #162). Maven does not compile src/bench, so the CI job "bench smoke"
 * compiles and runs this program. It prints each check that fails and exits with code 1 when one did.
 */
public final class FilterWalkCheck {
    private static final int ROWS = 12_000, DIM = 32, CLUSTERS = 4, K = 10, EF = 100;
    private static int checks, failures;

    private static void check(boolean ok, String what) {
        checks++;
        if (!ok) {
            failures++;
            System.out.println("FAILED " + what);
        }
    }

    public static void main(String[] args) {
        SplittableRandom r = new SplittableRandom(162);
        float[][] centres = new float[CLUSTERS][DIM];
        for (float[] c : centres) {
            for (int i = 0; i < DIM; i++) c[i] = (float) Bench.Gen.gauss(r);
            Bench.Gen.norm(c);
        }
        float[][] data = new float[ROWS][];
        int[] clusterOf = new int[ROWS];
        GraphIndex building = new GraphIndex(DIM, Metric.COSINE);
        for (int i = 0; i < ROWS; i++) {
            clusterOf[i] = r.nextInt(CLUSTERS);
            data[i] = row(r, centres[clusterOf[i]]);
            building.add(i, data[i]);
        }
        SealedGraph graph = building.seal();
        float[][] queries = new float[40][];
        for (int i = 0; i < queries.length; i++) queries[i] = row(r, centres[r.nextInt(CLUSTERS)]);
        int[][] members = FilterScreen.members(clusterOf, CLUSTERS);

        for (FilterWalk.Rule rule : FilterWalk.Rule.values()) {
            FilterWalk walk = new FilterWalk(graph, rule);
            nothing(graph, walk, queries[0], rule);
            // random filters: every share of the screen, with a share of 0.001 here being 12 rows, fewer than ef
            for (double share : FilterScreen.SHARES) {
                boolean[] allowed = new boolean[ROWS];
                int want = (int) Math.round(share * ROWS);
                for (int left = want; left > 0;) {
                    int x = r.nextInt(ROWS);
                    if (!allowed[x]) {
                        allowed[x] = true;
                        left--;
                    }
                }
                cell(graph, walk, data, queries, q -> allowed, rule + " random " + share, want <= EF, share >= 0.1);
            }
            // filters that follow the query, to one side and to the other
            for (boolean positive : new boolean[] {true, false}) {
                for (double share : new double[] {0.5, 0.1, 0.01}) {
                    int want = (int) Math.round(share * ROWS);
                    cell(graph, walk, data, queries,
                            q -> FilterScreen.correlated(new SplittableRandom(q), queries[q], centres, members, ROWS, want, positive),
                            rule + (positive ? " positive " : " negative ") + share, false, false);
                }
            }
            // a hint of at most ef rows is a scan
            SizedIdFilter hinted = new SizedIdFilter() {
                @Override public boolean test(long id) { return id < 50; }
                @Override public long allowedCount() { return 50; }
            };
            FilterWalk.Result small = walk.search(queries[0], K, hinted, EF);
            check(small.scanned() && Arrays.equals(small.hits(), graph.scan(queries[0], K, hinted, null)), rule + ": a small hint is a scan");
            check(walk.search(queries[0], 0, id -> true, EF).hits().length == 0, rule + ": k 0 returns nothing");
        }

        // at a share of 0.6 or more the ACORN rule leaves the search to the shipped walk
        LongPredicate most = id -> id % 5 != 0;
        FilterWalk acorn = new FilterWalk(graph, FilterWalk.Rule.ACORN);
        for (float[] q : queries) {
            check(Arrays.equals(acorn.search(q, K, most, EF).hits(), graph.search(q, K, most, EF, 0.0)), "ACORN at a share of 0.8 is the shipped walk");
        }

        // deleted rows are never returned
        for (int id = 0; id < ROWS; id += 3) graph.delete(id);
        for (FilterWalk.Rule rule : FilterWalk.Rule.values()) {
            FilterWalk walk = new FilterWalk(graph, rule);
            for (float[] q : queries) {
                for (Hit h : walk.search(q, K, id -> id % 2 == 0, EF).hits()) {
                    check(h.id() % 3 != 0 && h.id() % 2 == 0, rule + ": returned the deleted or filtered id " + h.id());
                }
            }
            check(walk.search(queries[0], K, id -> id % 3 == 0, EF).hits().length == 0, rule + ": only deleted rows allowed returns nothing");
        }

        System.out.println("filter walk checks: " + checks + " run, " + failures + " failed");
        System.exit(failures == 0 && checks > 0 ? 0 : 1);
    }

    /** a filter that allows nothing returns nothing, with and without a hint of the count, and with a wrong hint */
    private static void nothing(SealedGraph graph, FilterWalk walk, float[] q, FilterWalk.Rule rule) {
        check(walk.search(q, K, id -> false, EF).hits().length == 0, rule + ": a filter that allows nothing returns nothing");
        for (long hint : new long[] {0, ROWS, SizedIdFilter.UNKNOWN}) {
            SizedIdFilter none = new SizedIdFilter() {
                @Override public boolean test(long id) { return false; }
                @Override public long allowedCount() { return hint; }
            };
            check(walk.search(q, K, none, EF).hits().length == 0, rule + ": a filter that allows nothing returns nothing, hint " + hint);
        }
        check(graph.search(q, K, id -> false, EF, 0.0).length == 0, "the shipped walk returns nothing for a filter that allows nothing");
    }

    private interface Allowed {
        boolean[] of(int query);
    }

    /**
     * One filter on every query: only allowed ids, no id twice, hits in the order of their score, as many hits as there can be, the exact
     * answer when a scan ran, one predicate call per row in a walk, and a loose bound on recall.
     */
    private static void cell(SealedGraph graph, FilterWalk walk, float[][] data, float[][] queries, Allowed filters, String what, boolean mustScan,
                             boolean mustWalk) {
        double recall = 0;
        int scans = 0;
        for (int i = 0; i < queries.length; i++) {
            boolean[] allowed = filters.of(i);
            float[] sim = new float[ROWS];
            for (int x = 0; x < ROWS; x++) sim[x] = FilterScreen.dot(queries[i], data[x]);
            int[] truth = FilterScreen.top(sim, allowed, K);
            int[] calls = new int[ROWS];
            FilterWalk.Result result = walk.search(queries[i], K, id -> {
                calls[(int) id]++;
                return allowed[(int) id];
            }, EF);
            Set<Long> ids = new HashSet<>();
            float last = Float.POSITIVE_INFINITY;
            for (Hit h : result.hits()) {
                check(allowed[(int) h.id()], what + ": a hit that the filter does not allow");
                check(ids.add(h.id()), what + ": an id twice");
                check(h.score() <= last, what + ": hits out of order");
                last = h.score();
            }
            check(result.hits().length == truth.length, what + ": " + result.hits().length + " hits, not " + truth.length);
            double one = FilterScreen.recall(result.hits(), truth);
            if (result.scanned()) {
                scans++;
                check(one == 1.0, what + ": a scan that is not exact");
            } else {
                // the sample of the ACORN rule asks some rows once more
                int limit = what.startsWith("ACORN") ? 2 : 1;
                int most = Arrays.stream(calls).max().orElse(0);
                check(most <= limit, what + ": a row asked " + most + " times in a walk");
            }
            recall += one;
        }
        recall /= queries.length;
        System.out.println(Bench.f("%s: recall %.4f, scans %d of %d", what, recall, scans, queries.length));
        check(!mustScan || scans == queries.length, what + ": fewer than ef rows pass, so every query must scan");
        check(!mustWalk || scans < queries.length / 2, what + ": the walk answered " + (queries.length - scans) + " queries, too few to test it");
        // the bound is on random filters at the large shares; a correlated filter may lower the recall of a walk, which the screen measures
        check(!mustWalk && !mustScan || recall >= 0.9, what + ": recall " + recall);
    }

    private static float[] row(SplittableRandom r, float[] centre) {
        float[] v = new float[centre.length];
        for (int i = 0; i < v.length; i++) v[i] = centre[i] + 0.15f * (float) Bench.Gen.gauss(r);
        Bench.Gen.norm(v);
        return v;
    }
}
