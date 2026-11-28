package ai.humem.tiervec;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.function.LongPredicate;
import java.util.stream.IntStream;

/**
 * The filter screen (docs/SCREEN-IDEAS.md, "The filter screen", ideas S7 and S8, issue #162): one sealed graph, searched with filters of
 * several allowed shares that are random, or correlated positively or negatively with the query, by the shipped walk, the shipped scan-first
 * plan and the two filter-aware walks of {@link FilterWalk}. It reports recall against a brute force scan of the allowed rows, distance
 * computations, filter calls and how often a scan answered. Nothing is timed. It needs the copy of the library that
 * scripts/instrument_distances.py makes, because the library itself counts no distances.
 *
 * <p>Options, each with a value: {@code --seed}, {@code --rows}, {@code --dim}, {@code --clusters}, {@code --queries}, {@code --out}. The
 * registered run uses the defaults, with the seeds 1, 2 and 3.
 */
public final class FilterScreen {
    static final double[] SHARES = {0.5, 0.1, 0.01, 0.001};
    static final String[] KINDS = {"random", "positive", "negative"};
    static final String[] ARMS = {"walk", "scanfirst", "acorn", "navix"};
    private static final int K = 10, EF = 100, NEIGHBOURS = 100;
    /** as the screen draws its rows (Runbook): the length of the noise around the centre of a cluster */
    private static final double SPREAD = 0.8;

    private final List<String> lines = new ArrayList<>();
    private final List<String> table = new ArrayList<>();
    private final List<String> summary = new ArrayList<>();
    private int failures;

    private void line(String s) {
        System.out.println(s);
        lines.add(s);
    }

    public static void main(String[] args) throws IOException {
        Map<String, String> o = new LinkedHashMap<>();
        o.put("seed", "1");
        o.put("rows", "100000");
        o.put("dim", "768");
        o.put("clusters", "8");
        o.put("queries", "500");
        o.put("out", "target/filtered-walk/out");
        for (int i = 0; i < args.length; i += 2) {
            String name = args[i].startsWith("--") ? args[i].substring(2) : "";
            if (!o.containsKey(name) || i + 1 >= args.length) {
                System.err.println("unknown option or missing value: " + args[i]);
                System.exit(2);
            }
            o.put(name, args[i + 1]);
        }
        long seed = Long.parseLong(o.get("seed"));
        int rows = Integer.parseInt(o.get("rows")), dim = Integer.parseInt(o.get("dim"));
        int clusters = Integer.parseInt(o.get("clusters")), queries = Integer.parseInt(o.get("queries"));
        // the shipped walk gives no flag when it scans; its calls tell, as long as a walk cannot ask about every row
        if (rows <= 2 * Math.max(50 * EF, rows / 20) || dim < 2 || clusters < 2 || queries < 1) {
            System.err.println("needs more than " + 100 * EF + " rows, a dimension and clusters of at least 2, and a query");
            System.exit(2);
        }
        if (!DistanceCount.active()) {
            System.err.println("the filter screen reports distance computations: compile the copy of the library that scripts/instrument_distances.py makes");
            System.exit(2);
        }
        FilterScreen run = new FilterScreen();
        run.run(o, seed, rows, dim, clusters, queries);
        Path out = Path.of(o.get("out"));
        Files.createDirectories(out);
        Files.write(out.resolve("result.txt"), run.lines);
        Files.write(out.resolve("filter.tsv"), run.table);
        Files.write(out.resolve("summary.md"), run.summary);
        System.exit(run.failures == 0 ? 0 : 1);
    }

    private void run(Map<String, String> o, long seed, int rows, int dim, int clusters, int queries) {
        line("FILTER-SCREEN format=1");
        String commit = Bench.sh("git", "rev-parse", "HEAD");
        String machine = "commit=" + (commit == null ? "unknown" : commit) + " jdk=" + System.getProperty("java.runtime.version")
                + " arch=" + System.getProperty("os.arch") + " kernels=" + (Kernels.VECTOR ? "vector" : "scalar") + " lanes=" + Kern.lanes();
        line("RUN started=" + Instant.now() + " " + machine);
        line("OPTIONS " + o + " k=" + K + " ef=" + EF);

        SplittableRandom r = new SplittableRandom(Bench.mix(seed));
        float[][] centres = new float[clusters][dim];
        for (float[] c : centres) {
            for (int i = 0; i < dim; i++) c[i] = (float) Bench.Gen.gauss(r);
            Bench.Gen.norm(c);
        }
        float[][] data = new float[rows][];
        int[] clusterOf = new int[rows];
        for (int i = 0; i < rows; i++) {
            clusterOf[i] = r.nextInt(clusters);
            data[i] = draw(r, centres[clusterOf[i]]);
        }
        float[][] asked = new float[queries][];
        for (int i = 0; i < queries; i++) asked[i] = draw(r, centres[r.nextInt(clusters)]);
        int[][] members = members(clusterOf, clusters);

        GraphIndex building = new GraphIndex(dim, Metric.COSINE);
        for (int i = 0; i < rows; i++) building.add(i, data[i]);
        SealedGraph graph = building.seal();
        if (graph.size() != rows) throw new IllegalStateException("the sealed graph holds " + graph.size() + " of " + rows + " rows");
        line("GRAPH rows=" + graph.size() + " M=" + graph.params.M() + " efConstruction=" + graph.params.efConstruction() + " alpha=" + graph.params.alpha()
                + " sq8=" + graph.params.sq8() + " maxLevel=" + graph.maxLevel + " layer0Edges=" + graph.edges0.length);

        // the similarity of every row to every query, by a loop of the benchmark: no counted kernel
        float[][] sims = new float[queries][rows];
        IntStream.range(0, queries).parallel().forEach(i -> {
            for (int row = 0; row < rows; row++) sims[i][row] = dot(asked[i], data[row]);
        });
        int[][] nearest = new int[queries][];
        IntStream.range(0, queries).parallel().forEach(i -> nearest[i] = top(sims[i], null, NEIGHBOURS));

        long distances = 0;
        double recall = 0;
        for (int i = 0; i < queries; i++) {
            long before = DistanceCount.soFar();
            Hit[] hits = graph.search(asked[i], K, null, EF);
            distances += DistanceCount.soFar() - before;
            recall += recall(hits, top(sims[i], null, K));
        }
        line(Bench.f("UNFILTERED recall=%.4f distancesPerQuery=%.1f", recall / queries, (double) distances / queries));

        table.add(String.join("\t", "seed", "filter", "share", "correlation", "arm", "recall", "distancesPerQuery", "filterCallsPerQuery", "scanRate",
                "shortAnswers"));
        summary.add("## Filter screen, seed " + seed);
        summary.add("");
        summary.add(rows + " rows, " + dim + " dimensions, " + clusters + " clusters, " + queries + " queries per cell, k " + K + ", ef " + EF + ". "
                + machine.replace(" ", ", ") + ". No timing.");
        summary.add("");
        summary.add("| Filter | Share | Correlation | Arm | Recall@10 | Distances per query | Filter calls per query | Scan rate | Short answers |");
        summary.add("|---|---|---|---|---|---|---|---|---|");

        FilterWalk acorn = new FilterWalk(graph, FilterWalk.Rule.ACORN);
        FilterWalk navix = new FilterWalk(graph, FilterWalk.Rule.NAVIX);
        int[] all = IntStream.range(0, rows).toArray();
        for (int kind = 0; kind < KINDS.length; kind++) {
            for (int s = 0; s < SHARES.length; s++) {
                int m = (int) Math.round(SHARES[s] * rows);
                double[] recalls = new double[ARMS.length];
                long[] dist = new long[ARMS.length], calls = new long[ARMS.length];
                int[] scans = new int[ARMS.length], shortOnes = new int[ARMS.length];
                double correlation = 0;
                for (int i = 0; i < queries; i++) {
                    SplittableRandom pick = new SplittableRandom(Bench.mix(Bench.mix(seed * 1000 + kind * 10 + s) + i));
                    boolean[] allowed = kind == 0 ? random(pick, all, m)
                            : correlated(pick, asked[i], centres, members, rows, m, kind == 1);
                    int count = 0;
                    for (boolean b : allowed) if (b) count++;
                    if (count != m) throw new IllegalStateException("the filter allows " + count + " rows, not " + m);
                    int near = 0;
                    for (int row : nearest[i]) if (allowed[row]) near++;
                    correlation += near / (double) nearest[i].length / ((double) m / rows);
                    int[] truth = top(sims[i], allowed, K);
                    for (int a = 0; a < ARMS.length; a++) {
                        long[] asks = new long[1];
                        LongPredicate filter = id -> {
                            asks[0]++;
                            return allowed[(int) id];
                        };
                        long before = DistanceCount.soFar();
                        Hit[] hits;
                        boolean scanned;
                        if (a == 0) {
                            hits = graph.search(asked[i], K, filter, EF, 0.0);
                            scanned = asks[0] >= rows;
                        } else if (a == 1) {
                            hits = graph.search(asked[i], K, filter, EF, 1.0);
                            scanned = asks[0] >= rows;
                            if (!scanned) fail("scanfirst did not scan");
                        } else {
                            FilterWalk.Result result = (a == 2 ? acorn : navix).search(asked[i], K, filter, EF);
                            hits = result.hits();
                            scanned = result.scanned();
                        }
                        dist[a] += DistanceCount.soFar() - before;
                        calls[a] += asks[0];
                        if (scanned) scans[a]++;
                        if (hits.length < Math.min(K, m)) shortOnes[a]++;
                        Set<Long> ids = new HashSet<>();
                        for (Hit h : hits) {
                            if (!allowed[(int) h.id()]) fail(ARMS[a] + " returned a row that the filter does not allow");
                            if (!ids.add(h.id())) fail(ARMS[a] + " returned an id twice");
                        }
                        recalls[a] += recall(hits, truth);
                    }
                }
                for (int a = 0; a < ARMS.length; a++) {
                    String[] cells = {Long.toString(seed), KINDS[kind], Double.toString(SHARES[s]), Bench.f("%.3f", correlation / queries), ARMS[a],
                            Bench.f("%.4f", recalls[a] / queries), Bench.f("%.1f", (double) dist[a] / queries), Bench.f("%.1f", (double) calls[a] / queries),
                            Bench.f("%.4f", (double) scans[a] / queries), Integer.toString(shortOnes[a])};
                    table.add(String.join("\t", cells));
                    line("RESULT seed=" + cells[0] + " filter=" + cells[1] + " share=" + cells[2] + " correlation=" + cells[3] + " arm=" + cells[4]
                            + " recall=" + cells[5] + " distancesPerQuery=" + cells[6] + " filterCallsPerQuery=" + cells[7] + " scanRate=" + cells[8]
                            + " shortAnswers=" + cells[9]);
                    summary.add("| " + String.join(" | ", Arrays.copyOfRange(cells, 1, cells.length)) + " |");
                }
            }
        }
        line("SUMMARY status=" + (failures == 0 ? "OK" : "FAILED") + " failures=" + failures);
        summary.add("");
        summary.add(failures == 0 ? "No failure." : "**" + failures + " failures**, see `result.txt`.");
    }

    private void fail(String what) {
        if (failures++ < 20) line("FAILURE " + what);
    }

    private static float[] draw(SplittableRandom r, float[] centre) {
        float scale = (float) (SPREAD / StrictMath.sqrt(centre.length));
        float[] v = new float[centre.length];
        for (int i = 0; i < v.length; i++) v[i] = centre[i] + scale * (float) Bench.Gen.gauss(r);
        Bench.Gen.norm(v);
        return v;
    }

    static int[][] members(int[] clusterOf, int clusters) {
        int[] size = new int[clusters];
        for (int c : clusterOf) size[c]++;
        int[][] members = new int[clusters][];
        for (int c = 0; c < clusters; c++) members[c] = new int[size[c]];
        int[] at = new int[clusters];
        for (int row = 0; row < clusterOf.length; row++) members[clusterOf[row]][at[clusterOf[row]]++] = row;
        return members;
    }

    /** a uniform choice of {@code m} of the rows in {@code pool}: the first steps of a shuffle of a copy */
    private static void choose(SplittableRandom r, int[] pool, int m, boolean[] allowed) {
        int[] p = pool.clone();
        for (int i = 0; i < m; i++) {
            int j = i + r.nextInt(p.length - i);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
            allowed[p[i]] = true;
        }
    }

    private static boolean[] random(SplittableRandom r, int[] all, int m) {
        boolean[] allowed = new boolean[all.length];
        choose(r, all, m, allowed);
        return allowed;
    }

    /** whole clusters in the order of the similarity of their centre to the query, and a uniform choice from the one that does not fit */
    static boolean[] correlated(SplittableRandom r, float[] query, float[][] centres, int[][] members, int rows, int m, boolean positive) {
        Integer[] byNearness = new Integer[centres.length];
        float[] sim = new float[centres.length];
        for (int c = 0; c < centres.length; c++) {
            byNearness[c] = c;
            sim[c] = dot(query, centres[c]);
        }
        Arrays.sort(byNearness, (a, b) -> positive ? Float.compare(sim[b], sim[a]) : Float.compare(sim[a], sim[b]));
        boolean[] allowed = new boolean[rows];
        int left = m;
        for (int c : byNearness) {
            if (left == 0) break;
            if (members[c].length <= left) {
                for (int row : members[c]) allowed[row] = true;
                left -= members[c].length;
            } else {
                choose(r, members[c], left, allowed);
                left = 0;
            }
        }
        return allowed;
    }

    static float dot(float[] a, float[] b) {
        float s = 0;
        for (int i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }

    /** the rows with the {@code k} highest similarities among the allowed ones (all when {@code allowed} is null) */
    static int[] top(float[] sim, boolean[] allowed, int k) {
        Heap h = new Heap();
        for (int row = 0; row < sim.length; row++) {
            if (allowed != null && !allowed[row]) continue;
            if (h.n < k) h.push(sim[row], row);
            else if (sim[row] > h.topKey()) {
                h.pop();
                h.push(sim[row], row);
            }
        }
        int[] out = new int[h.n];
        for (int i = out.length - 1; i >= 0; i--) {
            out[i] = h.topVal();
            h.pop();
        }
        return out;
    }

    static double recall(Hit[] hits, int[] truth) {
        Set<Long> want = new HashSet<>();
        for (int row : truth) want.add((long) row);
        int found = 0;
        for (Hit h : hits) if (want.remove(h.id())) found++;
        return truth.length == 0 ? 1.0 : found / (double) truth.length;
    }
}
