package ai.humem.tiervec;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.LongPredicate;

/**
 * The cost of polling a {@link SearchContext} (issue #11; docs/BENCHMARKS.md, "The cost of polling"). A run without arguments is one sample in one JVM: it
 * times the same queries without a context and with one that never expires (a cancel flag and a deadline read with {@code System.nanoTime()}), in rounds that
 * alternate which arm goes first, and prints one {@code POLLCOST} line per cell. The ratio is the time with the context divided by the time without it, and the
 * line has the median over the rounds. {@code summary FILE} reads the lines of many JVMs and prints the median, smallest and largest ratio of every cell.
 * The exit code is 1 when the two arms answer differently.
 */
public final class PollCost {
    private static final int N = 20_000, DIM = 128, QUERIES = 200, ROUNDS = 40, WARMUP = 15, K = 10;
    private static long sink;

    private PollCost() {}

    public static void main(String[] args) throws IOException {
        if (args.length == 2 && args[0].equals("summary")) {
            summary(Path.of(args[1]));
            return;
        }
        if (args.length != 0) throw new IllegalArgumentException("use no argument, or: summary FILE");
        SplittableRandom r = new SplittableRandom(42);
        float[][] queries = new float[QUERIES][];
        for (int i = 0; i < QUERIES; i++) queries[i] = vector(r);
        TieredIndex.Config manual = new TieredIndex.Config(1 << 28, 0L, 1000, 1000, 4, 1.01, 0, 4, Quant.NONE, GraphIndex.Params.defaults(),
                Math.max(1, Runtime.getRuntime().availableProcessors()));
        try (TieredIndex tail = new TieredIndex(DIM, Metric.COSINE, manual); TieredIndex segment = new TieredIndex(DIM, Metric.COSINE, manual)) {
            SplittableRandom data = new SplittableRandom(7);
            for (int i = 0; i < N; i++) {
                float[] v = vector(data);
                tail.add(i, v);
                segment.add(i, v);
            }
            segment.sealNow();
            if (tail.stats().segments() != 0 || segment.stats().segments() != 1 || segment.stats().tailSize() != 0)
                throw new IllegalStateException("the two tiers are not a tail and one segment");
            System.out.println("POLLCOST-ENV java=" + Runtime.version() + " processors=" + Runtime.getRuntime().availableProcessors() + " vector=" + Kernels.VECTOR);
            cell("tail", tail, queries, null);
            cell("walk", segment, queries, null);
            segment.filterScanBelow(1.0);
            cell("scan", segment, queries, id -> id % 10 == 3);
        }
        if (sink == 42) System.out.println();
    }

    private static float[] vector(SplittableRandom r) {
        float[] v = new float[DIM];
        for (int i = 0; i < DIM; i++) v[i] = (float) r.nextGaussian();
        return v;
    }

    private static void cell(String name, TieredIndex ix, float[][] queries, LongPredicate allow) {
        AtomicBoolean cancel = new AtomicBoolean();
        long deadline = System.nanoTime() + 3_600_000_000_000L;
        SearchContext context = () -> cancel.get() || System.nanoTime() - deadline >= 0;
        AtomicLong polls = new AtomicLong();
        SearchContext counting = () -> polls.incrementAndGet() < 0;
        for (float[] q : queries) {
            Hit[] without = ix.search(q, K, allow, (SearchContext) null);
            if (without.length != K || !Arrays.equals(without, ix.search(q, K, allow, context)) || !Arrays.equals(without, ix.search(q, K, allow, counting))) {
                System.out.println("POLLCOST-FAILURE cell=" + name + " the arms answer differently");
                System.exit(1);
            }
        }
        for (int i = 0; i < WARMUP; i++) {
            pass(ix, queries, allow, null);
            pass(ix, queries, allow, context);
        }
        double[] none = new double[ROUNDS], with = new double[ROUNDS], ratio = new double[ROUNDS];
        for (int i = 0; i < ROUNDS; i++) {
            if (i % 2 == 0) {
                none[i] = pass(ix, queries, allow, null);
                with[i] = pass(ix, queries, allow, context);
            } else {
                with[i] = pass(ix, queries, allow, context);
                none[i] = pass(ix, queries, allow, null);
            }
            ratio[i] = with[i] / none[i];
        }
        System.out.println(String.format(Locale.ROOT, "POLLCOST cell=%s n=%d dim=%d queries=%d rounds=%d interval=%d pollsPerQuery=%.1f noneMs=%.3f contextMs=%.3f ratio=%.4f",
                name, N, DIM, QUERIES, ROUNDS, Poll.INTERVAL, (double) polls.get() / QUERIES, median(none), median(with), median(ratio)));
    }

    /** milliseconds of one pass over the queries */
    private static double pass(TieredIndex ix, float[][] queries, LongPredicate allow, SearchContext context) {
        long t0 = System.nanoTime();
        for (float[] q : queries) sink += ix.search(q, K, allow, context)[0].id();
        return (System.nanoTime() - t0) / 1e6;
    }

    private static double median(double[] x) {
        double[] s = x.clone();
        Arrays.sort(s);
        return (s[(s.length - 1) / 2] + s[s.length / 2]) / 2;
    }

    private static void summary(Path file) throws IOException {
        Map<String, List<Map<String, String>>> cells = new LinkedHashMap<>();
        for (String line : Files.readAllLines(file)) {
            if (!line.startsWith("POLLCOST cell=")) continue;
            Map<String, String> m = new LinkedHashMap<>();
            for (String part : line.substring("POLLCOST ".length()).split(" ")) {
                int eq = part.indexOf('=');
                m.put(part.substring(0, eq), part.substring(eq + 1));
            }
            cells.computeIfAbsent(m.get("cell"), c -> new ArrayList<>()).add(m);
        }
        if (cells.size() != 3) throw new IllegalStateException("expected three cells, found " + cells.keySet());
        for (Map.Entry<String, List<Map<String, String>>> e : cells.entrySet()) {
            double[] ratio = column(e.getValue(), "ratio");
            double[] sorted = ratio.clone();
            Arrays.sort(sorted);
            System.out.println(String.format(Locale.ROOT, "POLLCOST-SUMMARY cell=%s jvms=%d pollsPerQuery=%.1f noneMs=%.3f contextMs=%.3f ratioMedian=%.4f ratioMin=%.4f ratioMax=%.4f",
                    e.getKey(), ratio.length, median(column(e.getValue(), "pollsPerQuery")), median(column(e.getValue(), "noneMs")),
                    median(column(e.getValue(), "contextMs")), median(ratio), sorted[0], sorted[sorted.length - 1]));
        }
    }

    private static double[] column(List<Map<String, String>> rows, String key) {
        return rows.stream().mapToDouble(m -> Double.parseDouble(m.get(key))).toArray();
    }
}
