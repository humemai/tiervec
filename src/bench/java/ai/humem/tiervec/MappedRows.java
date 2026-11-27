package ai.humem.tiervec;

import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.LongPredicate;
import java.util.stream.IntStream;

/**
 * What a row copy costs when the vectors of a sealed segment stay in a mapped file (issue #27; docs/BENCHMARKS.md, "The cost of a row copy").
 * On JDK 21 without preview features the kernels read a {@code float[]}, so a distance on a mapped row copies the row first.
 * {@code build FILE ROWS DIM} writes one segment file of random vectors. {@code run FILE} is one sample in one JVM and prints one
 * {@code MAPPEDROWS} line per case. {@code summary FILE} reads the lines of many JVMs and prints a Markdown table with the median, smallest,
 * and largest ratio of every case. The mapped vectors here exist for this measurement only: the library has no mapped reader. The exit code is
 * 1 when the file and the heap array differ, or when the heap graph and the mapped graph answer differently.
 */
public final class MappedRows {
    private static final int DISTANCES = 200_000, QUERIES = 100, SCANS = 10, ROUNDS = 15, WARMUP = 5, K = 10;
    private static double sink;

    private MappedRows() {}

    public static void main(String[] args) throws IOException {
        if (args.length == 4 && args[0].equals("build")) build(Path.of(args[1]), Integer.parseInt(args[2]), Integer.parseInt(args[3]));
        else if (args.length == 2 && args[0].equals("run")) run(Path.of(args[1]));
        else if (args.length == 2 && args[0].equals("summary")) summary(Path.of(args[1]));
        else throw new IllegalArgumentException("use: build FILE ROWS DIM, run FILE, or summary FILE");
        if (sink == 42) System.out.println();
    }

    /** the VEC section left in the file; every distance copies its row into one scratch array, so this serves one thread only */
    private static final class Mapped implements SealedVectors {
        private final FloatBuffer b;
        private final int dim;
        private final float[] scratch;

        Mapped(FloatBuffer b, int dim) {
            this.b = b;
            this.dim = dim;
            this.scratch = new float[dim];
        }

        private float[] load(int row) {
            b.get(row * dim, scratch);
            return scratch;
        }

        @Override
        public float l2(float[] q, int row) {
            return Kern.l2(q, 0, load(row), 0, dim);
        }

        @Override
        public float dot(float[] q, int row) {
            return Kern.dot(q, 0, load(row), 0, dim);
        }

        @Override
        public void copy(int row, float[] dst, int dstOff) {
            b.get(row * dim, dst, dstOff, dim);
        }

        @Override
        public float[] row(int row) {
            float[] r = new float[dim];
            copy(row, r, 0);
            return r;
        }

        @Override
        public int length() {
            return b.capacity();
        }

        @Override
        public void put(FloatBuffer dst, int from, int count) {
            dst.put(b.slice(from, count));
        }

        @Override
        public long heapBytes() {
            return 0;
        }
    }

    private static float[] vector(SplittableRandom r, int dim) {
        float[] v = new float[dim];
        for (int i = 0; i < dim; i++) v[i] = (float) r.nextGaussian();
        return v;
    }

    private static void build(Path file, int rows, int dim) throws IOException {
        SplittableRandom r = new SplittableRandom(7);
        float[][] v = new float[rows][];
        for (int i = 0; i < rows; i++) v[i] = vector(r, dim);
        long t0 = System.nanoTime();
        try (GraphIndex g = new GraphIndex(dim, Metric.COSINE, GraphIndex.Params.defaults().withEfConstruction(64).withSq8(true))) {
            IntStream.range(0, rows).parallel().forEach(i -> g.add(i, v[i]));
            g.seal().serialize(file);
        }
        System.out.println("MAPPEDROWS-BUILD rows=" + rows + " dim=" + dim + " seconds=" + (System.nanoTime() - t0) / 1_000_000_000L + " bytes=" + Files.size(file));
    }

    /** the graph of g over other vectors, walking with its SQ8 codes or with the fp32 vectors */
    private static SealedGraph view(SealedGraph g, SealedVectors v, boolean sq8) {
        return new SealedGraph(g.dim, g.metric, g.params, g.n, v, sq8 ? g.codes : null, sq8 ? g.qbase : null, sq8 ? g.qdelta : null,
                sq8 ? g.qnorm : null, g.ids, g.level, g.off0, g.edges0, g.upIdx, g.upOff, g.upEdges, g.entryRow, g.maxLevel, g.sortedRows);
    }

    private static void fail(String why) {
        System.out.println("MAPPEDROWS-FAILURE " + why);
        System.exit(1);
    }

    private static void run(Path file) throws IOException {
        SealedGraph loaded = SealedGraph.load(file);
        final int n = loaded.n, dim = loaded.dim;
        if (loaded.codes == null) fail("the segment has no SQ8 codes");
        float[] heap = new float[n * dim];
        for (int row = 0; row < n; row++) loaded.copyVector(row, heap, row * dim);
        SegmentInfo.Section vec = SegmentReader.inspect(file).section(SegmentFormat.S_VEC);
        FloatBuffer mapped;
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
            mapped = ch.map(FileChannel.MapMode.READ_ONLY, vec.offset(), vec.length()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        }
        if (mapped.capacity() != heap.length) fail("the mapped section has " + mapped.capacity() + " floats, expected " + heap.length);
        for (int i = 0; i < heap.length; i++) if (Float.floatToRawIntBits(mapped.get(i)) != Float.floatToRawIntBits(heap[i])) fail("float " + i + " differs");
        System.out.println("MAPPEDROWS-ENV jdk=" + Runtime.version().feature() + " java=" + Runtime.version() + " processors="
                + Runtime.getRuntime().availableProcessors() + " vector=" + Kernels.VECTOR + " lanes=" + Kern.lanes());

        SplittableRandom r = new SplittableRandom(42);
        float[][] queries = new float[QUERIES][];
        for (int i = 0; i < QUERIES; i++) queries[i] = vector(r, dim);
        int[] random = new int[DISTANCES], inOrder = new int[DISTANCES];
        for (int i = 0; i < DISTANCES; i++) {
            random[i] = r.nextInt(n);
            inOrder[i] = i % n;
        }
        distanceCase("distance-random", heap, mapped, queries[0], random, n);
        distanceCase("distance-in-order", heap, mapped, queries[0], inOrder, n);

        SealedVectors onHeap = new SealedVectors.OnHeap(heap, dim);
        SealedVectors inFile = new Mapped(mapped, dim);
        searchCase("walk-fp32", view(loaded, onHeap, false), view(loaded, inFile, false), queries, null, QUERIES);
        searchCase("walk-sq8", view(loaded, onHeap, true), view(loaded, inFile, true), queries, null, QUERIES);
        searchCase("exact-scan", view(loaded, onHeap, false), view(loaded, inFile, false), queries, id -> true, SCANS);
    }

    private static void distanceCase(String name, float[] heap, FloatBuffer mapped, float[] q, int[] rows, int n) {
        float[] scratch = new float[q.length];
        for (int row : Arrays.copyOf(rows, 1000)) {
            float want = onHeap(q, heap, new int[] {row});
            if (want != heapCopy(q, heap, new int[] {row}, scratch) || want != mappedCopy(q, mapped, new int[] {row}, scratch)) fail(name + " the arms differ on row " + row);
        }
        double[][] ns = new double[3][ROUNDS];
        for (int round = -WARMUP; round < ROUNDS; round++) {
            for (int j = 0; j < 3; j++) {
                int arm = Math.floorMod(round + j, 3); // the arm that goes first rotates
                long t0 = System.nanoTime();
                sink += arm == 0 ? onHeap(q, heap, rows) : arm == 1 ? heapCopy(q, heap, rows, scratch) : mappedCopy(q, mapped, rows, scratch);
                if (round >= 0) ns[arm][round] = (double) (System.nanoTime() - t0) / rows.length;
            }
        }
        System.out.println(String.format(Locale.ROOT, "MAPPEDROWS case=%s rows=%d dim=%d unit=ns heap=%.2f heapCopyRatio=%.4f mappedRatio=%.4f",
                name, n, q.length, median(ns[0]), median(ratios(ns[1], ns[0])), median(ratios(ns[2], ns[0]))));
    }

    private static float onHeap(float[] q, float[] a, int[] rows) {
        final int dim = q.length;
        float s = 0;
        for (int row : rows) s += Kern.dot(q, 0, a, row * dim, dim);
        return s;
    }

    private static float heapCopy(float[] q, float[] a, int[] rows, float[] scratch) {
        final int dim = q.length;
        float s = 0;
        for (int row : rows) {
            System.arraycopy(a, row * dim, scratch, 0, dim);
            s += Kern.dot(q, 0, scratch, 0, dim);
        }
        return s;
    }

    private static float mappedCopy(float[] q, FloatBuffer b, int[] rows, float[] scratch) {
        final int dim = q.length;
        float s = 0;
        for (int row : rows) {
            b.get(row * dim, scratch);
            s += Kern.dot(q, 0, scratch, 0, dim);
        }
        return s;
    }

    private static void searchCase(String name, SealedGraph heap, SealedGraph mapped, float[][] queries, LongPredicate scanned, int count) {
        for (int i = 0; i < count; i++) {
            Hit[] want = answer(heap, queries[i], scanned);
            if (want.length != K || !Arrays.equals(want, answer(mapped, queries[i], scanned))) fail(name + " the graphs answer query " + i + " differently");
        }
        double[][] ms = new double[2][ROUNDS];
        for (int round = -WARMUP; round < ROUNDS; round++) {
            for (int j = 0; j < 2; j++) {
                int arm = Math.floorMod(round + j, 2);
                SealedGraph g = arm == 0 ? heap : mapped;
                long t0 = System.nanoTime();
                for (int i = 0; i < count; i++) sink += answer(g, queries[i], scanned)[0].id();
                if (round >= 0) ms[arm][round] = (System.nanoTime() - t0) / 1e6 / count;
            }
        }
        System.out.println(String.format(Locale.ROOT, "MAPPEDROWS case=%s rows=%d dim=%d unit=ms heap=%.4f heapCopyRatio=- mappedRatio=%.4f",
                name, heap.n, heap.dim, median(ms[0]), median(ratios(ms[1], ms[0]))));
    }

    /** the walk of the graph, or with a filter the exact scan of the rows that it passes */
    private static Hit[] answer(SealedGraph g, float[] q, LongPredicate scanned) {
        return scanned == null ? g.search(q, K, null) : g.scan(q, K, scanned, null);
    }

    private static double[] ratios(double[] x, double[] base) {
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) out[i] = x[i] / base[i];
        return out;
    }

    private static double median(double[] x) {
        double[] s = x.clone();
        Arrays.sort(s);
        return (s[(s.length - 1) / 2] + s[s.length / 2]) / 2;
    }

    private static void summary(Path file) throws IOException {
        Map<String, List<Map<String, String>>> cases = new LinkedHashMap<>();
        String jdk = "?";
        for (String line : Files.readAllLines(file)) {
            boolean env = line.startsWith("MAPPEDROWS-ENV ");
            if (!env && !line.startsWith("MAPPEDROWS case=")) continue;
            Map<String, String> m = new LinkedHashMap<>();
            for (String part : line.substring(line.indexOf(' ') + 1).split(" ")) {
                int eq = part.indexOf('=');
                m.put(part.substring(0, eq), part.substring(eq + 1));
            }
            if (env) jdk = m.get("java") + ", " + m.get("lanes") + " float lanes";
            else cases.computeIfAbsent(m.get("rows") + " x " + m.get("dim") + " " + m.get("case"), c -> new ArrayList<>()).add(m);
        }
        if (cases.isEmpty()) throw new IllegalStateException("no MAPPEDROWS lines in " + file);
        System.out.println("Java " + jdk + ". Ratios are the time divided by the time on the heap: median of the JVMs (smallest to largest).");
        System.out.println();
        System.out.println("| Rows x dim | Case | JVMs | On the heap | Heap, row copied first | Mapped file, row copied first |");
        System.out.println("| --- | --- | --- | --- | --- | --- |");
        for (List<Map<String, String>> rows : cases.values()) {
            Map<String, String> first = rows.get(0);
            boolean distance = first.get("unit").equals("ns");
            System.out.println(String.format(Locale.ROOT, "| %s x %s | `%s` | %d | %.3f %s | %s | %s |", first.get("rows"), first.get("dim"), first.get("case"),
                    rows.size(), median(column(rows, "heap")), distance ? "ns per distance" : "ms per query",
                    distance ? spread(column(rows, "heapCopyRatio")) : "not measured", spread(column(rows, "mappedRatio"))));
        }
    }

    private static String spread(double[] x) {
        double[] s = x.clone();
        Arrays.sort(s);
        return String.format(Locale.ROOT, "%.2f (%.2f to %.2f)", median(s), s[0], s[s.length - 1]);
    }

    private static double[] column(List<Map<String, String>> rows, String key) {
        return rows.stream().mapToDouble(m -> Double.parseDouble(m.get(key))).toArray();
    }
}
