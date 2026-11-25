package ai.humem.tiervec;

import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.ListRandomAccessByteVectorValues;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.RandomAccessByteVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.disk.GraphIndexWriter;
import io.github.jbellis.jvector.graph.disk.GraphIndexWriterTypes;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.Feature;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.InlineByteVectors;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.management.GraphIndexBuilderConfig;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.ByteVectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ForkJoinPool;
import java.util.function.IntFunction;
import java.util.function.LongPredicate;

/**
 * The JVector arm of the benchmark harness ({@code --engine jvector}, docs/BENCHMARKS.md, issues #108 and #118). It builds one in-memory JVector
 * graph over the same vectors as the tier, with the same {@code efConstruction} and the {@code M} of the run, and answers searches with the beam
 * width {@code ef}, so that {@link Bench} prints the same BUILD, MEMORY, SEARCH and matched-recall lines for both.
 *
 * <p>This file is not part of the library and not compiled by Maven. It lives in {@code src/bench/jvector}, apart from {@code src/bench/java},
 * because it needs the JVector jars (profile {@code bench} of the pom). {@link Bench} reaches it by reflection and works without it. JVector is
 * used through its public builder, searcher and writer API, and no JVector code is copied.
 *
 * <p>The vectors are either float32 ({@code --quant none}) or int8 ({@code --quant int8}). For int8 every value of the index and of a query is
 * multiplied by one factor, {@code 127 / max|x|} over the indexed vectors, rounded and clamped to [-127, 127]; the graph is built and walked with
 * byte by byte dot products. The int8 arm has two search modes: {@code int8} ranks by the byte scores, and {@code int8-rerank} walks on the byte
 * scores and scores the {@code ef} candidates again in float32 (which is what the SQ8 graph of the tier does).
 *
 * <p>What can be the same as in the tier: the vectors, the dimension, {@code efConstruction}, the metric (the vectors have unit length, so the dot
 * product is the cosine), the queries, {@code k}, the beam width of the search, the allowed ids of a filtered search and one search thread. What
 * differs by design: JVector builds the graph in one parallel pass over all vectors (the tier adds them one by one, or rebuilds one segment), it
 * has one graph and no segments, its {@code M} is the maximum degree of a node (the tier allows {@code 2 M} links on its base layer), it prunes
 * with {@code alpha} 1.2 and {@code neighborOverflow} 1.2 (the values of its tutorial, the tier uses {@code alpha} 1.0), it adds the HNSW-style
 * upper layers, and its SIMD provider is chosen by JVector (native code on Java 22 and later when the library is available, else the Vector API).
 */
final class JVectorArm implements Bench.Arm {
    private static final VectorTypeSupport VTS = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final ByteVectorSimilarityFunction BYTE_DOT = ByteVectorSimilarityFunction.DOT_PRODUCT;

    private final int n;
    private final int dim;
    private final ListRandomAccessVectorValues ravv;
    private final RandomAccessByteVectorValues bytes; // null: the float32 arm
    private final float scale;                        // int8: the factor of the conversion
    private final GraphIndexBuilder builder;
    private final ImmutableGraphIndex graph;
    private final GraphSearcher searcher;
    private final ForkJoinPool pool; // null: JVector's own pools
    private final String description;

    JVectorArm(float[][] data, int dim, int m, int efConstruction, float alpha, float overflow, int buildThreads, boolean int8) {
        this.n = data.length;
        this.dim = dim;
        long t0 = System.nanoTime();
        List<VectorFloat<?>> list = new ArrayList<>(data.length);
        for (float[] v : data) list.add(VTS.createFloatVector(v));
        ravv = new ListRandomAccessVectorValues(list, dim);
        if (int8) {
            float max = 0;
            for (float[] v : data) for (float x : v) max = Math.max(max, Math.abs(x));
            if (!(max > 0)) throw new IllegalArgumentException("int8 needs a value that is not 0 in the vectors");
            scale = 127f / max;
            List<ByteSequence<?>> converted = new ArrayList<>(data.length);
            for (float[] v : data) converted.add(toBytes(v));
            bytes = new ListRandomAccessByteVectorValues(converted, dim);
        } else {
            scale = 0;
            bytes = null;
        }
        double convertS = (System.nanoTime() - t0) / 1e9;

        GraphIndexBuilder.Builder b = int8 ? GraphIndexBuilder.builder(bytes, BYTE_DOT, m) : GraphIndexBuilder.builder(ravv, VectorSimilarityFunction.DOT_PRODUCT, m);
        b = b.withBeamWidth(efConstruction).withNeighborOverflow(overflow).withAlpha(alpha);
        if (buildThreads > 0) {
            pool = new ForkJoinPool(buildThreads);
            b = b.withSimdExecutor(pool).withParallelExecutor(pool);
        } else {
            pool = null;
        }
        builder = b.build();
        t0 = System.nanoTime();
        graph = int8 ? builder.build(bytes) : builder.build(ravv);
        double buildS = (System.nanoTime() - t0) / 1e9;
        searcher = new GraphSearcher(graph);

        GraphIndexBuilderConfig cfg = GraphIndexBuilderConfig.getInstance();
        description = String.format(Locale.ROOT,
                "jar=%s provider=%s similarity=DOT_PRODUCT M=%d efConstruction=%d neighborOverflow=%s alpha=%s hierarchy=%b refineFinalGraph=%b levels=%d buildThreads=%s"
                        + " convertSeconds=%.1f graphBuildSeconds=%.1f graphRamBytesMB=%.1f quant=%s int8Scale=%s",
                jarName(), VectorizationProvider.getInstance().getClass().getSimpleName(), m, efConstruction, overflow, alpha, cfg.isAddHierarchy(), cfg.isRefineFinalGraph(),
                graph.getMaxLevel() + 1, buildThreads > 0 ? String.valueOf(buildThreads) : "jvector-default", convertS, buildS, graph.ramBytesUsed() / 1e6,
                int8 ? "int8" : "none", int8 ? String.valueOf(scale) : "none");
    }

    private static String jarName() {
        try {
            String p = GraphIndexBuilder.class.getProtectionDomain().getCodeSource().getLocation().getPath();
            return p.substring(p.lastIndexOf('/') + 1);
        } catch (RuntimeException e) {
            return "unknown";
        }
    }

    private ByteSequence<?> toBytes(float[] v) {
        ByteSequence<?> s = VTS.createByteSequence(v.length);
        for (int i = 0; i < v.length; i++) s.set(i, (byte) Math.max(-127, Math.min(127, Math.round(v[i] * scale))));
        return s;
    }

    @Override
    public List<String> modes() {
        return bytes == null ? List.of("fp32") : List.of("int8", "int8-rerank");
    }

    /**
     * The query is converted to a JVector vector inside the call (the tier's search takes a float[] too). {@code ef} is the beam width: JVector
     * keeps {@code rerankK} candidates in its result queue; without a reranker it returns the best {@code k} of them, and in the mode
     * {@code int8-rerank} it scores all of them in float32 first. The score of the float32 DOT_PRODUCT function is (1 + dot) / 2, and the hit has
     * the dot product, as the tier's hit has. The byte score is {@code dot + 1} for a byte dot product that is not negative and
     * {@code 1 / (1 - dot)} for a negative one; a hit of the mode {@code int8} has that byte dot product divided by the square of the factor, so
     * it is close to the float32 dot product and keeps the order.
     */
    @Override
    public Hit[] search(float[] q, int k, int ef, int mode, LongPredicate allow) {
        Bits bits = allow == null ? Bits.ALL : node -> allow.test(node);
        SearchScoreProvider ssp;
        if (bytes == null) {
            ssp = DefaultSearchScoreProvider.exact(VTS.createFloatVector(q), VectorSimilarityFunction.DOT_PRODUCT, ravv);
        } else {
            ByteSequence<?> qb = toBytes(q);
            if (mode == 0) {
                ScoreFunction.ExactScoreFunction byByte = node -> BYTE_DOT.compare(qb, bytes.getVector(node));
                ssp = new DefaultSearchScoreProvider(byByte);
            } else {
                VectorFloat<?> qv = VTS.createFloatVector(q);
                ScoreFunction.ApproximateScoreFunction walk = node -> BYTE_DOT.compare(qb, bytes.getVector(node));
                ScoreFunction.ExactScoreFunction rerank = node -> VectorSimilarityFunction.DOT_PRODUCT.compare(qv, ravv.getVector(node));
                ssp = new DefaultSearchScoreProvider(walk, rerank);
            }
        }
        SearchResult res = searcher.search(ssp, k, Math.max(ef, k), Float.NEGATIVE_INFINITY, 0.0f, bits);
        SearchResult.NodeScore[] ns = res.getNodes();
        Hit[] h = new Hit[ns.length];
        boolean byteScores = bytes != null && mode == 0;
        for (int i = 0; i < ns.length; i++) {
            float s = ns[i].score;
            h[i] = new Hit(ns[i].node, byteScores ? (s >= 1f ? s - 1f : 1f - 1f / s) / (scale * scale) : 2f * s - 1f);
        }
        return h;
    }

    @Override
    public String describe() {
        return description;
    }

    /** the graph is JVector's own estimate; the vectors are counted by their size (the float32 values can be the arrays of the harness) */
    @Override
    public String memoryBytes() {
        long graphBytes = graph.ramBytesUsed();
        long fp32 = (long) n * dim * 4, int8 = bytes == null ? 0 : (long) n * dim;
        return String.format(Locale.ROOT, "graphBytes=%d fp32VectorBytes=%d int8VectorBytes=%d searchBytes=%d searchWithRerankBytes=%d", graphBytes, fp32, int8,
                graphBytes + (bytes == null ? fp32 : int8), graphBytes + fp32 + int8);
    }

    /** one graph file with the vectors of the arm inline: float32 values, or int8 values (the float32 values for a rerank are not in the int8 file) */
    @Override
    public void writeDisk(Path dir) throws IOException {
        if (bytes == null) {
            OnDiskGraphIndex.write(graph, ravv, dir.resolve("jvector-fp32.graph"));
            return;
        }
        IntFunction<Feature.State> state = node -> new InlineByteVectors.State(bytes.getVector(node));
        try (GraphIndexWriter w = GraphIndexWriter.getBuilderFor(GraphIndexWriterTypes.RANDOM_ACCESS_PARALLEL, graph, dir.resolve("jvector-int8.graph"))
                .with(new InlineByteVectors(dim)).build()) {
            w.write(Map.of(FeatureId.INLINE_BYTE_VECTORS, state));
        }
    }

    @Override
    public void close() throws Exception {
        searcher.close();
        builder.close();
        graph.close();
        if (pool != null) pool.shutdown();
    }
}
