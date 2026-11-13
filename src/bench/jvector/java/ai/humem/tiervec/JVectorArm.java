package ai.humem.tiervec;

import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.management.GraphIndexBuilderConfig;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ForkJoinPool;

/**
 * The JVector arm of the benchmark harness ({@code --engine jvector}, docs/BENCHMARKS.md, issue #108). It builds one in-memory JVector graph
 * over the same vectors as the tier, with the same {@code M} and {@code efConstruction}, and answers unfiltered searches with the beam width
 * {@code ef}, so that {@link Bench} prints the same BUILD, MEMORY and SEARCH lines for both.
 *
 * <p>This file is not part of the library and not compiled by Maven. It lives in {@code src/bench/jvector}, apart from {@code src/bench/java},
 * because it needs the JVector jars (profile {@code bench} of the pom, JVector 4.0.2). {@link Bench} reaches it by reflection and works
 * without it. JVector is used through its public builder and searcher API, and no JVector code is copied.
 *
 * <p>What is the same as in the tier: the vectors (float32, no quantization), the dimension, {@code M}, {@code efConstruction}, the metric (the
 * vectors have unit length, so the dot product is the cosine), the queries, {@code k}, the beam width of the search and one search thread. What
 * differs by design: JVector builds the graph in one parallel pass over all vectors (the tier adds them one by one, or rebuilds one segment),
 * it has one graph and no segments, it prunes with {@code alpha} 1.2 and {@code neighborOverflow} 1.2 (the values of its tutorial, the tier uses
 * {@code alpha} 1.0), it adds the HNSW-style upper layers, and its SIMD provider is chosen by JVector (native code on Java 22 and later when the
 * library is available, else the Vector API).
 */
final class JVectorArm implements Bench.Arm {
    private static final VectorTypeSupport VTS = VectorizationProvider.getInstance().getVectorTypeSupport();

    private final ListRandomAccessVectorValues ravv;
    private final GraphIndexBuilder builder;
    private final ImmutableGraphIndex graph;
    private final GraphSearcher searcher;
    private final ForkJoinPool pool; // null: JVector's own pools
    private final String description;

    JVectorArm(float[][] data, int dim, int m, int efConstruction, float alpha, float overflow, int buildThreads) {
        long t0 = System.nanoTime();
        List<VectorFloat<?>> list = new ArrayList<>(data.length);
        for (float[] v : data) list.add(VTS.createFloatVector(v));
        ravv = new ListRandomAccessVectorValues(list, dim);
        double convertS = (System.nanoTime() - t0) / 1e9;

        GraphIndexBuilder.Builder b = GraphIndexBuilder.builder(ravv, VectorSimilarityFunction.DOT_PRODUCT, m)
                .withBeamWidth(efConstruction)
                .withNeighborOverflow(overflow)
                .withAlpha(alpha);
        if (buildThreads > 0) {
            pool = new ForkJoinPool(buildThreads);
            b = b.withSimdExecutor(pool).withParallelExecutor(pool);
        } else {
            pool = null;
        }
        builder = b.build();
        t0 = System.nanoTime();
        graph = builder.build(ravv);
        double buildS = (System.nanoTime() - t0) / 1e9;
        searcher = new GraphSearcher(graph);

        GraphIndexBuilderConfig cfg = GraphIndexBuilderConfig.getInstance();
        description = String.format(Locale.ROOT,
                "jar=%s provider=%s similarity=DOT_PRODUCT M=%d efConstruction=%d neighborOverflow=%s alpha=%s hierarchy=%b refineFinalGraph=%b levels=%d buildThreads=%s"
                        + " convertSeconds=%.1f graphBuildSeconds=%.1f graphRamBytesMB=%.1f",
                jarName(), VectorizationProvider.getInstance().getClass().getSimpleName(), m, efConstruction, overflow, alpha, cfg.isAddHierarchy(), cfg.isRefineFinalGraph(),
                graph.getMaxLevel() + 1, buildThreads > 0 ? String.valueOf(buildThreads) : "jvector-default", convertS, buildS, graph.ramBytesUsed() / 1e6);
    }

    private static String jarName() {
        try {
            String p = GraphIndexBuilder.class.getProtectionDomain().getCodeSource().getLocation().getPath();
            return p.substring(p.lastIndexOf('/') + 1);
        } catch (RuntimeException e) {
            return "unknown";
        }
    }

    /**
     * The query is converted to a JVector vector inside the call (the tier's search takes a float[] too). {@code ef} is the beam width: JVector
     * keeps {@code rerankK} candidates in its result queue, and without a reranker it returns the best {@code k} of them. The score of the
     * DOT_PRODUCT function is (1 + dot) / 2, and the hit has the dot product, as the tier's hit has.
     */
    @Override
    public Hit[] search(float[] q, int k, int ef) {
        VectorFloat<?> qv = VTS.createFloatVector(q);
        SearchScoreProvider ssp = DefaultSearchScoreProvider.exact(qv, VectorSimilarityFunction.DOT_PRODUCT, ravv);
        SearchResult res = searcher.search(ssp, k, Math.max(ef, k), Float.NEGATIVE_INFINITY, 0.0f, Bits.ALL);
        SearchResult.NodeScore[] ns = res.getNodes();
        Hit[] h = new Hit[ns.length];
        for (int i = 0; i < ns.length; i++) h[i] = new Hit(ns[i].node, 2f * ns[i].score - 1f);
        return h;
    }

    @Override
    public String describe() {
        return description;
    }

    @Override
    public void close() throws Exception {
        searcher.close();
        builder.close();
        graph.close();
        if (pool != null) pool.shutdown();
    }
}
