package vq;

import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.disk.ReaderSupplier;
import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.disk.SimpleWriter;
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.graph.disk.*;
import io.github.jbellis.jvector.graph.disk.feature.*;
import io.github.jbellis.jvector.graph.similarity.*;
import io.github.jbellis.jvector.quantization.*;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.nio.file.*;
import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ForkJoinPool;
import java.util.function.IntFunction;
import java.util.stream.IntStream;

/** jvector 4.0.1 DIRECTLY (the jar the wheel ships), no ArcadeDB. Same graph parameters as LSMVectorIndex (main cec5eb8a19 and the 26.10.2 metadata defaults):
 *  M (maxConnections) 32, beamWidth 100, neighborOverflowFactor 1.2, alphaDiversityRelaxation 1.2, addHierarchy false, concurrent updates true, graph file = topology only
 *  (InlineVectors(0)); PQ: 256 clusters, centred globally, trained on a fixed-seed sample of min(N, 128000). The build is JvScreen.java's, written to disk so that a query process can
 *  reopen it (what a reopened index walks). The fp32 vectors stay on the heap (the rerank source); NVQ re-ranks from the graph file (view.rerankerFor). */
final class JvArm {
  private JvArm() { }
  static final int M = 32, BEAM = 100; static final float OVERFLOW = 1.2f, ALPHA = 1.2f;

  /** Thrown for an arm that cannot run here; the caller prints SKIP and a RESULT row with phase=skip. */
  static final class Skip extends RuntimeException { Skip(final String m) { super(m); } }

  /** jv-INT8 is routed to JvInt8Arm by Arms (int8 build, L1JVECTOR=int8); reaching here with INT8 means that route was bypassed. */
  static void checkInt8(final Arms.Spec s) {
    if (s.kind.equals("INT8")) throw new Skip("jv-INT8 runs only in the int8 build (jvector 4.1.0-rc.1 in jv-int8-rc1-20261124/, L1JVECTOR=int8); see README 'jvector INT8 arm'");
  }

  static List<VectorFloat<?>> wrap(final float[][] base) { final List<VectorFloat<?>> l = new ArrayList<>(base.length); for (final float[] r : base) l.add(U.vec(r)); return l; }
  static VectorSimilarityFunction vsf(final U.Data d) { return VectorSimilarityFunction.valueOf(d.similarity); }

  static Map<String, Object> build(final Arms.Spec s, final U.Data d, final Path work) throws Exception {
    checkInt8(s); Files.createDirectories(work);
    final VectorSimilarityFunction VSF = vsf(d); final int n = d.n, dim = d.dim; final List<VectorFloat<?>> base = wrap(d.base);
    final RandomAccessVectorValues ravv = new ListRandomAccessVectorValues(base, dim);
    final int cpus = Runtime.getRuntime().availableProcessors(); final ForkJoinPool pool = new ForkJoinPool(cpus);
    final Path gpath = work.resolve(s.name + ".vecgraph"), ppath = work.resolve(s.name + ".vecpq"), bpath = work.resolve(s.name + ".vecbq");
    Files.deleteIfExists(gpath); Files.deleteIfExists(ppath); Files.deleteIfExists(bpath);
    final long t0 = System.nanoTime(); MutablePQVectors pqv = null; BQVectors bqv = null; BuildScoreProvider bsp;
    try (U.Peak peak = new U.Peak()) {
      if (s.kind.equals("PQ") || s.kind.equals("NVQ")) {
        final int sample = Math.min(n, 128000); final List<VectorFloat<?>> train;
        if (sample >= n) train = base; else { final int[] ord = new Random(42).ints(0, n).distinct().limit(sample).toArray(); train = new ArrayList<>(sample); for (final int o : ord) train.add(base.get(o)); }
        if (s.subs < 1 || dim % s.subs != 0) throw new IllegalArgumentException("sub must divide dim: " + s.subs + " " + dim);
        final ProductQuantization pq = ProductQuantization.compute(new ListRandomAccessVectorValues(train, dim), s.subs, 256, true);
        pqv = new MutablePQVectors(pq); for (int i = 0; i < n; i++) pqv.encodeAndSet(i, base.get(i));
        bsp = BuildScoreProvider.pqBuildScoreProvider(VSF, pqv);
      } else {
        bsp = BuildScoreProvider.randomAccessScoreProvider(ravv, VSF);
        if (s.kind.equals("BQ")) bqv = (BQVectors) BinaryQuantization.compute(ravv, pool).encodeAll(ravv, pool);
      }
      final GraphIndexBuilder builder = new GraphIndexBuilder(bsp, dim, M, BEAM, OVERFLOW, ALPHA, false, true, pool, pool);
      pool.submit(() -> IntStream.range(0, n).parallel().forEach(i -> builder.addGraphNode(i, base.get(i)))).join();
      pool.submit(builder::cleanup).join();
      final ImmutableGraphIndex graph = builder.getGraph();
      NVQuantization nvq = null;
      if (s.kind.equals("NVQ")) {
        final io.github.jbellis.jvector.vector.types.VectorFloat<?> mean = U.VTS.createFloatVector(dim); final double[] sum = new double[dim];
        for (final float[] v : d.base) for (int j = 0; j < dim; j++) sum[j] += v[j];
        for (int j = 0; j < dim; j++) mean.set(j, (float) (sum[j] / n));
        nvq = NVQuantization.create(mean, 2);
      }
      final NVQuantization nvqF = nvq;
      try (SimpleWriter w = new SimpleWriter(gpath)) {
        final OnDiskSequentialGraphIndexWriter.Builder wb = new OnDiskSequentialGraphIndexWriter.Builder(graph, w);
        final Map<FeatureId, IntFunction<Feature.State>> st = new HashMap<>();
        if (nvqF != null) { wb.with(new NVQ(nvqF)); st.put(FeatureId.NVQ_VECTORS, o -> new NVQ.State(nvqF.encode(base.get(o)))); }
        else { wb.with(new InlineVectors(0)); final VectorFloat<?> empty = U.VTS.createFloatVector(0); st.put(FeatureId.INLINE_VECTORS, o -> new InlineVectors.State(empty)); }
        try (OnDiskSequentialGraphIndexWriter iw = wb.build()) { iw.write(st); }
      }
      long codes = 0;
      if (pqv != null) { try (SimpleWriter pw = new SimpleWriter(ppath)) { pqv.write(pw); } codes = Files.size(ppath); }
      if (bqv != null) { try (SimpleWriter bw = new SimpleWriter(bpath)) { bqv.write(bw); } codes = Files.size(bpath); }
      final double buildS = (System.nanoTime() - t0) / 1e9; pool.shutdown();
      final Map<String, Object> info = new LinkedHashMap<>();
      info.put("build_s", U.r(buildS, 2)); info.put("graph_bytes", Files.size(gpath)); info.put("codes_bytes", codes);
      info.put("nvq_bytes", nvq == null ? 0 : (long) nvq.compressedVectorSize() * n); info.put("fp32_bytes", (long) n * dim * 4);
      info.put("peak_heap_mb", peak.heapMb()); info.put("peak_rss_mb", peak.rssMb()); info.put("vmhwm_mb", U.hwmMb());
      info.put("m", M); info.put("beam_width", BEAM); info.put("neighbor_overflow", OVERFLOW); info.put("alpha", ALPHA); info.put("cpus", cpus);
      return info;
    }
  }

  static Arms.Searcher open(final Arms.Spec s, final U.Data d, final Path work) throws Exception {
    checkInt8(s);
    final VectorSimilarityFunction VSF = vsf(d); final int n = d.n;
    final Path gpath = work.resolve(s.name + ".vecgraph"), ppath = work.resolve(s.name + ".vecpq"), bpath = work.resolve(s.name + ".vecbq");
    if (!Files.exists(gpath)) throw new IllegalStateException("not built: " + gpath);
    final List<VectorFloat<?>> baseV = wrap(d.base); final VectorFloat<?>[] qv = new VectorFloat<?>[d.nq]; for (int i = 0; i < d.nq; i++) qv[i] = U.vec(d.queries[i]);
    final OnDiskGraphIndex disk = OnDiskGraphIndex.load(ReaderSupplierFactory.open(gpath));
    PQVectors pq = null; BQVectors bq = null;
    if (s.kind.equals("PQ") || s.kind.equals("NVQ")) { try (ReaderSupplier rs = ReaderSupplierFactory.open(ppath); RandomAccessReader rd = rs.get()) { pq = PQVectors.load(rd); } }
    if (s.kind.equals("BQ")) { try (ReaderSupplier rs = ReaderSupplierFactory.open(bpath); RandomAccessReader rd = rs.get()) { bq = BQVectors.load(rd, 0L); } }
    final PQVectors pqF = pq; final BQVectors bqF = bq; final String kind = s.kind;
    final ConcurrentLinkedQueue<GraphSearcher> all = new ConcurrentLinkedQueue<>();
    final ThreadLocal<GraphSearcher> tl = ThreadLocal.withInitial(() -> { final GraphSearcher g = new GraphSearcher(disk); all.add(g); return g; });
    return new Arms.Searcher() {
      @Override public Object search(final int qi, final int ef, final int rr) {
        final VectorFloat<?> q = qv[qi]; final GraphSearcher gs = tl.get();
        final int efEff = ef != 0 ? ef : (n < 10_000 ? 100 : 20);       // ArcadeDB's default beam (README "efSearch values")
        final int rerankK = Math.max(10 * Math.max(rr, 1), efEff);
        final ScoreFunction.ExactScoreFunction fp = o -> VSF.compare(q, baseV.get(o));
        final SearchScoreProvider ssp;
        if (kind.equals("FP32")) ssp = new DefaultSearchScoreProvider(fp);
        else {
          final ScoreFunction.ApproximateScoreFunction approx = pqF != null ? pqF.precomputedScoreFunctionFor(q, VSF) : bqF.precomputedScoreFunctionFor(q, VSF);
          if (rr == 0) ssp = new DefaultSearchScoreProvider(approx);
          else ssp = new DefaultSearchScoreProvider(approx, kind.equals("NVQ") ? ((OnDiskGraphIndex.View) gs.getView()).rerankerFor(q, VSF) : fp);
        }
        return gs.search(ssp, 10, rerankK, 0.0f, 0.0f, Bits.ALL);
      }
      @Override public int[] ids(final Object raw) {
        final SearchResult.NodeScore[] ns = ((SearchResult) raw).getNodes().clone(); Arrays.sort(ns, (a, b) -> Float.compare(b.score, a.score));
        final int[] out = new int[ns.length]; for (int i = 0; i < ns.length; i++) out[i] = ns[i].node; return out;
      }
      @Override public Object[] info() { return new Object[] {"api", "jvector (the jar on the classpath, see the MANIFEST) GraphSearcher.search over OnDiskGraphIndex", "vectorization", U.vectorization(), "m", M, "beam_width", BEAM}; }
      @Override public void close() { for (final GraphSearcher g : all) { try { g.close(); } catch (final Exception e) { } } try { disk.close(); } catch (final Exception e) { } }
    };
  }
}
