package vq;

import io.github.jbellis.jvector.disk.ReaderSupplierFactory;
import io.github.jbellis.jvector.graph.*;
import io.github.jbellis.jvector.graph.disk.*;
import io.github.jbellis.jvector.graph.disk.feature.*;
import io.github.jbellis.jvector.graph.similarity.*;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.ByteVectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.nio.file.*;
import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ForkJoinPool;
import java.util.function.IntFunction;
import java.util.stream.IntStream;

/** jv-INT8: jvector 4.1.0-rc.1 (the release jar; contains PR #709 "Add int8 / byte-vector similarity support", merge commit ee81b2e7d6; jv-int8-rc1-20261124/), DIRECTLY, no ArcadeDB. Compiled only in the int8 Maven profile
 *  (pom.xml, build-int8-layer.sh) and run only with L1JVECTOR=int8 (env.sh), because it needs the #709 jar in place of the wheel's jvector-4.0.1.jar. README "jvector INT8 arm" has the rule.
 *  Same graph parameters as JvArm (M 32, beam 100, overflow 1.2, alpha 1.2, no hierarchy, same constructor), byte score provider for the build (BuildScoreProvider.byteVectorScoreProvider),
 *  the graph file written with the PR's tutorial writer (GraphIndexWriter RANDOM_ACCESS_PARALLEL) and InlineByteVectors, search through view.byteVectorRerankerFor (Int8Example).
 *  Byte vectors: ONLY the two conversions the PR itself contains or names (Quant below); no quantizer of our own. R = 0: int8 scores end to end (the fp32 base is on the heap, as for every jv arm, but untouched);
 *  R = 5, 10: int8 walk, the top max(10 R, E) candidates re-scored on fp32 from the heap (the rerankK rule of the other arms). */
public final class JvInt8Arm {
  private JvInt8Arm() { }

  /** The only two float32 -> int8 conversions that exist in #709, see README. uint8-shift: Int8Example's `(byte) ((int) v - 128)` for SIFT (floats that are integers in [0,255]); lossless.
   *  global-scale: the Int8Example comment ("for other float32 datasets you would typically scale by a dataset-specific factor and then clamp before casting"): ONE factor 127 / max|x| over the base, round, clamp to [-127,127].
   *  Queries use the same factor and clamp. No per-vector parameters, no percentile clipping, no tuning. */
  static final class Quant {
    final String mode; final float scale; final int dim;
    Quant(final String mode, final float scale, final int dim) { this.mode = mode; this.scale = scale; this.dim = dim; }
    boolean shift() { return mode.equals("uint8-shift"); }
    static Quant of(final U.Data d) {
      final int dim = d.dim; boolean shift = d.similarity.equals("EUCLIDEAN"); float max = 0f;
      for (final float[] r : d.base) for (final float v : r) {
        if (shift && (v != Math.rint(v) || v < 0f || v > 255f)) shift = false;
        final float a = Math.abs(v); if (a > max) max = a;
      }
      return shift ? new Quant("uint8-shift", 1f, dim) : new Quant("global-scale", 127f / max, dim);
    }
    byte q(final float v) { return shift() ? (byte) ((int) v - 128) : (byte) Math.max(-127, Math.min(127, Math.round(v * scale))); }
    boolean clamped(final float v) { return !shift() && Math.abs(Math.round(v * scale)) > 127; }
    ByteSequence<?> seq(final float[] r) { final ByteSequence<?> b = U.VTS.createByteSequence(dim); for (int i = 0; i < dim; i++) b.set(i, q(r[i])); return b; }
    float deq(final byte b) { return shift() ? (float) (b + 128) : b / scale; }
  }

  static ByteVectorSimilarityFunction bvsf(final U.Data d) { return ByteVectorSimilarityFunction.valueOf(d.similarity); }

  public static Map<String, Object> build(final Arms.Spec s, final U.Data d, final Path work) throws Exception {
    Files.createDirectories(work);
    final int n = d.n, dim = d.dim; final Quant qz = Quant.of(d); final ByteVectorSimilarityFunction BV = bvsf(d);
    final List<ByteSequence<?>> bytes = new ArrayList<>(n); double relErr = 0; final int errN = Math.min(n, 10000);
    for (int i = 0; i < n; i++) {
      final float[] r = d.base[i]; bytes.add(qz.seq(r));
      if (i < errN) { double e = 0, x = 0; for (int j = 0; j < dim; j++) { final double dq = qz.deq(bytes.get(i).get(j)); e += (r[j] - dq) * (r[j] - dq); x += (double) r[j] * r[j]; } relErr += Math.sqrt(e / x); }
    }
    final RandomAccessByteVectorValues rabvv = new ListRandomAccessByteVectorValues(bytes, dim);
    final int cpus = Runtime.getRuntime().availableProcessors(); final ForkJoinPool pool = new ForkJoinPool(cpus);
    final Path gpath = work.resolve(s.name + ".vecgraph"); Files.deleteIfExists(gpath);
    final long t0 = System.nanoTime();
    try (U.Peak peak = new U.Peak()) {
      final BuildScoreProvider bsp = BuildScoreProvider.byteVectorScoreProvider(rabvv, BV);
      final GraphIndexBuilder builder = new GraphIndexBuilder(bsp, dim, JvArm.M, JvArm.BEAM, JvArm.OVERFLOW, JvArm.ALPHA, false, true, pool, pool);
      pool.submit(() -> IntStream.range(0, n).parallel().forEach(i -> builder.addGraphNode(i, bytes.get(i)))).join();
      pool.submit(builder::cleanup).join();
      final ImmutableGraphIndex graph = builder.getGraph();
      final Map<FeatureId, IntFunction<Feature.State>> st = new HashMap<>(); st.put(FeatureId.INLINE_BYTE_VECTORS, o -> new InlineByteVectors.State(bytes.get(o)));
      try (GraphIndexWriter w = GraphIndexWriter.getBuilderFor(GraphIndexWriterTypes.RANDOM_ACCESS_PARALLEL, graph, gpath).with(new InlineByteVectors(dim)).build()) { w.write(st); }
      final double buildS = (System.nanoTime() - t0) / 1e9; pool.shutdown();
      final Map<String, Object> info = new LinkedHashMap<>();
      info.put("build_s", U.r(buildS, 2)); info.put("graph_bytes", Files.size(gpath)); info.put("codes_bytes", 0L); info.put("nvq_bytes", 0L); info.put("int8_bytes", (long) n * dim); info.put("fp32_bytes", (long) n * dim * 4);
      info.put("peak_heap_mb", peak.heapMb()); info.put("peak_rss_mb", peak.rssMb()); info.put("vmhwm_mb", U.hwmMb());
      info.put("m", JvArm.M); info.put("beam_width", JvArm.BEAM); info.put("neighbor_overflow", JvArm.OVERFLOW); info.put("alpha", JvArm.ALPHA); info.put("cpus", cpus);
      info.put("int8_mode", qz.mode); info.put("int8_scale", qz.scale); info.put("int8_rel_err_mean_first10k", U.r(relErr / errN, 5)); info.put("similarity", BV.name());
      return info;
    }
  }

  public static Arms.Searcher open(final Arms.Spec s, final U.Data d, final Path work) throws Exception {
    final int n = d.n; final Quant qz = Quant.of(d); final ByteVectorSimilarityFunction BV = bvsf(d); final VectorSimilarityFunction VSF = JvArm.vsf(d);
    final Path gpath = work.resolve(s.name + ".vecgraph"); if (!Files.exists(gpath)) throw new IllegalStateException("not built: " + gpath);
    final List<VectorFloat<?>> baseV = JvArm.wrap(d.base); final VectorFloat<?>[] qv = new VectorFloat<?>[d.nq]; final ByteSequence<?>[] qb = new ByteSequence<?>[d.nq]; long clampedQ = 0;
    for (int i = 0; i < d.nq; i++) { qv[i] = U.vec(d.queries[i]); qb[i] = qz.seq(d.queries[i]); for (final float v : d.queries[i]) if (qz.clamped(v)) clampedQ++; }
    final OnDiskGraphIndex disk = OnDiskGraphIndex.load(ReaderSupplierFactory.open(gpath)); final long clampedQueryValues = clampedQ;
    final ConcurrentLinkedQueue<GraphSearcher> all = new ConcurrentLinkedQueue<>();
    final ThreadLocal<GraphSearcher> tl = ThreadLocal.withInitial(() -> { final GraphSearcher g = new GraphSearcher(disk); all.add(g); return g; });
    return new Arms.Searcher() {
      @Override public Object search(final int qi, final int ef, final int rr) {
        final GraphSearcher gs = tl.get(); final VectorFloat<?> q = qv[qi];
        final int efEff = ef != 0 ? ef : (n < 10_000 ? 100 : 20);       // ArcadeDB's default beam (README "efSearch values")
        final int rerankK = Math.max(10 * Math.max(rr, 1), efEff);
        final ScoreFunction.ExactScoreFunction bf = ((OnDiskGraphIndex.View) gs.getView()).byteVectorRerankerFor(qb[qi], BV);
        final SearchScoreProvider ssp;
        if (rr == 0) ssp = new DefaultSearchScoreProvider(bf);
        else {
          final ScoreFunction.ApproximateScoreFunction approx = new ScoreFunction.ApproximateScoreFunction() { @Override public float similarityTo(final int node) { return bf.similarityTo(node); } };
          final ScoreFunction.ExactScoreFunction fp = o -> VSF.compare(q, baseV.get(o));
          ssp = new DefaultSearchScoreProvider(approx, fp);
        }
        return gs.search(ssp, 10, rerankK, 0.0f, 0.0f, Bits.ALL);
      }
      @Override public int[] ids(final Object raw) {
        final SearchResult.NodeScore[] ns = ((SearchResult) raw).getNodes().clone(); Arrays.sort(ns, (a, b) -> Float.compare(b.score, a.score));
        final int[] out = new int[ns.length]; for (int i = 0; i < ns.length; i++) out[i] = ns[i].node; return out;
      }
      @Override public Object[] info() {
        return new Object[] {"api", "jvector 4.1.0-rc.1 GraphSearcher.search over OnDiskGraphIndex, InlineByteVectors, view.byteVectorRerankerFor", "vectorization", U.vectorization(), "m", JvArm.M, "beam_width", JvArm.BEAM,
            "int8_mode", qz.mode, "int8_scale", qz.scale, "similarity", BV.name(), "clamped_query_values", clampedQueryValues};
      }
      @Override public void close() { for (final GraphSearcher g : all) { try { g.close(); } catch (final Exception e) { } } try { disk.close(); } catch (final Exception e) { } }
    };
  }
}
