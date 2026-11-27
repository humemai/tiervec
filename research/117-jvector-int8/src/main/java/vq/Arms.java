package vq;

import java.nio.file.Path;
import java.util.*;

/** Arm names and the common Searcher contract.
 *  jvector standalone : jv-FP32 | jv-PQ<S> | jv-BQ | jv-NVQ<S> | jv-INT8 (#665, SKIP unless the int8-support jar is on the classpath)
 *  Lucene 10.5.2      : lu-FP32 | lu-BBQ1 | lu-BBQ2 | lu-SQ8 (+ lu-SQ7, lu-SQ4); LuArm, compiled only in the Maven profile lucene (L1LUCENE=1), never part of all()
 *  ArcadeDB Java API  : ad-NONE | ad-INT8 | ad-BINARY | ad-PQ<S> (quantization PRODUCT, 256 clusters); ad-NONE-SVG = ad-NONE with storeVectorsInGraph=true (never part of all(); named in ARMS only) */
public final class Arms {
  private Arms() { }

  public static final class Spec {
    public final String name, engine, kind; public final int subs; public final boolean svg;   // svg: "-SVG" suffix, metadata storeVectorsInGraph=true (only ad-NONE-SVG; README "storeVectorsInGraph arm")
    Spec(final String name) {
      this.name = name; final int dash = name.indexOf('-'); if (dash < 0) throw new IllegalArgumentException("arm " + name);
      engine = name.substring(0, dash); String k = name.substring(dash + 1);
      svg = k.endsWith("-SVG"); if (svg) { if (!engine.equals("ad") || !k.equals("NONE-SVG")) throw new IllegalArgumentException("only ad-NONE-SVG has the -SVG suffix: " + name); k = k.substring(0, k.length() - 4); }
      final java.util.regex.Matcher m = java.util.regex.Pattern.compile("(PQ|NVQ)(\\d+)").matcher(k);
      if (m.matches()) { kind = m.group(1); subs = Integer.parseInt(m.group(2)); } else { kind = k; subs = 0; }
      if (!engine.equals("jv") && !engine.equals("ad") && !engine.equals("lu")) throw new IllegalArgumentException("engine " + engine);
    }
    public boolean jv() { return engine.equals("jv"); }
    public boolean lu() { return engine.equals("lu"); }
    /** Rerank sweep default: jvector PQ/BQ 0,5,10 (BQ also 20,50), NVQ 5,10; ArcadeDB PQ: -1 = the engine's own path (SQL equivalent), and 0,5,10 where the engine takes a rerank; others 0. */
    public int[] defaultReranks() {
      if (lu()) return kind.equals("FP32") ? new int[] {0} : new int[] {0, 5, 10};   // lucene (README "Lucene arms"): the LuceneScreen grid
      switch (kind) {
        case "PQ": return jv() ? new int[] {0, 5, 10} : new int[] {-1, 0, 5, 10};
        case "BQ": return new int[] {0, 5, 10, 20, 50};
        case "NVQ": return new int[] {5, 10};
        case "BINARY": return !jv() && System.getenv("L1BINRR") != null ? new int[] {0, 2, 5, 10, 20, 50} : new int[] {0};   // ArcadeDB PR #10975 (README "BINARY search, PR 10975"): L1BINRR set = the engine reranks max(10 R, efSearch) candidates for R (0 and 1 are the same search: at least one per neighbour); unset = as before (BINARY ignored the argument)
        case "INT8": return jv() ? new int[] {0, 5, 10} : new int[] {0};   // jv-INT8: 0 = int8 scores end to end (its own setting), 5 and 10 = int8 walk + fp32 rerank (README "jvector INT8 arm")
        default: return new int[] {0};
      }
    }
  }
  public static Spec spec(final String name) { return new Spec(name); }

  /** The arms of a cell: S in {dim/16, dim/8, dim/4} as the README lists. */
  public static List<String> all(final int dim) {
    final int s1 = dim / 16, s2 = dim / 8, s3 = dim / 4;
    return List.of("jv-FP32", "jv-PQ" + s1, "jv-PQ" + s2, "jv-PQ" + s3, "jv-BQ", "jv-NVQ" + s3, "jv-INT8", "ad-NONE", "ad-INT8", "ad-BINARY", "ad-PQ" + s1, "ad-PQ" + s2, "ad-PQ" + s3);
  }

  /** The Lucene reference arms of the timed lucene / build studies (not in all(): the existing studies list jv and ad arms only). */
  public static List<String> luArms() { return List.of("lu-FP32", "lu-BBQ1", "lu-BBQ2", "lu-SQ8"); }

  /** A query-side handle on a built arm. Thread-safe: every thread may call search. */
  public interface Searcher extends AutoCloseable {
    /** One top-10 query (query index qi of the dataset); ef 0 = the engine's default beam; rr = rerank (meaning per arm). Returns the raw result (consumed by the benchmark, converted by ids()). */
    Object search(int qi, int ef, int rr) throws Exception;
    /** Dataset row ids of a raw result, best first (untimed: recall). */
    int[] ids(Object raw) throws Exception;
    /** The rerank values this engine can actually vary (ArcadeDB PRODUCT on an engine without findNeighbors: only -1 = its own path and 0 = PQ scores alone). */
    default int[] rerankChoices(final int[] wanted) { return wanted; }
    /** key/value pairs for the INFO line (API used, index parameters). */
    Object[] info();
    @Override void close();
  }

  /** jv-INT8 lives in JvInt8Arm, compiled only in the int8 Maven profile against the jvector #709 jar (reflection: the default 4.0.1 build has no such class and prints SKIP). */
  private static boolean jvInt8(final Spec s) { return s.jv() && s.kind.equals("INT8"); }
  private static Object int8(final String method, final Spec s, final U.Data d, final Path work) throws Exception {
    final Class<?> c;
    try { c = Class.forName("vq.JvInt8Arm"); }
    catch (final ClassNotFoundException e) { throw new JvArm.Skip("jv-INT8 needs the jvector #709 build: run with L1JVECTOR=int8 (README 'jvector INT8 arm'); this classpath has jvector 4.0.1 and no JvInt8Arm"); }
    try { return c.getMethod(method, Spec.class, U.Data.class, Path.class).invoke(null, s, d, work); }
    catch (final java.lang.reflect.InvocationTargetException e) { if (e.getCause() instanceof Exception) throw (Exception) e.getCause(); throw e; }
  }
  private static Object lucene(final String method, final Spec s, final U.Data d, final Path work) throws Exception {
    final Class<?> c;
    try { c = Class.forName("vq.LuArm"); }
    catch (final ClassNotFoundException e) { throw new JvArm.Skip("lu-* arms need the Lucene build: run with L1LUCENE=1 (build-lucene-layer.sh); this classpath has no vq.LuArm"); }
    try { return c.getMethod(method, Spec.class, U.Data.class, Path.class).invoke(null, s, d, work); }
    catch (final java.lang.reflect.InvocationTargetException e) { if (e.getCause() instanceof Exception) throw (Exception) e.getCause(); throw e; }
  }
  @SuppressWarnings("unchecked")
  public static Map<String, Object> build(final Spec s, final U.Data d, final Path work) throws Exception {
    if (s.lu()) return (Map<String, Object>) lucene("build", s, d, work);
    if (jvInt8(s)) return (Map<String, Object>) int8("build", s, d, work);
    return s.jv() ? JvArm.build(s, d, work) : AdArm.build(s, d, work);
  }
  public static Searcher open(final Spec s, final U.Data d, final Path work) throws Exception {
    if (s.lu()) return (Searcher) lucene("open", s, d, work);
    if (jvInt8(s)) return (Searcher) int8("open", s, d, work);
    return s.jv() ? JvArm.open(s, d, work) : AdArm.open(s, d, work);
  }
  /** True when the arm needs the fp32 base on the heap in the query process. */
  public static boolean needsBase(final Spec s) { return s.jv(); }
}
