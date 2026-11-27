package vq;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.nio.file.Paths;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/** One query per invocation against an index built beforehand by Layer1Main build (same JVM flags as the engine). Select the metric on the command line:
 *    -bm sample -tu us   latency percentiles (primaryMetric.scorePercentiles: p50 / p95 / p99 in us), -t 1 or -t N
 *    -bm thrpt  -tu s    QPS at -t 1 / -t N threads
 *  Parameters: -p arm=ad-PQ16 -p dir=DATASETDIR -p work=WORKDIR -p ef=100 -p rerank=10 [-p nq=0]. ef and rerank come from the matched-recall calibration (Layer1Main query: RESULT phase=calib).
 *  The setup opens the index in this fork and runs one warm pass over all queries; nothing but search() is inside the measured method (no SQL, no record load, no id mapping). */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.SampleTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
public class QueryBench {
  @Param({"jv-FP32"}) public String arm;
  @Param({""}) public String dir;
  @Param({""}) public String work;
  @Param({"0"}) public int ef;
  @Param({"0"}) public int rerank;
  @Param({"0"}) public int nq;

  Arms.Searcher sr; int queries; static final AtomicInteger SEED = new AtomicInteger();

  @State(Scope.Thread) public static class T { int i = SEED.getAndAdd(97); }

  /** Full path of the jvector jar this JVM runs on (the re-run's per-arm jar identity line, jv-int8-rc1-20261124). */
  static String jvJar() {
    try { return java.nio.file.Paths.get(io.github.jbellis.jvector.vector.VectorizationProvider.class.getProtectionDomain().getCodeSource().getLocation().toURI()).toString(); }
    catch (final Exception e) { return "unknown"; }
  }

  @Setup(Level.Trial) public void open() throws Exception {
    final Arms.Spec s = Arms.spec(arm); final U.Data d = U.load(Paths.get(dir), Arms.needsBase(s), nq);
    sr = Arms.open(s, d, Paths.get(work)); queries = d.nq;
    System.out.println("INFO QueryBench arm=" + arm + " n=" + d.n + " ef=" + ef + " rerank=" + rerank + " jvector_jar=" + jvJar() + " " + U.json(sr.info()));
    for (int qi = 0; qi < queries; qi++) sr.search(qi, ef, rerank);
    System.out.println("\nRESULT " + U.json("layer", 1, "phase", "bench-mem", "arm", arm, "n", d.n, "heap_used_gc_mb", U.heapAfterGcMb(), "rss_mb", U.rssMb(), "rss_anon_mb", U.anonMb(), "rss_file_mb", U.fileMb(), "vmhwm_mb", U.hwmMb(), "direct_mb", U.directMb()));
  }

  @TearDown(Level.Trial) public void close() { if (sr != null) sr.close(); }

  @Benchmark public void query(final T t, final Blackhole bh) throws Exception {
    final int qi = Math.floorMod(t.i++, queries);
    bh.consume(sr.search(qi, ef, rerank));
  }
}
