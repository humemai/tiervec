package vq;

import org.openjdk.jmh.annotations.*;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.TimeUnit;

/** Build time, one build per invocation (SingleShotTime; run with -f 1 -wi 0 -i 1: one measurement per JMH fork, ab.py rounds supply the replicates).
 *    phase=index   jvector: quantizer training + encoding + graph build + files written (JvScreen's buildS). ArcadeDB: CREATE INDEX + buildVectorGraphIfPending() on a database
 *                  that was loaded in the untimed @Setup (the Python arm's create_index_s + build_graph_now_s).
 *    phase=ingest  ArcadeDB only: create the database and insert the N documents (the Python arm's load_s). jvector has no ingest (the vectors are an in-memory list).
 *  Memory peaks (sampled every 50 ms) and sizes are printed as RESULT lines. Works in WORKDIR/bb-ARM, deleted after each invocation. */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.SingleShotTime)
@OutputTimeUnit(TimeUnit.SECONDS)
public class BuildBench {
  @Param({"jv-FP32"}) public String arm;
  @Param({""}) public String dir;
  @Param({""}) public String work;
  @Param({"index"}) public String phase;

  Arms.Spec s; U.Data d; AdArm.Handle h; Path bb;

  @Setup(Level.Trial) public void load() throws Exception { s = Arms.spec(arm); d = U.load(Paths.get(dir), true, 1); bb = Paths.get(work).resolve("bb-" + arm); }

  @Setup(Level.Invocation) public void prepare() throws Exception {
    U.deleteTree(bb); h = null;
    if (!s.jv() && !s.lu() && phase.equals("index")) h = AdArm.ingest(s, d, bb.resolve(s.name));
  }

  @TearDown(Level.Invocation) public void cleanup() throws Exception {
    if (h != null && h.db != null) { try { if (h.db.isOpen()) h.db.close(); h.factory.close(); } catch (final Exception e) { } }
    h = null; U.deleteTree(bb);
  }

  static long cpuNs() { return ((com.sun.management.OperatingSystemMXBean) java.lang.management.ManagementFactory.getOperatingSystemMXBean()).getProcessCpuTime(); }

  @Benchmark public Object build() throws Exception {
    final java.util.Map<String, Object> info; final long cpu0 = cpuNs();
    if (s.lu()) { if (phase.equals("ingest")) throw new IllegalStateException("lucene has no separate ingest phase (documents are added inside the build)"); info = Arms.build(s, d, bb); }
    else if (s.jv()) { if (phase.equals("ingest")) throw new IllegalStateException("jvector has no ingest phase"); info = JvArm.build(s, d, bb); }
    else if (phase.equals("ingest")) {
      try (U.Peak peak = new U.Peak()) {
        h = AdArm.ingest(s, d, bb.resolve(s.name)); info = new java.util.LinkedHashMap<>(); info.put("ingest_s", U.r(h.ingestS, 2)); info.put("peak_heap_mb", peak.heapMb()); info.put("peak_rss_mb", peak.rssMb());
      }
    } else { final AdArm.Handle hh = h; h = null; info = AdArm.index(s, d, hh, null); }
    info.put("build_cpu_s", U.r((cpuNs() - cpu0) / 1e9, 2));   // process CPU seconds of the measured body (all threads: workers, JIT, GC); jvector and ArcadeDB build on every core they get, Lucene on one thread
    final java.util.List<Object> kv = new java.util.ArrayList<>(java.util.List.of("layer", 1, "phase", "bench-build-" + phase, "arm", arm, "dataset", d.dataset, "n", d.n, "dim", d.dim));
    info.forEach((k, v) -> { kv.add(k); kv.add(v); });
    final String line = U.json(kv.toArray()); System.out.println("\nRESULT " + line);
    return line;   // returned so JMH cannot drop the build
  }
}
