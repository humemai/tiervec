package vq;

import java.nio.file.*;
import java.util.*;

/** Layer 1 CLI (untimed: recall, memory, disk, matched-recall calibration; the TIMED figures come from the JMH classes QueryBench / BuildBench).
 *    list  --dim D [--set lucene]                    the arm names of a cell (--set lucene: the Lucene arms of the lucene / build studies)
 *    build --arm A --dir DATASETDIR --work WORKDIR   build once (jvector: graph + codes files; ArcadeDB: database), RESULT phase=build
 *    query --arm A --dir DATASETDIR --work WORKDIR [--nq N] [--efs 0,20,..] [--rerank 0,5,10] [--target 0.95]
 *                                                    reopen in this fresh process, warm, steady-state memory, recall@10/recall@1 sweep over efSearch x rerank,
 *                                                    then RESULT phase=calib: the cheapest tested setting (smallest single-thread search time of the sweep pass, indicative) with recall@10 >= target.
 *  Every step ends with a closing marker line (BUILD ok / QUERY ok / SKIP) after >= 1 RESULT line, the run_screen.sh convention. */
public final class Layer1Main {
  public static void main(final String[] args) throws Exception {
    final Map<String, String> a = new HashMap<>(); for (int i = 1; i + 1 < args.length; i += 2) a.put(args[i].substring(2), args[i + 1]);
    final String mode = args[0];
    if (mode.equals("list")) { (a.getOrDefault("set", "").equals("lucene") ? Arms.luArms() : Arms.all(Integer.parseInt(a.get("dim")))).forEach(System.out::println); return; }   // --set lucene: the lu arms
    final Arms.Spec s = Arms.spec(a.get("arm")); final Path dir = Paths.get(a.get("dir")), work = Paths.get(a.get("work")); Files.createDirectories(work);
    final int nq = Integer.parseInt(a.getOrDefault("nq", "0"));
    final U.Data d = U.load(dir, mode.equals("build") || Arms.needsBase(s), nq);
    System.out.printf("INFO layer1 mode=%s arm=%s dataset=%s n=%d dim=%d sim=%s nq=%d java=%s cpus=%d maxHeapMB=%d vectorization=%s jvector=%s%n", mode, s.name, d.dataset, d.n, d.dim, d.similarity, d.nq,
        System.getProperty("java.version"), Runtime.getRuntime().availableProcessors(), Runtime.getRuntime().maxMemory() >> 20, U.vectorization(), jvectorJar());
    try {
      if (mode.equals("build")) {
        final Map<String, Object> info = Arms.build(s, d, work);
        final List<Object> kv = new ArrayList<>(List.of("layer", 1, "phase", "build", "arm", s.name, "dataset", d.dataset, "n", d.n, "dim", d.dim));
        info.forEach((k, v) -> { kv.add(k); kv.add(v); });
        U.result(kv.toArray()); System.out.println("BUILD ok");
      } else if (mode.equals("query")) query(s, d, work, a);
      else throw new IllegalArgumentException("mode " + mode);
    } catch (final JvArm.Skip e) {
      System.out.println("SKIP " + s.name + ": " + e.getMessage());
      U.result("layer", 1, "phase", "skip", "arm", s.name, "dataset", d.dataset, "n", d.n, "dim", d.dim, "reason", e.getMessage());
    }
  }

  /** The jvector jar this JVM runs on (4.0.1 from the wheel, or the #709 build in the int8 variant). */
  static String jvectorJar() {
    try { return Paths.get(io.github.jbellis.jvector.vector.VectorizationProvider.class.getProtectionDomain().getCodeSource().getLocation().toURI()).getFileName().toString(); }
    catch (final Exception e) { return "unknown"; }
  }

  static int[] ints(final String csv) { return Arrays.stream(csv.split(",")).mapToInt(Integer::parseInt).toArray(); }

  static void query(final Arms.Spec s, final U.Data d, final Path work, final Map<String, String> a) throws Exception {
    final int[] efs = ints(a.getOrDefault("efs", "0,20,50,100,200,400")); int[] rrs = a.containsKey("rerank") ? ints(a.get("rerank")) : s.defaultReranks();
    final double target = Double.parseDouble(a.getOrDefault("target", "0.95"));
    try (U.Peak peak = new U.Peak(); Arms.Searcher sr = Arms.open(s, d, work)) {
      if (!a.containsKey("rerank")) rrs = sr.rerankChoices(rrs);
      final List<Object> info = new ArrayList<>(List.of("layer", 1, "phase", "info", "arm", s.name, "dataset", d.dataset, "n", d.n)); info.addAll(Arrays.asList(sr.info())); U.result(info.toArray());
      for (int qi = 0; qi < d.nq; qi++) sr.search(qi, 0, rrs[0]);                       // warm-up: loads the graph / PQ codes, JIT
      final long tq = System.nanoTime(); for (int qi = 0; qi < d.nq; qi++) sr.search(qi, 0, rrs[0]);    // the steady-state query pass
      final double passS = (System.nanoTime() - tq) / 1e9;
      final long heap = U.heapAfterGcMb();
      long diskBytes = 0, indexBytes = 0; String files = "{}";
      final Path art = s.jv() ? work : work.resolve(s.name);
      if (s.jv()) { try (var st = Files.list(work)) { diskBytes = st.filter(p -> p.getFileName().toString().startsWith(s.name + ".")).mapToLong(p -> { try { return Files.size(p); } catch (final java.io.IOException e) { return 0; } }).sum(); indexBytes = diskBytes; } }
      else { final SortedMap<String, Long> by = U.dirBytesByExt(art); diskBytes = by.values().stream().mapToLong(Long::longValue).sum(); files = U.byExtJson(by); indexBytes = s.lu() ? U.luIndexBytes(by) : U.indexBytes(by); }
      U.result("layer", 1, "phase", "mem", "arm", s.name, "dataset", d.dataset, "n", d.n, "dim", d.dim, "heap_used_gc_mb", heap, "rss_mb", U.rssMb(), "rss_anon_mb", U.anonMb(), "rss_file_mb", U.fileMb(), "vmhwm_mb", U.hwmMb(), "direct_mb", U.directMb(),
          "nonheap_mb", U.nonHeapMb(), "disk_bytes", diskBytes, "index_bytes", indexBytes, "files", files, "fp32_bytes", (long) d.n * d.dim * 4, "pass_s_info", U.r(passS, 3));
      double bestSearchS = Double.MAX_VALUE; Object[] best = null; double bestRecallRow = -1; Object[] bestRecall = null;
      for (final int ef : efs) for (final int rr : rrs) {
        long hit = 0, hit1 = 0, searchNs = 0;
        for (int qi = 0; qi < d.nq; qi++) {
          final long t = System.nanoTime(); final Object raw = sr.search(qi, ef, rr); searchNs += System.nanoTime() - t;
          final int[] ids = sr.ids(raw); final Set<Integer> want = new HashSet<>(); for (int j = 0; j < 10; j++) want.add(d.gt[qi][j]);
          for (int j = 0; j < Math.min(10, ids.length); j++) if (want.contains(ids[j])) hit++;
          if (ids.length > 0 && ids[0] == d.gt[qi][0]) hit1++;
        }
        final double r10 = hit / (10.0 * d.nq), r1 = hit1 / (double) d.nq, searchS = searchNs / 1e9;
        U.result("layer", 1, "phase", "query", "arm", s.name, "dataset", d.dataset, "n", d.n, "dim", d.dim, "ef", ef, "rerank", rr, "recall10", U.r(r10, 5), "recall1", U.r(r1, 5), "nq", d.nq, "search_s_info", U.r(searchS, 3));
        if (r10 >= target && searchS < bestSearchS) { bestSearchS = searchS; best = new Object[] {ef, rr, r10, r1}; }
        if (r10 > bestRecallRow) { bestRecallRow = r10; bestRecall = new Object[] {ef, rr, r10, r1}; }
      }
      final boolean reached = best != null; final Object[] pick = reached ? best : bestRecall;
      U.result("layer", 1, "phase", "calib", "arm", s.name, "dataset", d.dataset, "n", d.n, "dim", d.dim, "target", target, "reached", reached, "ef", pick[0], "rerank", pick[1], "recall10", U.r((Double) pick[2], 5), "recall1", U.r((Double) pick[3], 5),
          "rule", "cheapest tested (ef,rerank) by single-thread search time of the sweep pass with recall@10 >= target; not reached: the highest recall tested");
      U.result("layer", 1, "phase", "peak", "arm", s.name, "dataset", d.dataset, "n", d.n, "peak_heap_mb", peak.heapMb(), "peak_rss_mb", peak.rssMb(), "peak_rss_anon_mb", peak.anonMb());
      System.out.println("QUERY ok");
    }
  }
}
