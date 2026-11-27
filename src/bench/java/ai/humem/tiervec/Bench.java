package ai.humem.tiervec;

import java.io.IOException;
import java.io.PrintWriter;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.LockSupport;
import java.util.zip.CRC32;
import java.util.function.LongPredicate;
import java.util.stream.Stream;

/**
 * Benchmark harness for {@link TieredIndex} (docs/BENCHMARKS.md, issue #12). It is plain Java, no JUnit, and it lives in the package of the
 * library like {@code NativeSmoke}, because it reads the kernel switch ({@code Kernels.VECTOR}), which is package-private. Maven does not
 * compile it: the CI job {@code bench smoke} and the run rules in docs/BENCHMARKS.md compile it with the library sources:
 * {@code javac -d target/bench/classes --add-modules jdk.incubator.vector $(find src/main/java src/bench/java -name '*.java')}.
 *
 * <p>One run writes one result file (and the same lines to standard output), in tagged lines of {@code key=value} pairs. The file starts with the
 * hardware, the JDK, the JVM flags, the Vector API state, the commit of the repository (from {@code git rev-parse HEAD} at run time) and the seed.
 * Then the phases, each one a block of lines:
 * <ol>
 * <li>DATA: the vectors are generated from the seed (a clustered Gaussian mixture, the distribution of {@code GraphT.Gen} of the tests, with
 * {@code StrictMath} so the vectors are bit for bit the same on every JVM), and the exact answers of the query set are computed by an exact scan of
 * the harness, with its own scalar dot product (not a kernel of the library): unfiltered, and for each allowed share.</li>
 * <li>BUILD: the vectors go into a {@link TieredIndex} through {@code add} (or {@code rebuildFrom}), and the run waits until no seal or merge is due.</li>
 * <li>SEARCH: single-thread, warm latency (p50, p99, p99.9) and recall@k against the exact scan, unfiltered and filtered, for each allowed share and each
 * {@code filterScanBelow} value (0 is off, 1 scans first at every share).</li>
 * <li>CONC: writers (adds), one deleter and searchers run for a fixed time; the file reports adds, deletes and searches per second, the latency of the
 * searches per window, every error, and then checks the end state: the live count, the order and uniqueness of every answer, that no answer holds an id whose
 * delete had returned before the search began, and recall against an exact scan of the live set.</li>
 * </ol>
 *
 * <p>The filter is a hash of the id (uniform, independent of the vector): an id is allowed when {@code unit(id) < share}. The allowed sets of the
 * shares are nested. A filter that is correlated with the vectors is not measured.
 *
 * <p>Two options came with issue #108. {@code --data DIR} replaces the generated vectors, queries and exact answers by a real embedding set that
 * {@code scripts/prep_real_dataset.py} prepared (class {@link RealData}). {@code --engine jvector} builds and searches the same data with JVector
 * (class {@code JVectorArm} in {@code src/bench/jvector}, compiled apart because it needs the JVector jars) at the same {@code M} and
 * {@code efConstruction}, and prints the same BUILD, MEMORY and SEARCH lines (filtered cells only when --shares is given). Neither option changes a run that does not use it.
 *
 * <p>Four options came with issue #118 and the comparison of issue #121. {@code --quant} builds the tier with SQ8 codes ({@code sq8}) or the JVector
 * arm on int8 vectors ({@code int8}). {@code --targets} adds a MATCHED block: for every recall target and search mode it finds the smallest beam width
 * of {@code --ef-grid} that reaches the target, times that width and the one below it, and interpolates the latency at the target. {@code --disk DIR}
 * writes the index into an empty directory and prints the size of the files. {@code --props N} (the tier) checks that a write is found by the next
 * search, that an id reused after a delete never answers with its old vector, and prints the counts.
 *
 * <p>Options are {@code --key value} pairs; see {@link #defaults()}. The exit code is 1 when a check of integrity fails (an exception, a lost or phantom
 * vector, an answer out of order or with a repeated id, a deleted id returned) or when a recall floor that was set (it is 0 by default) is not met, and 2 for a wrong option.
 */
public final class Bench {
    private Bench() {}

    static final long QUERY_BASE = 1L << 50;
    static final long ADD_BASE = 1L << 40;
    static final long ADD_STRIDE = 1L << 32;

    // ------------------------------------------------------------------------------------------------ options

    static Map<String, String> defaults() {
        Map<String, String> d = new LinkedHashMap<>();
        d.put("n", "1000000");              // vectors in the build
        d.put("dim", "768");
        d.put("k", "10");
        d.put("ef", "100");                 // beam width of a segment search (the default of the library)
        d.put("ef-sweep", "20,50,200,400"); // extra unfiltered cells at these beam widths (a recall curve); empty is none
        d.put("efc", "200");                // efConstruction
        d.put("m", "16");                   // graph degree
        d.put("queries", "200");            // distinct queries
        d.put("reps", "5");                 // timed passes over the queries per cell, after one warm-up pass that also measures recall
        d.put("seed", "42");
        d.put("clusters", "100");
        d.put("sigma", "0.4");
        d.put("shares", "0.5,0.1,0.01,0.001"); // allowed shares of the filtered cells
        d.put("scan-below", "0,1");         // filterScanBelow values of the filtered cells; 0 is off, 1 scans first at every share
        d.put("tail-limit", "2000");
        d.put("small-limit", "1000");
        d.put("max-segments", "8");
        d.put("merge-fan", "4");
        d.put("build-threads", "0");        // threads of one seal or merge build; 0 is the default of TieredIndex.Config
        d.put("load", "add");               // add: n calls of add, then wait for the merges; rebuild: one rebuildFrom
        d.put("settle", "none");            // none, or merge: mergeNow() after the build, before the searches
        d.put("build-timeout-s", "14400");
        d.put("threads", "0");              // threads for the data generation and the exact scans; 0 is availableProcessors
        d.put("conc-seconds", "600");       // 0 skips the concurrent phase
        d.put("writers", "2");
        d.put("searchers", "2");
        d.put("write-rate", "500");         // adds per second over all writers; 0 is unthrottled
        d.put("delete-rate", "100");        // deletes per second (one deleter thread, ids 0, 1, 2, ...); 0 is none
        d.put("conc-window-s", "30");
        d.put("final-queries", "100");      // queries of the recall check after the concurrent phase
        d.put("min-recall", "0");           // floor of recall@k, unfiltered and filtered with the walk; 0 is no floor
        d.put("min-scan-recall", "0");      // floor of recall@k of the cells that scan first; 0 is no floor
        d.put("host-label", "");            // label of the machine in the file; empty is the host name
        d.put("repo", ".");                 // directory for git rev-parse
        d.put("out", "");                   // result file; empty is standard output only
        // the options below came with the real data set and the JVector arm (issue #108); the PARAMS line leaves them out while they have their defaults
        d.put("data", "");                  // a prepared real data set (a directory, see docs/BENCHMARKS.md); empty is the generated data
        d.put("engine", "tiervec");         // tiervec, or jvector (the comparison arm, src/bench/jvector)
        d.put("jvector-alpha", "1.2");      // JVector: pruning alpha (the value of its tutorial; the tier uses 1.0)
        d.put("jvector-overflow", "1.2");   // JVector: neighbor overflow while building (the value of its tutorial)
        // the options below came with the int8 arm and the matched recall (issues #118 and #121); the same rule for the PARAMS line
        d.put("quant", "none");             // none (float32), sq8 (the tier: SQ8 codes in the segment graphs) or int8 (JVector: int8 vectors)
        d.put("targets", "");               // recall targets of the MATCHED block, such as 0.95,0.99; empty is none
        d.put("ef-grid", "10,12,14,16,18,20,23,26,30,35,40,46,53,60,70,80,92,106,122,140,160,185,212,245,280,320,370,425,490,560,640,800,1000"); // beam widths of the MATCHED block
        d.put("disk", "");                  // an empty or new directory: write the index there and print the file sizes; empty is none
        d.put("props", "0");                // the tier only: this many new ids, and this many reused ids, in the checks of the writes; 0 is none
        return d;
    }

    /** the options of issues #108, #118 and #121 */
    static final List<String> NEW_KEYS = List.of("data", "engine", "jvector-alpha", "jvector-overflow", "quant", "targets", "ef-grid", "disk", "props");

    /** true when a run uses one of those options: then the PARAMS line lists them and the MEMORY lines of the index are printed */
    static boolean extended(Opts o) {
        Map<String, String> d = defaults();
        return NEW_KEYS.stream().anyMatch(key -> !o.s(key).equals(d.get(key)));
    }

    /** the beam widths of the MATCHED block: ascending, each once, none below k */
    static int[] efGrid(Opts o) {
        int k = o.i("k");
        return Arrays.stream(o.ds("ef-grid")).mapToInt(v -> (int) v).filter(v -> v >= k).sorted().distinct().toArray();
    }

    static final class Opts {
        final Map<String, String> m = new TreeMap<>();
        /** the keys that the command line set (a key that has its default value is not the same as a key that was given) */
        final Set<String> given = new HashSet<>();

        Opts(String[] a) {
            m.putAll(defaults());
            for (int i = 0; i < a.length; i += 2) {
                if (!a[i].startsWith("--") || !m.containsKey(a[i].substring(2)) || i + 1 >= a.length)
                    throw new IllegalArgumentException("bad option " + a[i] + " (options are --key value pairs: " + defaults().keySet() + ")");
                m.put(a[i].substring(2), a[i + 1]);
                given.add(a[i].substring(2));
            }
        }

        String s(String k) { return m.get(k); }
        int i(String k) { return Integer.parseInt(m.get(k)); }
        long l(String k) { return Long.parseLong(m.get(k)); }
        double d(String k) { return Double.parseDouble(m.get(k)); }

        double[] ds(String k) {
            String v = m.get(k).trim();
            if (v.isEmpty()) return new double[0];
            String[] p = v.split(",");
            double[] r = new double[p.length];
            for (int j = 0; j < p.length; j++) r[j] = Double.parseDouble(p[j].trim());
            return r;
        }
    }

    // ------------------------------------------------------------------------------------------------ output

    static final class Out {
        private final PrintWriter file;

        Out(String path) throws Exception {
            if (path.isEmpty()) {
                file = null;
            } else {
                Path p = Path.of(path);
                if (p.getParent() != null) Files.createDirectories(p.getParent());
                file = new PrintWriter(Files.newBufferedWriter(p, StandardCharsets.UTF_8));
            }
        }

        synchronized void line(String s) {
            System.out.println(s);
            if (file != null) {
                file.println(s);
                file.flush();
            }
        }

        void close() {
            if (file != null) file.close();
        }
    }

    static String f(String fmt, Object... a) { return String.format(Locale.ROOT, fmt, a); }

    // ------------------------------------------------------------------------------------------------ the machine

    static List<String> readLines(String path) {
        try {
            return Files.readAllLines(Path.of(path), StandardCharsets.ISO_8859_1);
        } catch (Exception e) {
            return List.of();
        }
    }

    static String sh(String... cmd) {
        try {
            ProcessBuilder pb = new ProcessBuilder(cmd);
            pb.redirectError(ProcessBuilder.Redirect.DISCARD);
            Process pr = pb.start();
            byte[] out = pr.getInputStream().readAllBytes();
            if (!pr.waitFor(10, TimeUnit.SECONDS)) {
                pr.destroyForcibly();
                return null;
            }
            return pr.exitValue() == 0 ? new String(out, StandardCharsets.UTF_8).trim() : null;
        } catch (Exception e) {
            return null;
        }
    }

    static long statusKb(String key) {
        for (String l : readLines("/proc/self/status"))
            if (l.startsWith(key + ":")) return Long.parseLong(l.substring(key.length() + 1).trim().split("\\s+")[0]);
        return -1;
    }

    static long meminfoKb(String key) {
        for (String l : readLines("/proc/meminfo"))
            if (l.startsWith(key + ":")) return Long.parseLong(l.substring(key.length() + 1).trim().split("\\s+")[0]);
        return -1;
    }

    static String firstLine(String path) {
        List<String> l = readLines(path);
        return l.isEmpty() ? "unknown" : l.get(0).trim();
    }

    /** the processor block of /proc/cpuinfo for one logical cpu: model name and flags (a hybrid CPU has different cores, so the first allowed one is named) */
    static String[] cpuinfoOf(int cpu) {
        String model = "unknown", flags = "unknown";
        boolean in = false;
        for (String l : readLines("/proc/cpuinfo")) {
            int c = l.indexOf(':');
            if (c < 0) continue;
            String key = l.substring(0, c).trim(), val = l.substring(c + 1).trim();
            if (key.equals("processor")) in = Integer.parseInt(val) == cpu;
            else if (in && key.equals("model name")) model = val;
            else if (in && (key.equals("flags") || key.equals("Features"))) flags = val;
        }
        return new String[] { model, flags };
    }

    static String hostLabel(Opts o) {
        String l = o.s("host-label");
        return l.isEmpty() ? firstLine("/proc/sys/kernel/hostname") : l;
    }

    static void header(Out out, Opts o, Gen gen, RealData real) {
        out.line("BENCH-RESULT format=1 harness=ai.humem.tiervec.Bench");
        out.line("RUN started=" + Instant.now() + " host=" + hostLabel(o));
        String sha = sh("git", "-C", o.s("repo"), "rev-parse", "HEAD");
        String dirty = sh("git", "-C", o.s("repo"), "status", "--porcelain", "--untracked-files=no");
        out.line("COMMIT sha=" + (sha == null ? "unknown" : sha) + " modifiedFiles=" + (dirty == null ? "unknown" : String.valueOf(dirty.isEmpty() ? 0 : dirty.split("\n").length)));
        if (real == null) {
            out.line("SEED seed=" + o.l("seed") + " data=clustered-gaussian clusters=" + o.i("clusters") + " sigma=" + o.d("sigma")
                    + " filter=hash-uniform queryIdBase=" + QUERY_BASE);
        } else {
            // the vectors, the queries and the exact answers come from the files; the seed is left for the filter of the filtered cells
            out.line("SEED seed=" + o.l("seed") + " data=file filter=hash-uniform queryIdBase=" + QUERY_BASE);
        }
        String allowed = "unknown";
        for (String l : readLines("/proc/self/status")) if (l.startsWith("Cpus_allowed_list:")) allowed = l.substring(18).trim();
        int first = 0;
        try {
            first = Integer.parseInt(allowed.split("[-,]")[0]);
        } catch (Exception e) {
            first = 0;
        }
        String[] ci = cpuinfoOf(first);
        int logical = 0;
        for (String l : readLines("/proc/cpuinfo")) if (l.startsWith("processor")) logical++;
        out.line("CPU model=\"" + ci[0] + "\" logicalCpus=" + logical + " cpusAllowedList=" + allowed + " availableProcessors=" + Runtime.getRuntime().availableProcessors()
                + " flagsOfCpu=" + first);
        out.line("CPUFLAGS " + ci[1]);
        out.line("CPUFREQ governor=" + firstLine("/sys/devices/system/cpu/cpu" + first + "/cpufreq/scaling_governor") + " noTurbo="
                + firstLine("/sys/devices/system/cpu/intel_pstate/no_turbo") + " cpuMaxKHz=" + firstLine("/sys/devices/system/cpu/cpu" + first + "/cpufreq/cpuinfo_max_freq"));
        out.line("RAM totalKiB=" + meminfoKb("MemTotal") + " availableKiB=" + meminfoKb("MemAvailable"));
        out.line("OS name=" + System.getProperty("os.name") + " version=" + System.getProperty("os.version") + " arch=" + System.getProperty("os.arch"));
        out.line("JDK version=" + System.getProperty("java.version") + " vendor=" + System.getProperty("java.vendor") + " runtime=" + System.getProperty("java.runtime.version")
                + " vm=\"" + System.getProperty("java.vm.name") + "\" vmVersion=" + System.getProperty("java.vm.version"));
        StringBuilder gcs = new StringBuilder();
        for (GarbageCollectorMXBean b : ManagementFactory.getGarbageCollectorMXBeans()) gcs.append(gcs.length() == 0 ? "" : ",").append(b.getName());
        out.line("JVM maxHeapMiB=" + Runtime.getRuntime().maxMemory() / (1 << 20) + " collectors=" + gcs + " args=" + ManagementFactory.getRuntimeMXBean().getInputArguments());
        out.line("VECTORAPI module=" + ModuleLayer.boot().findModule("jdk.incubator.vector").isPresent() + " kernels=" + (Kernels.VECTOR ? "vector" : "scalar")
                + " lanes=" + Kern.lanes() + " scalarProperty=" + System.getProperty("ai.humem.tiervec.scalar"));
        // a run with the defaults of the later options (generated data, the tier) prints the PARAMS line it printed before the options existed
        boolean extended = extended(o);
        StringBuilder ps = new StringBuilder();
        for (Map.Entry<String, String> e : o.m.entrySet()) {
            if (!extended && NEW_KEYS.contains(e.getKey())) continue;
            ps.append(ps.length() == 0 ? "" : " ").append(e.getKey()).append('=').append(e.getValue());
        }
        out.line("PARAMS " + ps);
        if (real != null) out.line("DATASET " + real.describe());
    }

    // ------------------------------------------------------------------------------------------------ data

    /** Clustered generator of the tests ({@code GraphT.Gen}): unit centroids, each vector is normalize(centroid + sigma * Gaussian noise), deterministic by (seed, id). */
    static final class Gen {
        final int dim;
        final long seed;
        final float[][] cent;
        final float sigma;

        Gen(int dim, long seed, int clusters, float sigma) {
            this.dim = dim;
            this.seed = seed;
            this.sigma = sigma;
            cent = new float[Math.max(1, clusters)][dim];
            SplittableRandom r = new SplittableRandom(seed ^ 0x5DEECE66DL);
            for (float[] c : cent) {
                for (int i = 0; i < dim; i++) c[i] = (float) gauss(r);
                norm(c);
            }
        }

        float[] vec(long id) {
            SplittableRandom r = new SplittableRandom(seed * 0x9E3779B97F4A7C15L + id * 0xD1B54A32D192ED03L + 17);
            float[] v = new float[dim];
            float[] c = cent[(int) Math.floorMod(id * 0x9E3779B97F4A7C15L >>> 20, (long) cent.length)];
            float scale = sigma / (float) StrictMath.sqrt(dim) * 4f;
            for (int i = 0; i < dim; i++) v[i] = c[i] + scale * (float) gauss(r);
            norm(v);
            return v;
        }

        static double gauss(SplittableRandom r) {
            double u = 1 - r.nextDouble(), v = r.nextDouble();
            return StrictMath.sqrt(-2 * StrictMath.log(u)) * StrictMath.cos(2 * Math.PI * v);
        }

        static void norm(float[] v) {
            double n = 0;
            for (float x : v) n += (double) x * x;
            float inv = (float) (1 / StrictMath.sqrt(n));
            for (int i = 0; i < v.length; i++) v[i] *= inv;
        }
    }

    static long mix(long z) {
        z += 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /** a number in [0, 1) that depends on the id and the seed only */
    static double unit(long id, long seed) {
        return (mix(id ^ (seed * 0x2545F4914F6CDD1DL)) >>> 11) * 0x1.0p-53;
    }

    static LongPredicate allow(double share, long seed) { return id -> unit(id, seed) < share; }

    interface RangeTask {
        void run(int t, int lo, int hi) throws Exception;
    }

    /** runs the task on contiguous ranges of [0, n), one range per thread; rethrows the first failure */
    static void parallel(int threads, int n, RangeTask task) throws Exception {
        int t = Math.max(1, Math.min(threads, n));
        Thread[] ths = new Thread[t];
        ConcurrentLinkedQueue<Throwable> err = new ConcurrentLinkedQueue<>();
        for (int i = 0; i < t; i++) {
            final int ti = i;
            final int lo = (int) ((long) n * i / t), hi = (int) ((long) n * (i + 1) / t);
            ths[i] = new Thread(() -> {
                try {
                    task.run(ti, lo, hi);
                } catch (Throwable e) {
                    err.add(e);
                }
            }, "bench-par-" + i);
            ths[i].start();
        }
        for (Thread th : ths) th.join();
        Throwable e = err.peek();
        if (e != null) throw new RuntimeException("a worker failed: " + e, e);
    }

    // ------------------------------------------------------------------------------------------------ exact scan

    static float dot(float[] a, float[] b, int dim) {
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int i = 0;
        for (; i + 4 <= dim; i += 4) {
            s0 += a[i] * b[i];
            s1 += a[i + 1] * b[i + 1];
            s2 += a[i + 2] * b[i + 2];
            s3 += a[i + 3] * b[i + 3];
        }
        for (; i < dim; i++) s0 += a[i] * b[i];
        return (s0 + s1) + (s2 + s3);
    }

    /** the k best (score, id) pairs, best first */
    static final class Top {
        final float[] s;
        final long[] id;
        final int k;
        int n;

        Top(int k) {
            this.k = k;
            s = new float[k];
            id = new long[k];
        }

        void offer(float score, long i) {
            if (n == k) {
                if (score <= s[k - 1]) return;
                n--;
            }
            int p = n;
            while (p > 0 && s[p - 1] < score) {
                s[p] = s[p - 1];
                id[p] = id[p - 1];
                p--;
            }
            s[p] = score;
            id[p] = i;
            n++;
        }
    }

    /**
     * Exact top-k of every query over vecs[0, count), by the dot product (the vectors are unit length, so it is the cosine). Group 0 is unfiltered, group g
     * the ids with unit(id) below shares[g - 1]. The id of vecs[i] is idOf[i], or i when idOf is null. Result: [group][query] ids, best first.
     */
    static long[][][] exact(float[][] vecs, long[] idOf, int count, float[][] qs, int k, double[] shares, long fseed, int threads) throws Exception {
        final int nq = qs.length, groups = 1 + shares.length, dim = qs[0].length;
        if (shares.length > 30) throw new IllegalArgumentException("at most 30 shares");
        final int tn = Math.max(1, threads);
        final Top[][][] part = new Top[tn][groups][nq];
        for (int t = 0; t < tn; t++) for (int g = 0; g < groups; g++) for (int q = 0; q < nq; q++) part[t][g][q] = new Top(k);
        final int block = 64;
        parallel(tn, count, (t, lo, hi) -> {
            int[] mask = new int[block];
            for (int b = lo; b < hi; b += block) {
                int e = Math.min(hi, b + block);
                for (int i = b; i < e; i++) {
                    long id = idOf == null ? i : idOf[i];
                    int mk = 0;
                    for (int g = 0; g < shares.length; g++) if (unit(id, fseed) < shares[g]) mk |= 1 << g;
                    mask[i - b] = mk;
                }
                for (int q = 0; q < nq; q++) {
                    float[] qv = qs[q];
                    Top t0 = part[t][0][q];
                    for (int i = b; i < e; i++) {
                        float sc = dot(qv, vecs[i], dim);
                        long id = idOf == null ? i : idOf[i];
                        t0.offer(sc, id);
                        int mk = mask[i - b];
                        while (mk != 0) {
                            int g = Integer.numberOfTrailingZeros(mk);
                            mk &= mk - 1;
                            part[t][1 + g][q].offer(sc, id);
                        }
                    }
                }
            }
        });
        long[][][] res = new long[groups][nq][];
        for (int g = 0; g < groups; g++) {
            for (int q = 0; q < nq; q++) {
                Top all = new Top(k);
                for (int t = 0; t < tn; t++) {
                    Top p = part[t][g][q];
                    for (int j = 0; j < p.n; j++) all.offer(p.s[j], p.id[j]);
                }
                res[g][q] = Arrays.copyOf(all.id, all.n);
            }
        }
        return res;
    }

    static double recall(Hit[] got, long[] truth) {
        if (truth.length == 0) return 1;
        int c = 0;
        for (long t : truth) for (Hit h : got) if (h.id() == t) { c++; break; }
        return (double) c / truth.length;
    }

    /** null when the answer is well formed: at most k hits, scores not increasing, no id twice */
    static String malformed(Hit[] h, int k) {
        if (h.length > k) return "more than k hits";
        for (int j = 0; j < h.length; j++) {
            if (j > 0 && h[j].score() > h[j - 1].score()) return "scores out of order";
            for (int i = 0; i < j; i++) if (h[i].id() == h[j].id()) return "an id twice";
        }
        return null;
    }

    static long pct(long[] sorted, double p) {
        return sorted.length == 0 ? 0 : sorted[(int) Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
    }

    // ------------------------------------------------------------------------------------------------ small helpers

    static final class LongList {
        long[] a = new long[1024];
        int n;

        void add(long v) {
            if (n == a.length) a = Arrays.copyOf(a, n * 2);
            a[n++] = v;
        }
    }

    record Gc(long count, long millis) {
        static Gc take() {
            long c = 0, ms = 0;
            for (GarbageCollectorMXBean b : ManagementFactory.getGarbageCollectorMXBeans()) {
                if (b.getCollectionCount() > 0) c += b.getCollectionCount();
                if (b.getCollectionTime() > 0) ms += b.getCollectionTime();
            }
            return new Gc(c, ms);
        }

        String since(Gc a) { return "gcCount=" + (count - a.count) + " gcMillis=" + (millis - a.millis); }
    }

    static String stats(TieredIndex.Stats s) {
        return f("seals=%d merges=%d tail=%d segments=%d segmentSizes=%s bytesHeldMB=%.1f sealMs=%d mergeMs=%d maxSealMs=%d maxMergeMs=%d backpressureWaits=%d"
                        + " failures=%d live=%d floorEntries=%d totalWriteAmp=%.2f", s.seals(), s.merges(), s.tailSize(), s.segments(), Arrays.toString(s.segmentSizes()),
                s.bytesHeld() / 1e6, s.sealMillis(), s.mergeMillis(), s.maxSealMillis(), s.maxMergeMillis(), s.backpressureWaits(), s.failures(), s.live(), s.floorEntries(),
                s.totalWriteAmp());
    }

    static void memory(Out out, String tag) {
        System.gc();
        System.gc();
        Runtime rt = Runtime.getRuntime();
        out.line(f("MEMORY %s heapUsedMB=%.0f heapCommittedMB=%.0f rssMB=%.0f rssPeakMB=%.0f", tag, (rt.totalMemory() - rt.freeMemory()) / 1e6, rt.totalMemory() / 1e6,
                statusKb("VmRSS") / 1024.0, statusKb("VmHWM") / 1024.0));
    }

    // ------------------------------------------------------------------------------------------------ the engine under test, and the real data

    /**
     * An index under test other than the tier. The one implementation is {@code JVectorArm} in {@code src/bench/jvector}, which is compiled
     * apart (it needs the JVector jars) and is reached by reflection, so that the harness compiles and runs without JVector when
     * {@code --engine tiervec} is used.
     */
    interface Arm {
        /** the search modes, one or more; the harness runs its cells in each of them, and the mode of a search is the index in this list */
        List<String> modes();

        /** the k best hits of the query among the allowed ids (null: all), best first; the score is the dot product (the cosine, the vectors have unit length) */
        Hit[] search(float[] q, int k, int ef, int mode, LongPredicate allow);

        /** {@code key=value} pairs for the JVECTOR line of the result file: the version, the provider, the parameters */
        String describe();

        /** {@code key=value} pairs for the MEMORY-INDEX line: the bytes that a search needs, by part */
        String memoryBytes();

        /** writes the index into the directory, which is empty */
        void writeDisk(Path dir) throws IOException;

        void close() throws Exception;
    }

    /** what {@link #prepare} found: the real data set (null for generated data) and the constructor of the JVector arm (null for the tier) */
    static final class Plan {
        RealData real;
        Constructor<?> jvector;
    }

    /**
     * Checks the options that depend on each other and on the files, and rewrites {@code n}, {@code dim} and the other options that the data
     * file or the engine decide. Throws IllegalArgumentException for a wrong combination (exit code 2), before any data is loaded.
     */
    static Plan prepare(Opts o) {
        Plan plan = new Plan();
        String engine = o.s("engine");
        if (!engine.equals("tiervec") && !engine.equals("jvector")) throw new IllegalArgumentException("--engine is tiervec or jvector, not " + engine);
        if (!o.s("data").isEmpty()) {
            try {
                plan.real = RealData.open(Path.of(o.s("data")));
            } catch (IOException e) {
                throw new IllegalArgumentException("cannot read --data " + o.s("data") + ": " + e, e);
            }
            RealData real = plan.real;
            if (real.metric() != Metric.COSINE)
                throw new IllegalArgumentException("--data " + o.s("data") + " is an l2 set; this harness scores by the cosine (PacedWrites reads l2 sets)");
            if (o.given.contains("n") && o.i("n") != real.count)
                throw new IllegalArgumentException("--n " + o.i("n") + " does not match the data file, which holds " + real.count
                        + " vectors (with --data the size comes from the file; prepare a smaller file with --limit)");
            if (o.given.contains("dim") && o.i("dim") != real.dim)
                throw new IllegalArgumentException("--dim " + o.i("dim") + " does not match the data file, which holds vectors of dimension " + real.dim);
            if (o.i("k") > real.truthK)
                throw new IllegalArgumentException("--k " + o.i("k") + " is more than the " + real.truthK + " exact answers per query that the data file holds");
            if (o.given.contains("queries") && o.i("queries") > real.queries)
                throw new IllegalArgumentException("--queries " + o.i("queries") + " is more than the " + real.queries + " queries that the data file holds");
            o.m.put("n", String.valueOf(real.count));
            o.m.put("dim", String.valueOf(real.dim));
            if (!o.given.contains("queries")) o.m.put("queries", String.valueOf(Math.min(o.i("queries"), real.queries)));
        }
        if (plan.real != null || engine.equals("jvector")) {
            // the writers of the concurrent phase draw their vectors from the generator, and the JVector arm has no concurrent writes
            if (o.given.contains("conc-seconds") && o.i("conc-seconds") > 0)
                throw new IllegalArgumentException("--conc-seconds is not available with --data or --engine jvector: the concurrent phase adds generated vectors to the tier");
            o.m.put("conc-seconds", "0");
        }
        String quant = o.s("quant");
        if (!quant.equals("none") && !quant.equals(engine.equals("jvector") ? "int8" : "sq8"))
            throw new IllegalArgumentException("--quant is none or sq8 with --engine tiervec, and none or int8 with --engine jvector, not " + quant);
        for (double t : o.ds("targets")) if (!(t > 0 && t <= 1)) throw new IllegalArgumentException("--targets are recall values above 0 and at most 1, not " + t);
        if (o.ds("targets").length > 0 && efGrid(o).length == 0) throw new IllegalArgumentException("--ef-grid holds no beam width of at least --k");
        if (o.i("props") > 0) {
            if (engine.equals("jvector")) throw new IllegalArgumentException("--props checks the writes of the tier, it is not available with --engine jvector");
            if (o.i("conc-seconds") > 0) throw new IllegalArgumentException("--props needs --conc-seconds 0: it expects the ids of the build");
        }
        if (engine.equals("jvector")) {
            // the filtered cells of the JVector arm run when --shares is given (one search per share, there is no plan to choose)
            if (!o.given.contains("shares")) o.m.put("shares", "");
            if (o.given.contains("scan-below")) throw new IllegalArgumentException("--scan-below is a plan of the tier: the JVector arm has one filtered search");
            if (o.given.contains("load")) throw new IllegalArgumentException("--load is an option of the tier: JVector builds the graph in one parallel pass");
            try {
                Class<?> arm = Class.forName("ai.humem.tiervec.JVectorArm");
                plan.jvector = arm.getDeclaredConstructor(float[][].class, int.class, int.class, int.class, float.class, float.class, int.class, boolean.class);
                plan.jvector.setAccessible(true);
            } catch (ClassNotFoundException | LinkageError | NoSuchMethodException e) {
                throw new IllegalArgumentException("--engine jvector needs the JVector arm and the JVector jars on the class path (docs/BENCHMARKS.md, \"The JVector arm\"): " + e, e);
            }
        }
        return plan;
    }

    // ------------------------------------------------------------------------------------------------ the run

    static final class Run {
        final Opts o;
        final Out out;
        final List<String> failures = new ArrayList<>();
        final int n, dim, k, ef, nq, reps, threads;
        final long seed;
        final double[] shares, scanBelows;
        final double minRecall, minScanRecall;
        Gen gen;
        float[][] data;
        float[][] qs;
        long[][][] truth;
        TieredIndex tier;
        long filterSeed;
        final RealData real;      // null: the generated data
        final Constructor<?> jvectorCtor; // null: the engine is the tier
        Arm arm;                  // the JVector arm once it is built, else null
        int mode;                 // the search mode of the JVector arm that the cells run in

        Run(Opts o, Out out, Plan plan) {
            this.o = o;
            this.out = out;
            this.real = plan.real;
            this.jvectorCtor = plan.jvector;
            n = o.i("n");
            dim = o.i("dim");
            k = o.i("k");
            ef = o.i("ef");
            nq = o.i("queries");
            reps = o.i("reps");
            seed = o.l("seed");
            shares = o.ds("shares");
            scanBelows = o.ds("scan-below");
            minRecall = o.d("min-recall");
            minScanRecall = o.d("min-scan-recall");
            threads = o.i("threads") > 0 ? o.i("threads") : Runtime.getRuntime().availableProcessors();
            filterSeed = seed ^ 0x7A11E5L;
        }

        void fail(String why) {
            failures.add(why);
            out.line("FAIL " + why);
        }

        void phase(String name) { out.line("PHASE " + name + " at=" + Instant.now()); }

        // ---- DATA

        /** checksum of the first 1000 vectors and the first query, as the DATA line has it (the same bytes on every JVM for the same data) */
        long dataChecksum() {
            CRC32 crc = new CRC32();
            for (int i = 0; i < Math.min(n, 1000); i++) for (float x : data[i]) for (int sh = 0; sh < 32; sh += 8) crc.update(Float.floatToIntBits(x) >>> sh);
            for (float x : qs[0]) for (int sh = 0; sh < 32; sh += 8) crc.update(Float.floatToIntBits(x) >>> sh);
            return crc.getValue();
        }

        void truthLine() {
            StringBuilder tl = new StringBuilder();
            for (int g = 0; g < truth.length; g++) {
                int min = Integer.MAX_VALUE;
                for (long[] t : truth[g]) min = Math.min(min, t.length);
                tl.append(g == 0 ? "unfiltered" : f("share%s", o.ds("shares")[g - 1])).append(":minTruthSize=").append(min).append(' ');
            }
            out.line("TRUTH " + tl.toString().trim());
        }

        /** the number of ids that the two answers share, as a share of the second one */
        static double overlap(long[] a, long[] b) {
            if (b.length == 0) return 1;
            int c = 0;
            for (long x : b) for (long y : a) if (x == y) { c++; break; }
            return (double) c / b.length;
        }

        /**
         * DATA from the files of a prepared data set: the vectors, the queries and the exact answers of the unfiltered search (computed by
         * scripts/prep_real_dataset.py). The filtered cells need the exact answers of each share, and those the harness computes itself, as for
         * generated data; when it does, it also checks the file's answers against its own scan of the same queries.
         */
        void dataReal() throws Exception {
            phase("data");
            Gc g0 = Gc.take();
            long t0 = System.nanoTime();
            data = real.readVectors();
            qs = real.readQueries(nq);
            long[][] fileTruth = real.readTruth(nq, k);
            double loadS = (System.nanoTime() - t0) / 1e9;
            double exactS = 0;
            if (shares.length > 0) {
                t0 = System.nanoTime();
                truth = exact(data, null, n, qs, k, shares, filterSeed, threads);
                exactS = (System.nanoTime() - t0) / 1e9;
                double ov = 0;
                for (int q = 0; q < nq; q++) ov += overlap(truth[0][q], fileTruth[q]);
                ov /= Math.max(1, nq);
                out.line(f("TRUTH-CHECK fileVsExactScanOverlap=%.4f queries=%d", ov, nq));
                if (ov < 0.98) fail(f("the exact answers in the data file share only %.4f of their ids with the exact scan of the harness over the same vectors", ov));
                truth[0] = fileTruth;
            } else {
                truth = new long[][][] { fileTruth };
            }
            out.line(f("DATA source=file n=%d dim=%d queries=%d checksumFirst1000AndQuery0=%08x loadSeconds=%.1f exactScanSeconds=%.1f exactScanThreads=%d %s", n, dim, nq,
                    dataChecksum(), loadS, exactS, threads, Gc.take().since(g0)));
            truthLine();
        }

        void data() throws Exception {
            if (real != null) {
                dataReal();
                return;
            }
            phase("data");
            gen = new Gen(dim, seed, o.i("clusters"), (float) o.d("sigma"));
            Gc g0 = Gc.take();
            long t0 = System.nanoTime();
            data = new float[n][];
            parallel(threads, n, (t, lo, hi) -> {
                for (int i = lo; i < hi; i++) data[i] = gen.vec(i);
            });
            double genS = (System.nanoTime() - t0) / 1e9;
            qs = new float[nq][];
            for (int i = 0; i < nq; i++) qs[i] = gen.vec(QUERY_BASE + i);
            t0 = System.nanoTime();
            truth = exact(data, null, n, qs, k, shares, filterSeed, threads);
            double exactS = (System.nanoTime() - t0) / 1e9;
            out.line(f("DATA n=%d dim=%d queries=%d checksumFirst1000AndQuery0=%08x generateSeconds=%.1f exactScanSeconds=%.1f exactScanThreads=%d %s", n, dim, nq, dataChecksum(), genS, exactS,
                    threads, Gc.take().since(g0)));
            truthLine();
        }

        // ---- BUILD

        /**
         * BUILD of the JVector arm: the same data, the same M and efConstruction as the tier (--m, --efc), one parallel build pass. The line has
         * the keys of the tier's BUILD line so that the two files read side by side; the JVECTOR line says what JVector ran with.
         */
        void buildJVector() throws Exception {
            phase("build");
            Gc g0 = Gc.take();
            long t0 = System.nanoTime();
            try {
                arm = (Arm) jvectorCtor.newInstance(data, dim, o.i("m"), o.i("efc"), (float) o.d("jvector-alpha"), (float) o.d("jvector-overflow"), o.i("build-threads"),
                        o.s("quant").equals("int8"));
            } catch (InvocationTargetException e) {
                Throwable c = e.getCause();
                if (c instanceof Exception ex) throw ex;
                if (c instanceof Error er) throw er;
                throw e;
            }
            double s = (System.nanoTime() - t0) / 1e9;
            out.line(f("BUILD load=jvector addSeconds=%.1f settledSeconds=%.1f insertsPerSecondSettled=%.0f size=%d %s", s, s, n / s, n, Gc.take().since(g0)));
            out.line("JVECTOR " + arm.describe());
            memory(out, "afterBuild");
            out.line("MEMORY-INDEX " + arm.memoryBytes());
        }

        void build() throws Exception {
            if (extended(o)) memory(out, "beforeBuild");
            if (jvectorCtor != null) {
                buildJVector();
                return;
            }
            phase("build");
            GraphIndex.Params gp0 = GraphIndex.Params.defaults().withM(o.i("m")).withEfConstruction(o.i("efc"));
            GraphIndex.Params gp = new GraphIndex.Params(gp0.M(), gp0.efConstruction(), ef, gp0.alpha(), o.s("quant").equals("sq8"), gp0.seed());
            TieredIndex.Config cfg = TieredIndex.Config.defaults().withTail(o.i("tail-limit"), o.i("small-limit")).withMaxSegments(o.i("max-segments"))
                    .withMergeFan(o.i("merge-fan")).withGraph(gp);
            if (o.i("build-threads") > 0) cfg = cfg.withBuildThreads(o.i("build-threads"));
            out.line("CONFIG " + cfg);
            tier = new TieredIndex(dim, Metric.COSINE, cfg);
            Gc g0 = Gc.take();
            long deadline = System.nanoTime() + o.l("build-timeout-s") * 1_000_000_000L;
            long t0 = System.nanoTime();
            boolean timedOut = false;
            double addS, idleS;
            if (o.s("load").equals("rebuild")) {
                List<TieredIndex.IdVector> src = new ArrayList<>(n);
                for (int i = 0; i < n; i++) src.add(new TieredIndex.IdVector(i, data[i]));
                t0 = System.nanoTime();
                tier.rebuildFrom(src);
                addS = (System.nanoTime() - t0) / 1e9;
                idleS = addS;
            } else {
                long lastLine = t0;
                for (int i = 0; i < n; i++) {
                    tier.add(i, data[i]);
                    if ((i + 1) % 50_000 == 0 && System.nanoTime() - lastLine > 60_000_000_000L) {
                        lastLine = System.nanoTime();
                        out.line(f("BUILD-T added=%d seconds=%.0f %s", i + 1, (lastLine - t0) / 1e9, stats(tier.stats())));
                    }
                    if ((i & 1023) == 0 && System.nanoTime() > deadline) {
                        timedOut = true;
                        fail("build timeout after " + (i + 1) + " adds");
                        break;
                    }
                }
                addS = (System.nanoTime() - t0) / 1e9;
                long left = Math.max(1, (deadline - System.nanoTime()) / 1_000_000L);
                boolean idle = timedOut || tier.awaitIdle(left);
            if (timedOut) throw new IllegalStateException("the build did not finish within " + o.l("build-timeout-s") + " s, the later phases are skipped");
                if (!idle) fail("build: the tier was not idle within the timeout");
                idleS = (System.nanoTime() - t0) / 1e9;
            }
            out.line(f("BUILD load=%s addSeconds=%.1f settledSeconds=%.1f insertsPerSecondSettled=%.0f size=%d %s", o.s("load"), addS, idleS, n / idleS, tier.size(), Gc.take().since(g0)));
            out.line("BUILD-STATS " + stats(tier.stats()));
            if (tier.size() != (timedOut ? tier.size() : n)) fail("size after build is " + tier.size() + ", expected " + n);
            if (tier.lastError() != null) fail("tier.lastError after build: " + tier.lastError());
            if (o.s("settle").equals("merge")) {
                long a = System.nanoTime();
                tier.mergeNow();
                out.line(f("SETTLE mergeNow seconds=%.1f %s", (System.nanoTime() - a) / 1e9, stats(tier.stats())));
            }
            memory(out, "afterBuild");
            if (extended(o)) out.line("MEMORY-INDEX searchBytes=" + tier.stats().bytesHeld());
        }

        // ---- SEARCH

        void search() {
            phase("search");
            int modes = modeNames().size();
            for (mode = 0; mode < modes; mode++) {
                cell("unfiltered", -1, 0, null, truth[0], ef);
                for (double e : o.ds("ef-sweep")) cell("unfiltered", -1, 0, null, truth[0], (int) e);
                if (arm != null) for (int s = 0; s < shares.length; s++) cell("filtered", shares[s], 0, allow(shares[s], filterSeed), truth[1 + s], ef);
            }
            mode = 0;
            if (arm != null) return; // the JVector arm has no plan of its own for a filter, its filtered cells ran above
            for (int s = 0; s < shares.length; s++) {
                LongPredicate al = allow(shares[s], filterSeed);
                for (double sb : scanBelows) cell("filtered", shares[s], sb, al, truth[1 + s], ef);
            }
            tier.filterScanBelow(0);
        }

        List<String> modeNames() {
            return arm != null ? arm.modes() : List.of(o.s("quant").equals("sq8") ? "sq8" : "fp32");
        }

        /** one search of the engine under test */
        Hit[] one(float[] q, LongPredicate al, int efv) {
            return arm != null ? arm.search(q, k, efv, mode, al) : tier.search(q, k, al, efv);
        }

        void cell(String kind, double share, double scanBelow, LongPredicate al, long[][] tr, int efv) {
            if (tier != null) tier.filterScanBelow(scanBelow);
            boolean scans = al != null && scanBelow >= share && scanBelow > 0;
            long[] lat = new long[reps * nq];
            double rsum = 0, rmin = 1, maxScoreErr = 0;
            int bad = 0;
            long ln = 0;
            Gc g0 = Gc.take();
            for (int pass = 0; pass <= reps; pass++) {
                for (int q = 0; q < nq; q++) {
                    long a = System.nanoTime();
                    Hit[] h = one(qs[q], al, efv);
                    long b = System.nanoTime();
                    if (pass == 0) {
                        double r = recall(h, tr[q]);
                        rsum += r;
                        rmin = Math.min(rmin, r);
                        if (malformed(h, k) != null) bad++;
                        for (Hit x : h) {
                            if (x.id() < 0 || x.id() >= n) {
                                bad++;
                                continue;
                            }
                            maxScoreErr = Math.max(maxScoreErr, Math.abs(x.score() - dot(qs[q], data[(int) x.id()], dim)));
                            if (al != null && !al.test(x.id())) bad++;
                        }
                    } else {
                        lat[(int) ln++] = b - a;
                    }
                }
            }
            long[] so = Arrays.copyOf(lat, (int) ln);
            Arrays.sort(so);
            double sumS = 0;
            for (long v : so) sumS += v / 1e9;
            double rec = rsum / nq;
            out.line(f("SEARCH kind=%s share=%s scanBelow=%s scanFirst=%b k=%d ef=%d recall=%.4f recallMin=%.2f p50us=%d p99us=%d p999us=%d meanUs=%.0f maxUs=%d"
                            + " opsPerSecond=%.1f samples=%d badAnswers=%d maxScoreError=%.2e %s", kind, share < 0 ? "none" : String.valueOf(share), scanBelow, scans, k, efv, rec, rmin,
                    pct(so, .5) / 1000, pct(so, .99) / 1000, pct(so, .999) / 1000, sumS / Math.max(1, so.length) * 1e6, so.length == 0 ? 0 : so[so.length - 1] / 1000,
                    so.length / Math.max(1e-9, sumS), so.length, bad, maxScoreErr, Gc.take().since(g0)) + (modeNames().size() > 1 ? " mode=" + modeNames().get(mode) : ""));
            if (bad > 0) fail("search cell " + kind + " share " + share + " scanBelow " + scanBelow + ": " + bad + " bad answers");
            // the floors are for the configured ef; the cells of the ef sweep are a curve, not a bar
            double floor = scans ? minScanRecall : minRecall;
            if (efv == ef && rec < floor) fail(f("search cell %s share %s scanBelow %s: recall %.4f is below the floor %.4f", kind, share, scanBelow, rec, floor));
        }

        // ---- MATCHED

        /** the recall of one unfiltered pass over the queries at a beam width, with the checks of the answers of a SEARCH cell */
        double recallAt(int efv) {
            double sum = 0;
            int bad = 0;
            for (int q = 0; q < nq; q++) {
                Hit[] h = one(qs[q], null, efv);
                if (malformed(h, k) != null) bad++;
                for (Hit x : h) if (x.id() < 0 || x.id() >= n) bad++;
                sum += recall(h, truth[0][q]);
            }
            if (bad > 0) fail("matched recall at ef " + efv + ": " + bad + " bad answers");
            return sum / nq;
        }

        /** one warm-up pass and the timed passes at one beam width; the result is p50, p99 and the mean in microseconds */
        double[] timed(int efv) {
            for (int q = 0; q < nq; q++) one(qs[q], null, efv);
            long[] lat = new long[Math.max(1, reps) * nq];
            long sum = 0;
            for (int i = 0; i < lat.length; i++) {
                float[] q = qs[i % nq];
                long a = System.nanoTime();
                one(q, null, efv);
                lat[i] = System.nanoTime() - a;
                sum += lat[i];
            }
            Arrays.sort(lat);
            return new double[] { pct(lat, .5) / 1000.0, pct(lat, .99) / 1000.0, sum / 1000.0 / lat.length };
        }

        /**
         * MATCHED: latency at a fixed recall (--targets). The beam widths of the grid are tried from the smallest until the highest target is
         * reached. For each target the first width that reaches it and the one before are timed, and the latency at the target is interpolated
         * linearly in recall between the two. status=floor: the smallest width of the grid is above the target already, and its latency is
         * given as it is. status=unreached: no width of the grid reaches the target.
         */
        void matched() {
            double[] targets = o.ds("targets");
            if (targets.length == 0) return;
            phase("matched");
            if (tier != null) tier.filterScanBelow(0);
            int[] grid = efGrid(o);
            double top = Arrays.stream(targets).max().getAsDouble();
            List<String> names = modeNames();
            for (mode = 0; mode < names.size(); mode++) {
                String name = names.get(mode);
                double[] rec = new double[grid.length];
                int swept = 0;
                while (swept < grid.length) {
                    rec[swept] = recallAt(grid[swept]);
                    out.line(f("MATCH-SWEEP mode=%s ef=%d recall=%.5f", name, grid[swept], rec[swept]));
                    swept++;
                    if (rec[swept - 1] >= top) break;
                }
                int[] reach = new int[targets.length];
                TreeSet<Integer> need = new TreeSet<>();
                for (int t = 0; t < targets.length; t++) {
                    reach[t] = -1;
                    for (int i = 0; i < swept && reach[t] < 0; i++) if (rec[i] >= targets[t]) reach[t] = i;
                    if (reach[t] >= 0) need.add(reach[t]);
                    if (reach[t] > 0) need.add(reach[t] - 1);
                }
                Map<Integer, double[]> at = new TreeMap<>();
                for (int i : need) {
                    Gc g0 = Gc.take();
                    double[] r = timed(grid[i]);
                    at.put(i, r);
                    out.line(f("MATCH-TIMED mode=%s ef=%d recall=%.5f p50us=%.3f p99us=%.3f meanUs=%.3f samples=%d %s", name, grid[i], rec[i], r[0], r[1], r[2], Math.max(1, reps) * nq,
                            Gc.take().since(g0)));
                }
                for (int t = 0; t < targets.length; t++) {
                    int hi = reach[t];
                    if (hi < 0) {
                        double best = 0;
                        for (int i = 0; i < swept; i++) best = Math.max(best, rec[i]);
                        out.line(f("MATCHED mode=%s target=%s status=unreached maxEf=%d maxRecall=%.5f", name, targets[t], grid[swept - 1], best));
                        continue;
                    }
                    int lo = Math.max(0, hi - 1);
                    double w = hi == lo ? 1 : (targets[t] - rec[lo]) / (rec[hi] - rec[lo]);
                    double[] a = at.get(lo), b = at.get(hi);
                    double p50 = a[0] + w * (b[0] - a[0]), p99 = a[1] + w * (b[1] - a[1]), mean = a[2] + w * (b[2] - a[2]);
                    out.line(f("MATCHED mode=%s target=%s status=%s efLo=%d efHi=%d recallLo=%.5f recallHi=%.5f p50us=%.3f p99us=%.3f meanUs=%.3f qps=%.1f", name, targets[t],
                            hi == lo ? "floor" : "ok", grid[lo], grid[hi], rec[lo], rec[hi], p50, p99, mean, 1e6 / mean));
                }
            }
            mode = 0;
        }

        // ---- DISK

        /** writes the index into an empty directory (the tier: sealNow and checkpoint; JVector: one graph file) and prints the file sizes */
        void disk() throws Exception {
            if (o.s("disk").isEmpty()) return;
            phase("disk");
            Path dir = Path.of(o.s("disk"));
            Files.createDirectories(dir);
            try (Stream<Path> s = Files.list(dir)) {
                if (s.findAny().isPresent()) throw new IllegalStateException("--disk " + dir + " is not empty");
            }
            long t0 = System.nanoTime();
            if (arm != null) {
                arm.writeDisk(dir);
            } else {
                tier.sealNow();
                if (!tier.awaitIdle(600_000L)) fail("disk: the tier was not idle");
                tier.checkpoint(dir);
            }
            double seconds = (System.nanoTime() - t0) / 1e9;
            List<Path> files;
            try (Stream<Path> s = Files.list(dir)) {
                files = s.filter(Files::isRegularFile).filter(p -> !p.getFileName().toString().equals(DirectoryLock.FILE)).sorted().toList(); // the empty lock file is not part of the index
            }
            long bytes = 0;
            StringBuilder detail = new StringBuilder();
            for (Path file : files) {
                long size = Files.size(file);
                bytes += size;
                detail.append(detail.length() == 0 ? "" : ",").append(file.getFileName()).append(':').append(size);
            }
            if (bytes == 0) fail("disk: nothing was written");
            out.line(f("DISK seconds=%.1f files=%d bytes=%d detail=%s", seconds, files.size(), bytes, detail));
        }

        // ---- PROPS

        /** a vector near v that is not v: v plus Gaussian noise of length about 0.3, scaled to length 1 */
        static float[] near(float[] v, SplittableRandom r) {
            float[] w = new float[v.length];
            float scale = 0.3f / (float) Math.sqrt(v.length);
            for (int i = 0; i < v.length; i++) w[i] = v[i] + scale * (float) Gen.gauss(r);
            Gen.norm(w);
            return w;
        }

        static boolean holds(Hit[] h, long id) {
            for (Hit x : h) if (x.id() == id) return true;
            return false;
        }

        static boolean first(Hit[] h, long id) { return h.length > 0 && h[0].id() == id; }

        /** 1 when the answer holds the id with a score that is not the one of the vector the id has now (an old copy was returned) */
        int stale(Hit[] h, long id, float[] q, float[] now) {
            for (Hit x : h) if (x.id() == id && Math.abs(x.score() - dot(q, now, dim)) > 1e-3f) return 1;
            return 0;
        }

        /**
         * PROPS (--props N, the tier): N new ids are added, and each is searched for at once; then N ids of the build are deleted and added again
         * with another vector, and each is searched for after the delete and after the add. Then the tail is sealed, the tier settles, and the
         * searches are repeated. The run fails when a write is not in the answer of the next search, when a deleted id is returned, when an id
         * answers with the score of its old vector, or when the live count is wrong. Whether the written vector is the first hit once it is in a
         * segment depends on the graph walk, so that is a count and not a check.
         */
        void props() {
            int writes = o.i("props");
            if (writes <= 0) return;
            phase("props");
            tier.filterScanBelow(0);
            SplittableRandom r = new SplittableRandom(seed ^ 0x9120F5L);
            int reused = Math.min(writes, n);
            float[][] added = new float[writes][], replaced = new float[reused][];
            int newMissed = 0, newNotFirst = 0, deletedReturned = 0, reusedMissed = 0, reusedNotFirst = 0, staleNext = 0;
            TieredIndex.Stats before = tier.stats();
            for (int i = 0; i < writes; i++) {
                added[i] = near(data[(int) (i * 7919L % n)], r);
                tier.add((long) n + i, added[i]);
                Hit[] h = tier.search(added[i], k, null, ef);
                if (!holds(h, (long) n + i)) newMissed++;
                if (!first(h, (long) n + i)) newNotFirst++;
            }
            for (int id = 0; id < reused; id++) {
                float[] old = data[id];
                tier.delete(id);
                if (holds(tier.search(old, k, null, ef), id)) deletedReturned++;
                replaced[id] = near(data[(int) ((id + n / 2L) % n)], r);
                tier.add(id, replaced[id]);
                Hit[] h = tier.search(replaced[id], k, null, ef);
                if (!holds(h, id)) reusedMissed++;
                if (!first(h, id)) reusedNotFirst++;
                staleNext += stale(h, id, replaced[id], replaced[id]) + stale(tier.search(old, k, null, ef), id, old, replaced[id]);
            }
            tier.sealNow();
            boolean idle = tier.awaitIdle(600_000L);
            int newFirstSettled = 0, reusedFirstSettled = 0, staleSettled = 0;
            for (int i = 0; i < writes; i++) if (first(tier.search(added[i], k, null, ef), (long) n + i)) newFirstSettled++;
            for (int id = 0; id < reused; id++) {
                Hit[] h = tier.search(replaced[id], k, null, ef);
                if (first(h, id)) reusedFirstSettled++;
                staleSettled += stale(h, id, replaced[id], replaced[id]) + stale(tier.search(data[id], k, null, ef), id, data[id], replaced[id]);
            }
            TieredIndex.Stats after = tier.stats();
            long expected = (long) n + writes;
            out.line(f("PROPS newIds=%d newMissedByNextSearch=%d newNotFirstAtOnce=%d reusedIds=%d deletedIdReturned=%d reusedMissedByNextSearch=%d reusedNotFirstAtOnce=%d"
                            + " staleScoresAtOnce=%d settledIdle=%b settledNewFirst=%d settledReusedFirst=%d settledStaleScores=%d sealsDuring=%d mergesDuring=%d size=%d expectedSize=%d",
                    writes, newMissed, newNotFirst, reused, deletedReturned, reusedMissed, reusedNotFirst, staleNext, idle, newFirstSettled, reusedFirstSettled, staleSettled,
                    after.seals() - before.seals(), after.merges() - before.merges(), tier.size(), expected));
            if (newMissed + reusedMissed > 0) fail("props: " + (newMissed + reusedMissed) + " writes were not in the answer of the next search");
            if (deletedReturned > 0) fail("props: " + deletedReturned + " searches returned an id after its delete");
            if (staleNext + staleSettled > 0) fail("props: " + (staleNext + staleSettled) + " answers held a reused id with the score of another vector");
            if (!idle) fail("props: the tier was not idle after the writes");
            if (tier.size() != expected) fail("props: size is " + tier.size() + ", expected " + expected);
            if (tier.lastError() != null) fail("props: tier.lastError is " + tier.lastError());
        }

        // ---- CONC

        void concurrent() throws Exception {
            final int seconds = o.i("conc-seconds");
            if (seconds <= 0) return;
            phase("concurrent");
            final int writers = Math.max(1, o.i("writers")), searchers = Math.max(1, o.i("searchers")), windowS = Math.max(1, o.i("conc-window-s"));
            final double writeRate = o.d("write-rate"), deleteRate = o.d("delete-rate");
            final AtomicBoolean stop = new AtomicBoolean();
            final ConcurrentLinkedQueue<Throwable> errors = new ConcurrentLinkedQueue<>();
            final AtomicLong[] added = new AtomicLong[writers];
            for (int w = 0; w < writers; w++) added[w] = new AtomicLong();
            final AtomicLong deleted = new AtomicLong(), searches = new AtomicLong(), orderViol = new AtomicLong(), deletedReturned = new AtomicLong();
            final AtomicLong deletedBelowV = new AtomicLong(); // written by the deleter only: every id below it has been deleted
            final LongList[] lat = new LongList[searchers], win = new LongList[searchers];
            final long start = System.nanoTime();
            final int windows = (seconds + windowS - 1) / windowS;
            List<Thread> ths = new ArrayList<>();
            for (int w = 0; w < writers; w++) {
                final int wi = w;
                ths.add(new Thread(() -> {
                    try {
                        long t0 = System.nanoTime(), j = 0;
                        double rate = writeRate / writers;
                        while (!stop.get()) {
                            long id = ADD_BASE + ((long) wi * ADD_STRIDE) + j;
                            tier.add(id, gen.vec(id));
                            j++;
                            added[wi].set(j);
                            pace(t0, j, rate);
                        }
                    } catch (Throwable e) {
                        errors.add(e);
                    }
                }, "bench-writer-" + w));
            }
            if (deleteRate > 0) {
                ths.add(new Thread(() -> {
                    try {
                        long t0 = System.nanoTime(), j = 0;
                        while (!stop.get() && j < n) {
                            tier.delete(j);
                            j++;
                            deleted.set(j);
                            deletedBelowV.set(j);
                            pace(t0, j, deleteRate);
                        }
                    } catch (Throwable e) {
                        errors.add(e);
                    }
                }, "bench-deleter"));
            }
            for (int s = 0; s < searchers; s++) {
                final int si = s;
                lat[s] = new LongList();
                win[s] = new LongList();
                ths.add(new Thread(() -> {
                    try {
                        long c = si * 7919L;
                        while (!stop.get()) {
                            float[] q = qs[(int) (c++ % nq)];
                            long below = deletedBelowV.get();
                            long a = System.nanoTime();
                            Hit[] h = tier.search(q, k, null, ef);
                            long b = System.nanoTime();
                            lat[si].add(b - a);
                            win[si].add(Math.min(windows - 1, (b - start) / (windowS * 1_000_000_000L)));
                            searches.incrementAndGet();
                            String m = malformed(h, k);
                            if (m != null) {
                                orderViol.incrementAndGet();
                                if (errors.size() < 20) errors.add(new IllegalStateException("malformed answer: " + m));
                            }
                            for (Hit x : h) if (x.id() < below) deletedReturned.incrementAndGet();
                        }
                    } catch (Throwable e) {
                        errors.add(e);
                    }
                }, "bench-searcher-" + s));
            }
            Gc g0 = Gc.take();
            for (Thread t : ths) t.start();
            long end = start + seconds * 1_000_000_000L;
            long nextLine = start + windowS * 1_000_000_000L;
            while (System.nanoTime() < end) {
                LockSupport.parkNanos(Math.min(200_000_000L, Math.max(1, end - System.nanoTime())));
                if (System.nanoTime() >= nextLine) {
                    nextLine += windowS * 1_000_000_000L;
                    long ad = 0;
                    for (AtomicLong x : added) ad += x.get();
                    TieredIndex.Stats st = tier.stats();
                    out.line(f("CONC-T seconds=%.0f adds=%d deletes=%d searches=%d tail=%d segments=%d seals=%d merges=%d backpressureWaits=%d errors=%d", (System.nanoTime() - start) / 1e9, ad,
                            deleted.get(), searches.get(), st.tailSize(), st.segments(), st.seals(), st.merges(), st.backpressureWaits(), errors.size()));
                }
            }
            stop.set(true);
            for (Thread t : ths) {
                t.join(180_000L);
                if (t.isAlive()) errors.add(new IllegalStateException("thread did not finish: " + t.getName()));
            }
            double elapsed = (System.nanoTime() - start) / 1e9;
            long totalAdds = 0;
            for (AtomicLong x : added) totalAdds += x.get();
            long dels = deleted.get();
            LongList all = new LongList();
            for (LongList l : lat) for (int i = 0; i < l.n; i++) all.add(l.a[i]);
            long[] so = Arrays.copyOf(all.a, all.n);
            Arrays.sort(so);
            out.line(f("CONC seconds=%.1f writers=%d searchers=%d addsTargetPerSecond=%s addsPerSecond=%.0f adds=%d deletesTargetPerSecond=%s deletesPerSecond=%.0f deletes=%d searchesPerSecond=%.0f searches=%d"
                            + " searchP50us=%d searchP99us=%d searchP999us=%d searchMaxUs=%d errors=%d orderViolations=%d deletedReturned=%d %s", elapsed, writers, searchers,
                    writeRate <= 0 ? "unthrottled" : String.valueOf(writeRate), totalAdds / elapsed, totalAdds, deleteRate <= 0 ? "none" : String.valueOf(deleteRate), dels / elapsed, dels,
                    searches.get() / elapsed, searches.get(), pct(so, .5) / 1000, pct(so, .99) / 1000, pct(so, .999) / 1000, so.length == 0 ? 0 : so[so.length - 1] / 1000, errors.size(),
                    orderViol.get(), deletedReturned.get(), Gc.take().since(g0)));
            for (int w = 0; w < windows; w++) {
                LongList one = new LongList();
                for (int s = 0; s < searchers; s++) for (int i = 0; i < lat[s].n; i++) if (win[s].a[i] == w) one.add(lat[s].a[i]);
                long[] ws = Arrays.copyOf(one.a, one.n);
                Arrays.sort(ws);
                out.line(f("CONC-WINDOW window=%d fromSeconds=%d searches=%d searchesPerSecond=%.0f p50us=%d p99us=%d maxUs=%d", w, w * windowS, ws.length,
                        ws.length / (double) windowS, pct(ws, .5) / 1000, pct(ws, .99) / 1000, ws.length == 0 ? 0 : ws[ws.length - 1] / 1000));
            }
            int shown = 0;
            for (Throwable e : errors) {
                if (shown++ < 5) out.line("CONC-ERROR " + e);
            }
            if (!errors.isEmpty()) fail("concurrent phase: " + errors.size() + " errors, first: " + errors.peek());
            if (orderViol.get() > 0) fail("concurrent phase: " + orderViol.get() + " malformed answers");
            if (deletedReturned.get() > 0) fail("concurrent phase: " + deletedReturned.get() + " answers held an id whose delete had returned before the search began");

            // the end state
            long tIdle = System.nanoTime();
            boolean idle = tier.awaitIdle(600_000L);
            out.line(f("CONC-SETTLE idle=%b seconds=%.1f %s", idle, (System.nanoTime() - tIdle) / 1e9, stats(tier.stats())));
            long expected = (long) n - dels + totalAdds;
            boolean sizeOk = tier.size() == expected;
            out.line("CONC-CHECK size=" + tier.size() + " expected=" + expected + " sizeMatches=" + sizeOk + " lastError=" + tier.lastError() + " workerAlive=" + tier.workerAlive());
            if (!sizeOk) fail("size after the concurrent phase is " + tier.size() + ", expected " + expected + " (lost or phantom vectors)");
            if (tier.lastError() != null) fail("tier.lastError after the concurrent phase: " + tier.lastError());
            if (!tier.workerAlive()) fail("the tier worker is not alive after the concurrent phase");
            memory(out, "afterConcurrent");

            // recall of the end state against an exact scan of the live set
            final int live = (int) expected;
            final float[][] vecs = new float[live][];
            final long[] ids = new long[live];
            int p = 0;
            for (long i = dels; i < n; i++) {
                vecs[p] = data[(int) i];
                ids[p++] = i;
            }
            final int base = p;
            final long[] startOf = new long[writers + 1];
            for (int w = 0; w < writers; w++) startOf[w + 1] = startOf[w] + added[w].get();
            parallel(threads, (int) totalAdds, (t, lo, hi) -> {
                for (int x = lo; x < hi; x++) {
                    int w = 0;
                    while (startOf[w + 1] <= x) w++;
                    long id = ADD_BASE + ((long) w * ADD_STRIDE) + (x - startOf[w]);
                    vecs[base + x] = gen.vec(id);
                    ids[base + x] = id;
                }
            });
            int fq = Math.min(nq, o.i("final-queries"));
            float[][] fqs = Arrays.copyOf(qs, fq);
            long[][][] ft = exact(vecs, ids, live, fqs, k, new double[0], filterSeed, threads);
            double rs = 0;
            int bad = 0;
            long[] fl = new long[fq];
            for (int q = 0; q < fq; q++) {
                long a = System.nanoTime();
                Hit[] h = tier.search(fqs[q], k, null, ef);
                fl[q] = System.nanoTime() - a;
                rs += recall(h, ft[0][q]);
                if (malformed(h, k) != null) bad++;
            }
            Arrays.sort(fl);
            double rec = rs / Math.max(1, fq);
            out.line(f("CONC-RECALL queries=%d live=%d recall=%.4f p50us=%d p99us=%d badAnswers=%d (the tier is idle, unfiltered, single thread)", fq, live, rec, pct(fl, .5) / 1000, pct(fl, .99) / 1000, bad));
            if (bad > 0) fail("recall check after the concurrent phase: " + bad + " malformed answers");
            if (rec < minRecall) fail(f("recall after the concurrent phase is %.4f, below the floor %.4f", rec, minRecall));
        }

        static void pace(long startNs, long done, double rate) {
            if (rate <= 0) return;
            long due = startNs + (long) (done / rate * 1e9);
            long now = System.nanoTime();
            if (due > now) LockSupport.parkNanos(due - now);
        }
    }

    public static void main(String[] args) throws Exception {
        Opts o;
        Plan plan;
        try {
            o = new Opts(args);
            plan = prepare(o);
        } catch (IllegalArgumentException e) {
            System.err.println(e.getMessage());
            System.exit(2);
            return;
        }
        Out out = new Out(o.s("out"));
        int code = 0;
        try {
            Run r = new Run(o, out, plan);
            if (plan.real == null) r.gen = new Gen(r.dim, r.seed, o.i("clusters"), (float) o.d("sigma"));
            header(out, o, r.gen, plan.real);
            long t0 = System.nanoTime();
            try {
                r.data();
                r.build();
                r.search();
                r.matched();
                r.disk();
                r.concurrent();
                r.props();
            } catch (Throwable e) {
                r.fail("the run ended with " + e);
                e.printStackTrace(System.out);
            }
            out.line(f("MEMORY end rssPeakMB=%.0f", statusKb("VmHWM") / 1024.0));
            out.line(f("BENCH-SUMMARY status=%s failures=%d wallSeconds=%.0f ended=%s", r.failures.isEmpty() ? "OK" : "FAIL", r.failures.size(), (System.nanoTime() - t0) / 1e9, Instant.now()));
            for (String x : r.failures) out.line("BENCH-FAILURE " + x);
            out.line("BENCH-END");
            code = r.failures.isEmpty() ? 0 : 1;
            if (r.tier != null) r.tier.close();
            if (r.arm != null) {
                try {
                    r.arm.close();
                } catch (Exception e) {
                    System.err.println("closing the JVector arm failed: " + e);
                }
            }
        } finally {
            out.close();
        }
        System.exit(code);
    }
}
