package com.humemai.vec;

import java.io.PrintWriter;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.LockSupport;
import java.util.zip.CRC32;
import java.util.function.LongPredicate;

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
        return d;
    }

    static final class Opts {
        final Map<String, String> m = new TreeMap<>();

        Opts(String[] a) {
            m.putAll(defaults());
            for (int i = 0; i < a.length; i += 2) {
                if (!a[i].startsWith("--") || !m.containsKey(a[i].substring(2)) || i + 1 >= a.length)
                    throw new IllegalArgumentException("bad option " + a[i] + " (options are --key value pairs: " + defaults().keySet() + ")");
                m.put(a[i].substring(2), a[i + 1]);
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

    static void header(Out out, Opts o, Gen gen) {
        out.line("BENCH-RESULT format=1 harness=com.humemai.vec.Bench");
        out.line("RUN started=" + Instant.now() + " host=" + hostLabel(o));
        String sha = sh("git", "-C", o.s("repo"), "rev-parse", "HEAD");
        String dirty = sh("git", "-C", o.s("repo"), "status", "--porcelain", "--untracked-files=no");
        out.line("COMMIT sha=" + (sha == null ? "unknown" : sha) + " modifiedFiles=" + (dirty == null ? "unknown" : String.valueOf(dirty.isEmpty() ? 0 : dirty.split("\n").length)));
        out.line("SEED seed=" + o.l("seed") + " data=clustered-gaussian clusters=" + o.i("clusters") + " sigma=" + o.d("sigma")
                + " filter=hash-uniform queryIdBase=" + QUERY_BASE);
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
                + " lanes=" + Kern.lanes() + " scalarProperty=" + System.getProperty("humemai.vec.scalar"));
        StringBuilder ps = new StringBuilder();
        for (Map.Entry<String, String> e : o.m.entrySet()) ps.append(ps.length() == 0 ? "" : " ").append(e.getKey()).append('=').append(e.getValue());
        out.line("PARAMS " + ps);
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

        Run(Opts o, Out out) {
            this.o = o;
            this.out = out;
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

        void data() throws Exception {
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
            CRC32 crc = new CRC32();
            for (int i = 0; i < Math.min(n, 1000); i++) for (float x : data[i]) for (int sh = 0; sh < 32; sh += 8) crc.update(Float.floatToIntBits(x) >>> sh);
            for (float x : qs[0]) for (int sh = 0; sh < 32; sh += 8) crc.update(Float.floatToIntBits(x) >>> sh);
            out.line(f("DATA n=%d dim=%d queries=%d checksumFirst1000AndQuery0=%08x generateSeconds=%.1f exactScanSeconds=%.1f exactScanThreads=%d %s", n, dim, nq, crc.getValue(), genS, exactS,
                    threads, Gc.take().since(g0)));
            StringBuilder tl = new StringBuilder();
            for (int g = 0; g < truth.length; g++) {
                int min = Integer.MAX_VALUE;
                for (long[] t : truth[g]) min = Math.min(min, t.length);
                tl.append(g == 0 ? "unfiltered" : f("share%s", o.ds("shares")[g - 1])).append(":minTruthSize=").append(min).append(' ');
            }
            out.line("TRUTH " + tl.toString().trim());
        }

        // ---- BUILD

        void build() throws Exception {
            phase("build");
            GraphIndex.Params gp0 = GraphIndex.Params.defaults().withM(o.i("m")).withEfConstruction(o.i("efc"));
            GraphIndex.Params gp = new GraphIndex.Params(gp0.M(), gp0.efConstruction(), ef, gp0.alpha(), gp0.sq8(), gp0.seed());
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
        }

        // ---- SEARCH

        void search() {
            phase("search");
            cell("unfiltered", -1, 0, null, truth[0]);
            for (int s = 0; s < shares.length; s++) {
                LongPredicate al = allow(shares[s], filterSeed);
                for (double sb : scanBelows) cell("filtered", shares[s], sb, al, truth[1 + s]);
            }
            tier.filterScanBelow(0);
        }

        void cell(String kind, double share, double scanBelow, LongPredicate al, long[][] tr) {
            tier.filterScanBelow(scanBelow);
            boolean scans = al != null && scanBelow >= share && scanBelow > 0;
            long[] lat = new long[reps * nq];
            double rsum = 0, rmin = 1, maxScoreErr = 0;
            int bad = 0;
            long ln = 0;
            Gc g0 = Gc.take();
            for (int pass = 0; pass <= reps; pass++) {
                for (int q = 0; q < nq; q++) {
                    long a = System.nanoTime();
                    Hit[] h = tier.search(qs[q], k, al, ef);
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
                            + " opsPerSecond=%.1f samples=%d badAnswers=%d maxScoreError=%.2e %s", kind, share < 0 ? "none" : String.valueOf(share), scanBelow, scans, k, ef, rec, rmin,
                    pct(so, .5) / 1000, pct(so, .99) / 1000, pct(so, .999) / 1000, sumS / Math.max(1, so.length) * 1e6, so.length == 0 ? 0 : so[so.length - 1] / 1000,
                    so.length / Math.max(1e-9, sumS), so.length, bad, maxScoreErr, Gc.take().since(g0)));
            if (bad > 0) fail("search cell " + kind + " share " + share + " scanBelow " + scanBelow + ": " + bad + " bad answers");
            double floor = scans ? minScanRecall : minRecall;
            if (rec < floor) fail(f("search cell %s share %s scanBelow %s: recall %.4f is below the floor %.4f", kind, share, scanBelow, rec, floor));
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
        try {
            o = new Opts(args);
        } catch (IllegalArgumentException e) {
            System.err.println(e.getMessage());
            System.exit(2);
            return;
        }
        Out out = new Out(o.s("out"));
        int code = 0;
        try {
            Run r = new Run(o, out);
            r.gen = new Gen(r.dim, r.seed, o.i("clusters"), (float) o.d("sigma"));
            header(out, o, r.gen);
            long t0 = System.nanoTime();
            try {
                r.data();
                r.build();
                r.search();
                r.concurrent();
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
        } finally {
            out.close();
        }
        System.exit(code);
    }
}
