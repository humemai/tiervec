package ai.humem.tiervec;

import java.util.Arrays;
import java.util.SplittableRandom;
import java.util.function.LongPredicate;

/** Tiny test helper: asserts, deterministic vector generators, brute force, percentiles. No JUnit. */
final class GraphT {
    private GraphT() {}
    static int failures, checks;
    /** false while a JUnit wrapper runs the suite inside the test JVM: done() must then not end the JVM, the wrapper asserts the counters */
    static boolean exitOnFailure = true;

    static void check(boolean ok, String msg) {
        checks++;
        if (!ok) { failures++; System.out.println("  FAIL: " + msg); }
    }

    static void run(String name, Runnable r) {
        long t0 = System.nanoTime(); int f0 = failures;
        System.out.println("TEST " + name);
        try { r.run(); } catch (Throwable e) { failures++; System.out.println("  EXCEPTION: " + e); e.printStackTrace(System.out); }
        System.out.printf("%s %s (%.1f s)%n", failures == f0 ? "PASS" : "FAIL", name, (System.nanoTime() - t0) / 1e9);
    }

    /** Clustered generator: nClusters unit centroids, each vector = normalize(centroid + sigma * gaussian noise). Deterministic by (seed, id). */
    static final class Gen {
        final int dim; final long seed; final float[][] cent; final float sigma;
        Gen(int dim, long seed, int nClusters, float sigma) {
            this.dim = dim; this.seed = seed; this.sigma = sigma;
            cent = new float[Math.max(1, nClusters)][dim];
            SplittableRandom r = new SplittableRandom(seed ^ 0x5DEECE66DL);
            for (float[] c : cent) { for (int i = 0; i < dim; i++) c[i] = (float) gauss(r); norm(c); }
        }
        float[] vec(long id) {
            SplittableRandom r = new SplittableRandom(seed * 0x9E3779B97F4A7C15L + id * 0xD1B54A32D192ED03L + 17);
            float[] v = new float[dim];
            float[] c = cent[(int) (Math.floorMod(id * 0x9E3779B97F4A7C15L >>> 20, (long) cent.length))];
            for (int i = 0; i < dim; i++) v[i] = c[i] + sigma * (float) gauss(r) / (float) Math.sqrt(dim) * 4f;
            norm(v); return v;
        }
    }
    static double gauss(SplittableRandom r) {
        double u = 1 - r.nextDouble(), v = r.nextDouble();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }
    static void norm(float[] v) { double n = 0; for (float x : v) n += (double) x * x; float inv = (float) (1 / Math.sqrt(n)); for (int i = 0; i < v.length; i++) v[i] *= inv; }

    /** exact top-k ids by cosine/dot over data[ids...] (ids are indexes into data) honoring skip and allow */
    static long[] brute(float[][] data, boolean[] skip, float[] q, int k, LongPredicate allow, Metric m) {
        Heap h = new Heap();
        int dim = q.length;
        for (int i = 0; i < data.length; i++) {
            if (data[i] == null || (skip != null && skip[i]) || (allow != null && !allow.test(i))) continue;
            float d = m == Metric.L2 ? Kern.l2(q, 0, data[i], 0, dim) : -Kern.dot(q, 0, data[i], 0, dim);
            if (h.n < k) h.push(-d, i); else if (d < -h.topKey()) { h.pop(); h.push(-d, i); }
        }
        long[] out = new long[h.n];
        for (int i = out.length - 1; i >= 0; i--) { out[i] = h.topVal(); h.pop(); }
        return out;
    }

    static double recall(Hit[] got, long[] truth) {
        if (truth.length == 0) return 1;
        int c = 0;
        for (long t : truth) for (Hit h : got) if (h.id() == t) { c++; break; }
        return (double) c / truth.length;
    }

    static long pct(long[] sortedNs, double p) { return sortedNs.length == 0 ? 0 : sortedNs[(int) Math.min(sortedNs.length - 1, Math.floor(p * sortedNs.length))]; }
    static long[] sorted(long[] a, int n) { long[] c = Arrays.copyOf(a, n); Arrays.sort(c); return c; }
}
