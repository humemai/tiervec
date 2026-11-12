package ai.humem.tiervec;

import java.util.Random;
import java.util.function.LongPredicate;

/** Assert helper, seeded clustered data and a plain scalar brute-force oracle for the IVF tests (independent of the Kernels class). */
final class IvfT {
    private IvfT() {}

    static int pass, fail;

    /** false while a JUnit wrapper runs the suite inside the test JVM: done() must then not end the JVM, the wrapper asserts the counters */
    static boolean exitOnFailure = true;

    static synchronized void check(boolean ok, String name) {
        if (ok) pass++;
        else {
            fail++;
            System.out.println("FAIL " + name);
        }
    }

    static void section(String name, int passBefore, int failBefore) {
        System.out.println((fail == failBefore ? "PASS " : "FAIL ") + name + " (" + (pass - passBefore) + " checks, " + (fail - failBefore) + " failed)");
    }

    static void done(String suite) {
        System.out.println(suite + ": " + pass + " passed, " + fail + " failed");
        if (fail > 0 && exitOnFailure) System.exit(1);
    }

    // ---- data ----

    static float[][] centers(int clusters, int dim, long seed) {
        Random r = new Random(seed);
        float[][] c = new float[clusters][dim];
        for (int i = 0; i < clusters; i++) for (int j = 0; j < dim; j++) c[i][j] = (float) r.nextGaussian();
        return c;
    }

    static float[] point(float[][] centers, Random r, double sigma) {
        float[] c = centers[r.nextInt(centers.length)];
        float[] v = new float[c.length];
        for (int i = 0; i < v.length; i++) v[i] = c[i] + (float) (r.nextGaussian() * sigma);
        return v;
    }

    static float[][] clustered(int n, int dim, int clusters, double sigma, long centerSeed, long pointSeed, float shift) {
        float[][] centers = centers(clusters, dim, centerSeed);
        Random r = new Random(pointSeed);
        float[][] out = new float[n][];
        for (int i = 0; i < n; i++) {
            out[i] = point(centers, r, sigma);
            if (shift != 0f) for (int j = 0; j < dim; j++) out[i][j] += shift;
        }
        return out;
    }

    static float[] normalize(float[] v) {
        double s = 0;
        for (float x : v) s += (double) x * x;
        float[] r = new float[v.length];
        if (s > 0) {
            double inv = 1.0 / Math.sqrt(s);
            for (int i = 0; i < r.length; i++) r[i] = (float) (v[i] * inv);
        }
        return r;
    }

    /** the stored form of rows [from, to): normalized for COSINE, the same arrays otherwise */
    static float[][] stored(Metric m, float[][] raw, int from, int to) {
        float[][] r = new float[to - from][];
        for (int i = from; i < to; i++) r[i - from] = m == Metric.COSINE ? normalize(raw[i]) : raw[i];
        return r;
    }

    // ---- oracle ----

    static float score(Metric m, float[] a, float[] b) {
        double s = 0;
        if (m == Metric.L2) {
            for (int i = 0; i < a.length; i++) {
                double d = (double) a[i] - b[i];
                s -= d * d;
            }
        } else {
            for (int i = 0; i < a.length; i++) s += (double) a[i] * b[i];
        }
        return (float) s;
    }

    static boolean better(float s1, long i1, float s2, long i2) {
        return s1 > s2 || (s1 == s2 && i1 < i2);
    }

    /** exact top k over stored[i] with id ids[i] (a null stored[i] or alive[i] == false is skipped); qraw is the raw query */
    static Hit[] brute(Metric m, long[] ids, float[][] stored, boolean[] alive, float[] qraw, int k, LongPredicate allow) {
        float[] q = m == Metric.COSINE ? normalize(qraw) : qraw;
        int cnt = 0;
        float[] ts = new float[k];
        long[] ti = new long[k];
        for (int i = 0; i < ids.length; i++) {
            if (stored[i] == null || (alive != null && !alive[i])) continue;
            if (allow != null && !allow.test(ids[i])) continue;
            float sc = score(m, stored[i], q);
            if (cnt < k || better(sc, ids[i], ts[cnt - 1], ti[cnt - 1])) {
                int pos = cnt < k ? cnt++ : k - 1;
                while (pos > 0 && better(sc, ids[i], ts[pos - 1], ti[pos - 1])) {
                    ts[pos] = ts[pos - 1];
                    ti[pos] = ti[pos - 1];
                    pos--;
                }
                ts[pos] = sc;
                ti[pos] = ids[i];
            }
        }
        Hit[] r = new Hit[cnt];
        for (int i = 0; i < cnt; i++) r[i] = new Hit(ti[i], ts[i]);
        return r;
    }

    static long[] seqIds(int n) {
        long[] r = new long[n];
        for (int i = 0; i < n; i++) r[i] = i;
        return r;
    }

    static double recall(Hit[] got, Hit[] truth) {
        if (truth.length == 0) return 1.0;
        int hit = 0;
        for (Hit t : truth) {
            for (Hit g : got) {
                if (g.id() == t.id()) {
                    hit++;
                    break;
                }
            }
        }
        return (double) hit / truth.length;
    }

    static boolean sameIds(Hit[] a, Hit[] b) {
        if (a.length != b.length) return false;
        java.util.HashSet<Long> s = new java.util.HashSet<>();
        for (Hit h : a) s.add(h.id());
        for (Hit h : b) if (!s.contains(h.id())) return false;
        return true;
    }

    static Hit[][] truths(Metric m, long[] ids, float[][] stored, boolean[] alive, float[][] qs, int k, LongPredicate allow) {
        Hit[][] t = new Hit[qs.length][];
        for (int i = 0; i < qs.length; i++) t[i] = brute(m, ids, stored, alive, qs[i], k, allow);
        return t;
    }

    /** mean recall@k and mean microseconds per query at the given nprobe (0 = the index default); one warm-up pass first */
    static double[] eval(IvfIndex ix, float[][] qs, Hit[][] truth, int nprobe, int k) {
        for (int i = 0; i < Math.min(qs.length, 50); i++) ix.search(qs[i], k, null, nprobe);
        double rsum = 0;
        long t0 = System.nanoTime();
        Hit[][] got = new Hit[qs.length][];
        for (int i = 0; i < qs.length; i++) got[i] = ix.search(qs[i], k, null, nprobe);
        long dt = System.nanoTime() - t0;
        for (int i = 0; i < qs.length; i++) rsum += recall(got[i], truth[i]);
        return new double[] {rsum / qs.length, dt / 1000.0 / qs.length};
    }
}
