package com.humemai.vec;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.function.LongPredicate;

/** Tiny assert helper and reference code for the tier tests (named TierT so it does not clash with the flat and graph T classes). */
final class TierT {
    private TierT() {}

    static int pass, fail;

    /** false while a JUnit wrapper runs the suite inside the test JVM: done() must then not end the JVM, the wrapper asserts the counters */
    static boolean exitOnFailure = true;

    static synchronized void check(boolean ok, String name) {
        if (ok) pass++;
        else {
            fail++;
            System.out.println("  FAIL " + name);
        }
    }

    static void done(String suite) {
        System.out.println(suite + ": " + pass + " passed, " + fail + " failed");
        if (fail > 0 && exitOnFailure) System.exit(1);
    }

    /** deterministic unit vector by (seed, id) */
    static float[] vec(long seed, long id, int dim) {
        SplittableRandom r = new SplittableRandom(seed * 0x9E3779B97F4A7C15L + id * 0xD1B54A32D192ED03L + 17);
        float[] v = new float[dim];
        double n = 0;
        for (int i = 0; i < dim; i++) {
            v[i] = (float) r.nextGaussian();
            n += (double) v[i] * v[i];
        }
        float inv = (float) (1.0 / Math.sqrt(n));
        for (int i = 0; i < dim; i++) v[i] *= inv;
        return v;
    }

    /** higher is better for every metric, the same convention as Hit.score */
    static float score(Metric m, float[] q, float[] v) {
        double dot = 0, nq = 0, nv = 0, l2 = 0;
        for (int i = 0; i < q.length; i++) {
            dot += (double) q[i] * v[i];
            nq += (double) q[i] * q[i];
            nv += (double) v[i] * v[i];
            double d = (double) q[i] - v[i];
            l2 += d * d;
        }
        if (m == Metric.L2) return (float) -l2;
        if (m == Metric.DOT) return (float) dot;
        return (float) (dot / Math.sqrt(nq * nv));
    }

    static Hit[] brute(Map<Long, float[]> model, float[] q, int k, LongPredicate allow, Metric m) {
        List<Hit> all = new ArrayList<>(model.size());
        for (Map.Entry<Long, float[]> e : model.entrySet()) {
            if (allow != null && !allow.test(e.getKey())) continue;
            all.add(new Hit(e.getKey(), score(m, q, e.getValue())));
        }
        all.sort((a, b) -> {
            int c = Float.compare(b.score(), a.score());
            return c != 0 ? c : Long.compare(a.id(), b.id());
        });
        return all.subList(0, Math.min(k, all.size())).toArray(new Hit[0]);
    }

    static boolean containsId(Hit[] h, long id) {
        for (Hit x : h) if (x.id() == id) return true;
        return false;
    }

    static double recall(Hit[] got, Hit[] truth) {
        if (truth.length == 0) return 1.0;
        Set<Long> g = new HashSet<>();
        for (Hit h : got) g.add(h.id());
        int c = 0;
        for (Hit t : truth) if (g.contains(t.id())) c++;
        return (double) c / truth.length;
    }

    static boolean sameIdSet(Hit[] a, Hit[] b) {
        if (a.length != b.length) return false;
        return recall(a, b) == 1.0;
    }

    static boolean sortedDesc(Hit[] h) {
        for (int i = 1; i < h.length; i++) if (h[i - 1].score() < h[i].score()) return false;
        return true;
    }

    static boolean distinct(Hit[] h) {
        Set<Long> s = new HashSet<>();
        for (Hit x : h) if (!s.add(x.id())) return false;
        return true;
    }

    /** percentile (0..1) of the first n values */
    static long pct(long[] a, int n, double p) {
        if (n == 0) return 0;
        long[] c = Arrays.copyOf(a, n);
        Arrays.sort(c);
        return c[Math.min(n - 1, (int) Math.floor(p * n))];
    }
}
