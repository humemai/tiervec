package com.humemai.vec;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;

/**
 * Main program of NoIncubatorTest. It runs in a JVM that was started WITHOUT --add-modules jdk.incubator.vector: it builds a tier whose
 * sealed segments are graphs (fp32 and SQ8 codes), searches it, and compares the answers with an exact scan. It ends with exit code 0
 * and the line NO-INCUBATOR OK when everything holds. Plain Java, no JUnit.
 */
public final class NoIncubatorProbe {
    private NoIncubatorProbe() {}

    static final int DIM = 32, N = 3000, EXTRA = 60, QUERIES = 100, K = 10;

    static void fail(String msg) {
        System.out.println("FAIL " + msg);
        System.exit(1);
    }

    public static void main(String[] args) throws Exception {
        if (ModuleLayer.boot().findModule("jdk.incubator.vector").isPresent()) fail("the module jdk.incubator.vector is present, this probe needs a JVM without it");
        if (Kernels.VECTOR) fail("Kernels.VECTOR is true in a JVM without the module");
        System.out.println("scalarReason=" + Kernels.scalarReason);
        if (Kernels.scalarReason == null || !Kernels.scalarReason.startsWith("Vector API unavailable")) fail("unexpected scalarReason " + Kernels.scalarReason);
        run(false);
        run(true);
        System.out.println("NO-INCUBATOR OK");
    }

    static void run(boolean sq8) throws Exception {
        String label = sq8 ? "sq8" : "fp32";
        TieredIndex.Config c = new TieredIndex.Config(500, 0L, 300, 8, 4, 0.3, 0, 4, Quant.NONE, GraphIndex.Params.defaults().withEfConstruction(100).withSq8(sq8), 1);
        TieredIndex ix = new TieredIndex(DIM, Metric.COSINE, c);
        try {
            Random r = new Random(sq8 ? 6 : 5);
            float[][] v = new float[N + EXTRA][DIM];
            for (float[] x : v) for (int j = 0; j < DIM; j++) x[j] = r.nextFloat() - 0.5f;
            for (int i = 0; i < N; i++) ix.add(i, v[i]);
            if (!ix.awaitIdle(120_000)) fail(label + ": the tier did not become idle");
            if (ix.segmentCount() < 1) fail(label + ": no sealed graph segment was built");
            for (int i = N; i < N + EXTRA; i++) ix.add(i, v[i]);
            double recall = 0;
            for (int qi = 0; qi < QUERIES; qi++) {
                float[] q = new float[DIM];
                for (int j = 0; j < DIM; j++) q[j] = r.nextFloat() - 0.5f;
                long[] truth = exact(v, q);
                Hit[] h = ix.search(q, K, null, 200);
                int found = 0;
                for (Hit x : h) for (long t : truth) if (x.id() == t) found++;
                recall += (double) found / K;
            }
            recall /= QUERIES;
            System.out.println(label + ": segments " + ix.segmentCount() + ", recall@" + K + " " + recall);
            if (recall < 0.80) fail(label + ": recall@" + K + " is " + recall);
            for (int i = N; i < N + EXTRA; i++) {
                Hit[] h = ix.search(v[i], 1, null);
                if (h.length != 1 || h[0].id() != i) fail(label + ": the tail vector " + i + " is not the first hit for itself");
            }
            int missed = 0, probed = 0;
            for (int i = 0; i < N; i += 15) {
                boolean seen = false;
                for (Hit x : ix.search(v[i], 5, null)) if (x.id() == i) seen = true;
                probed++;
                if (!seen) missed++;
            }
            System.out.println(label + ": sealed vectors found by their own vector: " + (probed - missed) + " of " + probed);
            if (missed > probed / 50) fail(label + ": " + missed + " of " + probed + " sealed vectors are not among the first 5 hits for themselves");
            if (ix.size() != N + EXTRA || ix.stats().failures() != 0) fail(label + ": size " + ix.size() + ", failures " + ix.stats().failures());
        } finally {
            ix.close();
        }
    }

    /** ids of the K best cosine scores, by an exact scan in double precision */
    static long[] exact(float[][] v, float[] q) {
        double[] s = new double[v.length];
        double qq = 0;
        for (float f : q) qq += (double) f * f;
        for (int i = 0; i < v.length; i++) {
            double dot = 0, vv = 0;
            for (int j = 0; j < DIM; j++) {
                dot += (double) v[i][j] * q[j];
                vv += (double) v[i][j] * v[i][j];
            }
            s[i] = dot / Math.sqrt(vv * qq);
        }
        Integer[] idx = new Integer[v.length];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        Arrays.sort(idx, Comparator.comparingDouble((Integer i) -> -s[i]));
        long[] t = new long[K];
        for (int i = 0; i < K; i++) t[i] = idx[i];
        return t;
    }
}
