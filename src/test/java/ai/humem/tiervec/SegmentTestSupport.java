package ai.humem.tiervec;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

/** Data and helpers shared by the segment, manifest and tier-reopen tests. */
final class SegmentTestSupport {
    private SegmentTestSupport() {}

    static float[][] vectors(int n, int dim, long seed) {
        Random r = new Random(seed);
        float[][] v = new float[n][dim];
        for (float[] x : v) {
            for (int j = 0; j < dim; j++) x[j] = r.nextFloat() - 0.5f;
        }
        return v;
    }

    /** ids that are unique for i below 100003 and not in row order, so the stored id permutation is not the identity */
    static long idOf(int i) {
        return 1000L + (long) i * 7919L % 100003L;
    }

    static GraphIndex.Params params(boolean sq8) {
        return GraphIndex.Params.defaults().withEfConstruction(64).withSq8(sq8);
    }

    /** a sealed graph over the rows of v, row i having id idOf(i) */
    static SealedGraph graph(float[][] v, Metric metric, GraphIndex.Params p) {
        GraphIndex gi = new GraphIndex(v[0].length, metric, p);
        for (int i = 0; i < v.length; i++) gi.add(idOf(i), v[i]);
        return gi.seal();
    }

    /** per-row versions for a segment of n rows */
    static long[] versOf(int n) {
        long[] v = new long[n];
        for (int i = 0; i < n; i++) v[i] = 10L + i;
        return v;
    }

    private static double score(float[] a, float[] b, Metric m) {
        double dot = 0, na = 0, nb = 0, sq = 0;
        for (int j = 0; j < a.length; j++) {
            dot += (double) a[j] * b[j];
            na += (double) a[j] * a[j];
            nb += (double) b[j] * b[j];
            double d = (double) a[j] - b[j];
            sq += d * d;
        }
        switch (m) {
            case L2: return -sq;
            case DOT: return dot;
            default: return dot / Math.sqrt(na * nb);
        }
    }

    /** ids of the exact top k of row i = idOf(i), by brute force */
    static Set<Long> exactTopK(float[][] data, float[] q, int k, Metric metric) {
        final double[] s = new double[data.length];
        Integer[] idx = new Integer[data.length];
        for (int i = 0; i < data.length; i++) {
            s[i] = score(q, data[i], metric);
            idx[i] = i;
        }
        Arrays.sort(idx, (x, y) -> Double.compare(s[y], s[x]));
        Set<Long> out = new HashSet<>();
        for (int i = 0; i < k; i++) out.add(idOf(idx[i]));
        return out;
    }

    static Set<Long> idSet(Hit[] hits) {
        Set<Long> out = new HashSet<>();
        for (Hit h : hits) out.add(h.id());
        return out;
    }

    static double recall(Hit[] hits, Set<Long> exact) {
        int found = 0;
        for (Hit h : hits) {
            if (exact.contains(h.id())) found++;
        }
        return (double) found / exact.size();
    }
}
