package ai.humem.tiervec;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * The ground truth of the agent-memory benchmark (docs/AGENT-MEMORY-BENCHMARK.md, issue #159): the live rows of one collection with their
 * current vectors, and the exact answers over them. It scores with its own scalar cosine in double precision and calls none of the kernels of
 * the library, so that a mistake in a kernel cannot hide itself.
 */
final class LiveRows {
    /** a score that differs from the exact one by more than this is the score of another vector; it covers the rounding of float32 kernels */
    static final double TOLERANCE = 1e-4;
    /** two exact scores this close are a tie when recall is scored */
    private static final double TIE = 1e-6;

    private final Map<Long, float[]> rows = new HashMap<>();

    /** applies one write that is not a batch, or throws and changes nothing; the vector is held, not copied */
    void apply(Runbook.Op write) {
        switch (write.kind()) {
            case CREATE, REUSE -> {
                if (rows.putIfAbsent(write.id(), write.vec()) != null) throw new IllegalStateException(write.kind() + " of the live id " + write.id());
            }
            case UPDATE -> {
                if (rows.replace(write.id(), write.vec()) == null) throw new IllegalStateException("UPDATE of the id " + write.id() + ", which is not live");
            }
            case DELETE -> {
                if (rows.remove(write.id()) == null) throw new IllegalStateException("DELETE of the id " + write.id() + ", which is not live");
            }
            default -> throw new IllegalArgumentException(write.kind() + " is not a single write");
        }
    }

    boolean holds(long id) { return rows.containsKey(id); }

    /** the current vector of the id, or null when it is not live */
    float[] vector(long id) { return rows.get(id); }

    int size() { return rows.size(); }

    static double cosine(float[] a, float[] b) {
        double ab = 0, aa = 0, bb = 0;
        for (int i = 0; i < a.length; i++) {
            ab += (double) a[i] * b[i];
            aa += (double) a[i] * a[i];
            bb += (double) b[i] * b[i];
        }
        return aa > 0 && bb > 0 ? ab / Math.sqrt(aa * bb) : 0;
    }

    /** the exact answer: the best live rows, best first, a tie broken by the smaller id */
    record Top(long[] ids, double[] scores) {}

    Top top(float[] q, int k) {
        int n = Math.min(k, rows.size()), held = 0;
        long[] ids = new long[n];
        double[] scores = new double[n];
        for (Map.Entry<Long, float[]> e : rows.entrySet()) {
            long id = e.getKey();
            double s = cosine(q, e.getValue());
            if (held == n && (n == 0 || s < scores[n - 1] || s == scores[n - 1] && id > ids[n - 1])) continue;
            int j = held < n ? held++ : n - 1;
            while (j > 0 && (scores[j - 1] < s || scores[j - 1] == s && ids[j - 1] > id)) {
                scores[j] = scores[j - 1];
                ids[j] = ids[j - 1];
                j--;
            }
            scores[j] = s;
            ids[j] = id;
        }
        return new Top(ids, scores);
    }

    /**
     * Recall of an answer against the exact one: the share of the {@code truth.ids().length} places that the answer fills with different live
     * ids whose exact score is at least the last score of the truth (so a tie at the boundary counts). 1 when no row is live.
     */
    double recall(Top truth, float[] q, Hit[] hits) {
        int n = truth.ids().length;
        if (n == 0) return 1;
        double last = truth.scores()[n - 1];
        Set<Long> found = new HashSet<>();
        for (Hit h : hits) {
            float[] v = rows.get(h.id());
            if (v != null && cosine(q, v) >= last - TIE) found.add(h.id());
        }
        return (double) Math.min(n, found.size()) / n;
    }
}
