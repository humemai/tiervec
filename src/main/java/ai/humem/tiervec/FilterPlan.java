package ai.humem.tiervec;

import java.util.function.IntPredicate;
import java.util.function.IntToLongFunction;
import java.util.function.LongPredicate;

/**
 * Plan of a filtered graph search: walk the graph, or scan the allowed ids exactly from the start.
 * The walk of a filtered search spends its visit budget before it knows that too few ids qualify, and the exact scan then runs on top of it.
 * When the allowed share is small, the scan alone is cheaper and exact. The share is estimated from a fixed-size sample of the id slots, so
 * the cost of the estimate does not grow with the index (SAMPLE predicate calls) and the host does not have to count its allowed ids.
 */
final class FilterPlan {
    /** predicate calls spent on the estimate; an index with fewer id slots is sampled completely */
    static final int SAMPLE = 256;
    private static final double PHI = 0.6180339887498949;

    private FilterPlan() {}

    static void check(double below) {
        if (!(below >= 0 && below <= 1)) throw new IllegalArgumentException("filterScanBelow must be in [0, 1], got " + below);
    }

    /**
     * True when the sampled live slots pass the filter at a share of at most {@code below}: the caller scans exactly.
     * The slots are spread over [0, slots) with a golden-ratio sequence, so the sample is the same for every query and does not line up
     * with a regular pattern of ids. When no sampled slot is live the answer is false (the walk handles an empty index).
     */
    static boolean scanFirst(double below, int slots, IntPredicate live, IntToLongFunction idOf, LongPredicate allow) {
        if (below <= 0 || allow == null || slots <= 0) return false;
        int m = Math.min(SAMPLE, slots);
        int seen = 0, pass = 0;
        for (int j = 0; j < m; j++) {
            int o = m == slots ? j : Math.min(slots - 1, (int) (((j + 0.5) * PHI % 1.0) * slots));
            if (!live.test(o)) continue;
            seen++;
            if (allow.test(idOf.applyAsLong(o))) pass++;
        }
        return seen > 0 && pass <= below * seen;
    }
}
