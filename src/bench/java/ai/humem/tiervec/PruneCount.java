package ai.humem.tiervec;

import java.util.Arrays;

/**
 * What the neighbour selection of {@link GraphIndex} does, by the place that calls it, for the agent-memory benchmark (docs/SCREEN-IDEAS.md,
 * issue #161). Only the copy of the library that scripts/instrument_distances.py makes calls {@link #site} and {@link #selected}; in a build
 * of the library as it is, every count stays 0. The counts are plain statics: the replays of the benchmark write from one thread at a time.
 */
final class PruneCount {
    /** the callers of the selection: the lists of a new node, the list of a neighbour that is full, and the repair of consolidate() */
    static final String[] SITES = {"insert", "reverse", "repair"};
    static final String[] COUNTS = {"calls", "candidates", "examined", "keptByRule", "keptByFill", "stoppedAtLimit"};

    private static int site;
    private static final long[][] TOTAL = new long[SITES.length][COUNTS.length];

    private PruneCount() {}

    static void site(int s) { site = s; }

    /**
     * One selection, after the passes of the diversity rule and before its fill: of {@code n} candidates in increasing distance the rule kept
     * {@code m}, those marked in {@code taken}, with room for {@code limit}. {@code later} marks the candidates that the first pass left to the
     * second one (alpha above 1). The first pass stops when the list is full; then the candidates after the last kept one were never looked
     * at. If the second pass kept any, the first one did not fill the list, so it looked at every candidate.
     */
    static void selected(boolean[] taken, boolean[] later, int n, int m, int limit, boolean fill) {
        int examined = n;
        if (m >= limit) {
            int last = -1;
            boolean second = false;
            for (int i = 0; i < n; i++) {
                if (!taken[i]) continue;
                last = i;
                second |= later[i];
            }
            if (!second) examined = last + 1;
        }
        long[] t = TOTAL[site];
        t[0]++;
        t[1] += n;
        t[2] += examined;
        t[3] += m;
        t[4] += fill ? Math.max(0, Math.min(limit - m, n - m)) : 0;
        t[5] += examined < n ? 1 : 0;
    }

    static void reset() { for (long[] t : TOTAL) Arrays.fill(t, 0); }

    static long count(int s, int what) { return TOTAL[s][what]; }
}
