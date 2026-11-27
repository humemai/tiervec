package ai.humem.tiervec;

/**
 * The distance computations of the calling thread, full (float32) and quantized, for the agent-memory benchmark (issue #159). Only the copy of
 * the library that scripts/instrument_distances.py makes calls {@link #full} and {@link #quantized}; in a build of the library as it is, both
 * counts stay 0.
 */
final class DistanceCount {
    private static final ThreadLocal<long[]> COUNTS = ThreadLocal.withInitial(() -> new long[2]);

    private DistanceCount() {}

    static void full(int n) { COUNTS.get()[0] += n; }

    static void quantized(int n) { COUNTS.get()[1] += n; }

    static long fullSoFar() { return COUNTS.get()[0]; }

    static long quantizedSoFar() { return COUNTS.get()[1]; }

    static long soFar() { return fullSoFar() + quantizedSoFar(); }

    /** whether the library of this run counts: one exact search of two rows is tried */
    static boolean active() {
        try (FlatIndex two = new FlatIndex(2, Metric.DOT)) {
            two.add(1, new float[] {1, 0});
            two.add(2, new float[] {0, 1});
            long before = soFar();
            two.search(new float[] {1, 0}, 1, null);
            return soFar() > before;
        }
    }
}
