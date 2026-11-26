package ai.humem.tiervec;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * The child JVM of {@link TierHaltTest}: it builds a tier batch by batch, writes what each generation must serve to a file next to (not inside)
 * the index directory, and checkpoints into the directory after each batch. In the checkpoint that the parent names it halts itself with
 * {@link Runtime#halt} at the step that the parent names: no shutdown hook, no finally block, no flush of anything the process holds. It is not a
 * test (surefire does not pick the name up) and it is started only by {@code TierHaltTest}.
 *
 * <p>Arguments: index directory, directory for the expected answers, number of the checkpoint to cut (from 1), name of the {@link FaultHook.Step}, seed.
 */
public final class TierHaltChild {
    private TierHaltChild() {}

    public static void main(String[] args) throws Exception {
        final Path dir = Path.of(args[0]);
        final Path expect = Path.of(args[1]);
        final int haltIn = Integer.parseInt(args[2]);
        final FaultHook.Step haltAt = FaultHook.Step.valueOf(args[3]);
        final long seed = Long.parseLong(args[4]);
        final int rows = TierHaltTest.ROWS;

        float[][] data = SegmentTestSupport.vectors(TierHaltTest.GENS * rows, TierHaltTest.DIM, seed);
        float[][] qs = SegmentTestSupport.vectors(TierHaltTest.QUERIES, TierHaltTest.DIM, seed + 1);
        TieredIndex t = new TieredIndex(TierHaltTest.DIM, Metric.L2, TierReopenTest.cfg());
        for (int g = 1; g <= TierHaltTest.GENS; g++) {
            for (int i = (g - 1) * rows; i < g * rows; i++) t.add(SegmentTestSupport.idOf(i), data[i]);
            // from the second batch on, some ids of the batch before are deleted and some are replaced, so a generation holds tombstones
            if (g > 1) {
                for (int i = (g - 2) * rows; i < (g - 1) * rows; i += TierHaltTest.DELETE_EVERY) t.delete(SegmentTestSupport.idOf(i));
                for (int i = (g - 2) * rows + 1; i < (g - 1) * rows; i += TierHaltTest.DELETE_EVERY) t.add(SegmentTestSupport.idOf(i), data[i - 1]);
            }
            t.sealNow();
            if (!t.awaitIdle(60_000)) throw new IllegalStateException("the seal of batch " + g + " did not finish");
            // what generation g must serve is on disk before its publish starts, so that it is there whatever step the halt comes at
            Files.write(expect.resolve("gen-" + g + ".txt"), TierHaltTest.digest(t, qs));
            final int gen = g;
            t.checkpoint(dir, step -> {
                if (gen == haltIn && step == haltAt) Runtime.getRuntime().halt(TierHaltTest.HALT_CODE);
            });
        }
        t.close();
    }
}
