package ai.humem.tiervec;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;

/**
 * The child JVM of {@link TierKillLongTest}: it opens the index directory (or starts an empty tier when there is no generation yet) and
 * publishes one generation after another until it is killed. Generation g adds the rows of batch g and deletes those of batch g - {@value
 * #KEPT}, which a merge then compacts, so that publishes also delete files. Before each checkpoint it writes what that generation must serve to
 * a file outside the index directory (under another name first, then renamed, so that the parent never reads half of it), and it prints one
 * line before and one after the checkpoint. It is not a test.
 *
 * <p>Arguments: index directory, directory for the expected answers.
 */
public final class TierKillChild {
    static final int ROWS = 150;
    static final int KEPT = 4;
    static final int TAIL = 12;
    static final int QUERIES = 15;
    /** a child that nobody kills stops by itself */
    private static final long LIFE_NANOS = 60_000_000_000L;

    private TierKillChild() {}

    static long id(long batch, int row) {
        return batch * 1_000_000L + row;
    }

    /** a line that the parent can read as soon as this returns: a kill takes what is still in the buffer of the stream */
    private static void say(String line) {
        System.out.println(line);
        System.out.flush();
    }

    public static void main(String[] args) throws Exception {
        final Path dir = Path.of(args[0]);
        final Path expect = Path.of(args[1]);
        final long start = System.nanoTime();
        final float[][] qs = SegmentTestSupport.vectors(QUERIES, TierHaltTest.DIM, 5);
        final Manifest found = Files.isDirectory(dir) ? Manifest.readLatest(dir) : null;
        final TieredIndex t = found == null ? new TieredIndex(TierHaltTest.DIM, Metric.L2, TierReopenTest.cfg()) : TieredIndex.open(dir, TierReopenTest.cfg());
        say("ready");
        for (long g = (found == null ? 0 : found.generation()) + 1; System.nanoTime() - start < LIFE_NANOS; g++) {
            float[][] rows = SegmentTestSupport.vectors(ROWS, TierHaltTest.DIM, 1000 + g);
            for (int i = 0; i < ROWS; i++) t.add(id(g, i), rows[i]);
            t.sealNow();
            if (g > KEPT) {
                for (int i = 0; i < ROWS; i++) t.delete(id(g - KEPT, i));
                t.mergeNow();
            }
            if (!t.awaitIdle(60_000)) throw new IllegalStateException("generation " + g + " did not settle");
            // some rows stay in the tail at the checkpoint (the next batch, or the next child, seals them), one of them over a sealed id
            float[][] extra = SegmentTestSupport.vectors(TAIL, TierHaltTest.DIM, 2000 + g);
            for (int i = 0; i < TAIL - 1; i++) t.add(id(g, ROWS + i), extra[i]);
            t.add(id(g, 0), extra[TAIL - 1]);
            List<String> lines = TierHaltTest.digest(t, qs);
            lines.add(0, Integer.toString(t.size()));
            Path tmp = expect.resolve("gen-" + g + ".tmp");
            Files.write(tmp, lines);
            Files.move(tmp, expect.resolve("gen-" + g + ".txt"), StandardCopyOption.ATOMIC_MOVE);
            say("publishing " + g);
            t.checkpoint(dir);
            say("published " + g);
        }
        t.close();
    }
}
