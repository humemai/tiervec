package ai.humem.tiervec;

import static ai.humem.tiervec.SegmentTestSupport.vectors;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * A real process kill during a checkpoint. A child JVM ({@link TierHaltChild}) builds a tier batch by batch and checkpoints after each batch;
 * in the checkpoint and at the step that this test names it calls {@link Runtime#halt}, which ends the process at once with no shutdown hook and
 * no cleanup. The parent then opens the directory and compares what it serves with what the child's in-memory index served for that generation,
 * which the child wrote to a file before the checkpoint began.
 *
 * <p>The directory must hold exactly the generation before the cut one, or the cut one when the halt came at or after the rename (and when the
 * first checkpoint is cut before its rename there is no generation, and open says so with a {@link NoSuchFileException}). Whichever it is, open
 * must serve the answers of that generation, leave exactly its files, and not throw. Every step of the publish is cut once, in the first, second
 * or third checkpoint, each with its own seed; the run takes a few seconds because every case is one JVM. {@link TierHaltLongTest} (slow) runs
 * every step in every checkpoint with many seeds.
 *
 * <p>This is a process kill: the operating system keeps what the process wrote, flushed or not. It says nothing about power loss.
 */
class TierHaltTest {
    static final int DIM = TierReopenTest.DIM;
    static final int ROWS = 250;
    static final int GENS = 3;
    static final int QUERIES = 30;
    static final int HALT_CODE = 42;

    /** where the index is written (the child's checkpoints) and where the child puts what each generation must serve, and its log */
    @TempDir
    Path dir;

    @TempDir
    Path expect;

    static Stream<Arguments> points() {
        List<Arguments> l = new ArrayList<>();
        // the checkpoint that is cut: the first one is cut before and after its rename (no generation, then generation 1), the later ones between two generations
        final int[] checkpointOf = {1, 2, 3, 2, 1, 3};
        assertEquals(FaultHook.Step.values().length, checkpointOf.length, "one checkpoint number for every step");
        for (FaultHook.Step s : FaultHook.Step.values()) l.add(Arguments.of(s, checkpointOf[s.ordinal()], 1000L + s.ordinal()));
        return l.stream();
    }

    /** one line per query: the hits as {@code id:bits of the score}, best first */
    static List<String> digest(TieredIndex t, float[][] qs) {
        List<String> lines = new ArrayList<>();
        for (float[] q : qs) {
            StringBuilder sb = new StringBuilder();
            for (Hit h : t.search(q, 10, null)) {
                if (sb.length() > 0) sb.append(' ');
                sb.append(h.id()).append(':').append(Float.floatToIntBits(h.score()));
            }
            lines.add(sb.toString());
        }
        return lines;
    }

    /** the same hits in the same order, with scores that agree to a few units in the last place (the two sides ran in two JVMs) */
    static void assertSameAnswers(List<String> want, List<String> got, String what) {
        assertEquals(want.size(), got.size(), what + ": the number of queries");
        for (int i = 0; i < want.size(); i++) {
            String[] a = want.get(i).split(" ");
            String[] b = got.get(i).split(" ");
            assertEquals(a.length, b.length, what + ", query " + i + ": the number of hits");
            assertTrue(a.length > 0 && !a[0].isEmpty(), what + ", query " + i + ": the child's index returned hits");
            for (int j = 0; j < a.length; j++) {
                String[] x = a[j].split(":");
                String[] y = b[j].split(":");
                assertEquals(x[0], y[0], what + ", query " + i + ", hit " + j + ": the id");
                float fx = Float.intBitsToFloat(Integer.parseInt(x[1]));
                float fy = Float.intBitsToFloat(Integer.parseInt(y[1]));
                assertEquals(fx, fy, Math.abs(fx) * 1e-5f + 1e-6f, what + ", query " + i + ", hit " + j + ": the score");
            }
        }
    }

    private static Set<String> expectedNames(Manifest m) {
        Set<String> s = new TreeSet<>();
        s.add(Manifest.manifestName(m.generation()));
        s.add(Manifest.tombName(m.tombGeneration()));
        for (Manifest.Entry e : m.entries()) s.add(Manifest.segName(e.segId()));
        return s;
    }

    private static Set<String> names(Path d) throws IOException {
        try (Stream<Path> s = Files.list(d)) {
            return s.map(p -> p.getFileName().toString()).collect(Collectors.toCollection(TreeSet::new));
        }
    }

    /** starts the child and waits for it to halt; fails with the child's output when it ended any other way */
    private static void runChild(Path dir, Path expect, FaultHook.Step step, int gen, long seed) throws Exception {
        Path log = expect.resolve("child.log");
        List<String> cmd = new ArrayList<>();
        cmd.add(Path.of(System.getProperty("java.home"), "bin", "java").toString());
        cmd.add("-Xmx512m");
        cmd.add("--add-modules");
        cmd.add("jdk.incubator.vector");
        cmd.add("-cp");
        cmd.add(System.getProperty("java.class.path"));
        cmd.add(TierHaltChild.class.getName());
        cmd.add(dir.toString());
        cmd.add(expect.toString());
        cmd.add(Integer.toString(gen));
        cmd.add(step.name());
        cmd.add(Long.toString(seed));
        Process p = new ProcessBuilder(cmd).redirectErrorStream(true).redirectOutput(log.toFile()).start();
        if (!p.waitFor(120, TimeUnit.SECONDS)) {
            p.destroyForcibly();
            fail("the child did not halt within 120 s: " + Files.readString(log));
        }
        assertEquals(HALT_CODE, p.exitValue(), "the child halts itself at " + step + " of checkpoint " + gen + ", its output: " + Files.readString(log));
    }

    @ParameterizedTest(name = "halt after {0} in checkpoint {1}, seed {2}")
    @MethodSource("points")
    void aHaltAtAnyStepOfACheckpointLeavesTheGenerationBeforeOrTheOneBeingPublished(FaultHook.Step step, int gen, long seed) throws Exception {
        haltAndCheck(dir, expect, step, gen, seed);
    }

    /** one case: a child that publishes into {@code dir} and halts at {@code step} of checkpoint {@code gen}, then the open and the comparison */
    static void haltAndCheck(Path dir, Path expect, FaultHook.Step step, int gen, long seed) throws Exception {
        runChild(dir, expect, step, gen, seed);
        final boolean renamed = step.ordinal() >= FaultHook.Step.MANIFEST_RENAMED.ordinal();
        final int live = renamed ? gen : gen - 1;
        final String what = "halt after " + step + " in checkpoint " + gen;

        if (live == 0) {
            // the first checkpoint stopped before its rename: files of it are there, and no manifest names them
            assertNull(Manifest.readLatest(dir), what);
            assertThrows(NoSuchFileException.class, () -> TieredIndex.open(dir, TierReopenTest.cfg()), what);
            return;
        }

        Manifest latest = Manifest.readLatest(dir);
        assertNotNull(latest, what);
        assertEquals(live, latest.generation(), what);
        List<String> want = Files.readAllLines(expect.resolve("gen-" + live + ".txt"));
        try (TieredIndex u = TieredIndex.open(dir, TierReopenTest.cfg())) {
            assertNull(u.openFallbackCause(), what);
            assertEquals(live * ROWS, u.size(), what);
            assertSameAnswers(want, digest(u, vectors(QUERIES, DIM, seed + 1)), what); // the child's queries: its seed plus one
        }
        assertEquals(expectedNames(latest), names(dir), what + ": open left exactly the files of the live generation");
    }
}
