package ai.humem.tiervec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * A kill from outside at a moment that nobody chose. A child JVM ({@link TierKillChild}) publishes generation after generation into one
 * directory; the parent ends it with {@link Process#destroyForcibly} ({@code SIGKILL} on Linux and macOS), opens the directory, and then
 * starts the next child on what is left, {@value #KILLS} times for each seed. Every other kill comes after a random delay, and the ones between
 * are aimed at a publish: the parent waits for the child to announce one of its next checkpoints and kills a few milliseconds later. After every kill the directory must open as
 * the last generation that the child reported as published or the one it was publishing, with exactly that generation's answers and files.
 *
 * <p>This is a process kill: the operating system keeps what the process wrote. It says nothing about power loss.
 */
@Tag("slow")
class TierKillLongTest {
    static final int KILLS = 60;
    /** the longest wait between the child's first line and a kill at a random moment */
    static final int MAX_DELAY_MS = 600;
    /** the longest wait between the announcement of a checkpoint and a kill aimed at it */
    static final int MAX_AIMED_DELAY_NANOS = 8_000_000;

    @TempDir
    Path dir;

    @TempDir
    Path expect;

    @ParameterizedTest(name = "seed {0}")
    @ValueSource(longs = {1L, 2L})
    void aKillAtARandomMomentLeavesThePublishedGenerationOrTheNextOne(long seed) throws Exception {
        Random r = new Random(seed);
        float[][] qs = SegmentTestSupport.vectors(TierKillChild.QUERIES, TierHaltTest.DIM, 5);
        long live = 0; // the generation that the last open found, 0 for none
        int duringCheckpoint = 0;
        for (int kill = 0; kill < KILLS; kill++) {
            List<String> out = kill % 2 == 0 ? runAndKill(0, r.nextInt(MAX_DELAY_MS) * 1_000_000L) : runAndKill(1 + r.nextInt(5), r.nextInt(MAX_AIMED_DELAY_NANOS));
            long published = live, publishing = 0;
            for (String line : out) {
                if (line.startsWith("publishing ")) publishing = Long.parseLong(line.substring(11));
                else if (line.matches("published \\d+")) published = Long.parseLong(line.substring(10));
            }
            if (publishing > published) duringCheckpoint++;
            String what = "seed " + seed + ", kill " + kill + ", published " + published + ", publishing " + publishing;
            try (TieredIndex u = TieredIndex.open(dir, TierReopenTest.cfg())) {
                Manifest m = Manifest.readLatest(dir);
                // a line may be lost to the kill after the work it reports was done, never the other way round
                assertTrue(m.generation() == published || (publishing > published && m.generation() == publishing),
                        what + ": open found generation " + m.generation());
                List<String> want = new ArrayList<>(Files.readAllLines(expect.resolve("gen-" + m.generation() + ".txt")));
                assertEquals(Integer.parseInt(want.remove(0)), u.size(), what);
                assertNull(u.openFallbackCause(), what);
                TierHaltTest.assertSameAnswers(want, TierHaltTest.digest(u, qs), what);
                assertEquals(expected(m), names(dir), what + ": open left exactly the files of the live generation");
                live = m.generation();
            } catch (NoSuchFileException e) {
                assertEquals(0, published, what + ": " + e);
                assertEquals(0, live, what + ": " + e);
            }
        }
        assertTrue(live >= KILLS / 6, "the children made progress: generation " + live);
        assertTrue(duringCheckpoint > 0, "no kill fell inside a checkpoint");
        System.out.println("KILL seed=" + seed + " kills=" + KILLS + " duringCheckpoint=" + duringCheckpoint + " lastGeneration=" + live);
    }

    /**
     * starts a child, waits for its first line and then for its {@code checkpoints}-th announcement of a checkpoint (0: for no announcement),
     * kills it {@code delayNanos} later and returns the complete lines it printed
     */
    private List<String> runAndKill(int checkpoints, long delayNanos) throws Exception {
        Path log = expect.resolve("child.log");
        List<String> cmd = new ArrayList<>();
        cmd.add(Path.of(System.getProperty("java.home"), "bin", "java").toString());
        cmd.add("-Xmx512m");
        cmd.add("--add-modules");
        cmd.add("jdk.incubator.vector");
        cmd.add("-cp");
        cmd.add(System.getProperty("java.class.path"));
        cmd.add(TierKillChild.class.getName());
        cmd.add(dir.toString());
        cmd.add(expect.toString());
        Process p = new ProcessBuilder(cmd).redirectErrorStream(true).redirectOutput(log.toFile()).start();
        try {
            long deadline = System.nanoTime() + 60_000_000_000L;
            while (!reached(Files.readString(log), checkpoints)) {
                if (!p.isAlive()) fail("the child ended before it was ready: " + Files.readString(log));
                if (System.nanoTime() > deadline) fail("the child did not get far enough within 60 s: " + Files.readString(log));
                Thread.sleep(1);
            }
            TimeUnit.NANOSECONDS.sleep(delayNanos);
            assertTrue(p.isAlive(), "the child ended by itself: " + Files.readString(log));
        } finally {
            p.destroyForcibly();
            assertTrue(p.waitFor(60, TimeUnit.SECONDS), "the killed child did not end");
        }
        String text = Files.readString(log);
        List<String> lines = new ArrayList<>(text.lines().toList());
        if (!text.endsWith("\n") && !lines.isEmpty()) lines.remove(lines.size() - 1); // half a line
        assertTrue(lines.stream().noneMatch(l -> l.contains("Exception")), "the child failed: " + text);
        return lines;
    }

    private static boolean reached(String log, int checkpoints) {
        return log.contains("ready") && log.split("publishing ", -1).length - 1 >= checkpoints;
    }

    private static Set<String> expected(Manifest m) {
        Set<String> s = new TreeSet<>();
        s.add(Manifest.manifestName(m.generation()));
        s.add(Manifest.tombName(m.tombGeneration()));
        for (Manifest.Entry e : m.entries()) s.add(Manifest.segName(e.segId()));
        return s;
    }

    private static Set<String> names(Path d) throws IOException {
        try (Stream<Path> s = Files.list(d)) {
            return s.map(p -> p.getFileName().toString()).filter(n -> !n.equals(DirectoryLock.FILE)).collect(Collectors.toCollection(TreeSet::new));
        }
    }
}
