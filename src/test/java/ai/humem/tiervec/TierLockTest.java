package ai.humem.tiervec;

import static ai.humem.tiervec.SegmentTestSupport.idOf;
import static ai.humem.tiervec.SegmentTestSupport.vectors;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** One process owns an index directory (issue #27): a second process that opens it or writes into it fails with an error that names it. */
class TierLockTest {
    static final int LOCKED_CODE = 43;
    static final String READY = "holding";
    private static final int ROWS = 100;

    @TempDir
    Path dir;

    private final List<TieredIndex> opened = new ArrayList<>();

    @AfterEach
    void closeAll() {
        for (TieredIndex t : opened) t.close();
    }

    private TieredIndex track(TieredIndex t) {
        opened.add(t);
        return t;
    }

    /** a settled tier of {@link #ROWS} rows; the caller closes it */
    static TieredIndex small() {
        TieredIndex t = new TieredIndex(TierReopenTest.DIM, Metric.L2, TierReopenTest.cfg());
        float[][] data = vectors(ROWS, TierReopenTest.DIM, 3);
        for (int i = 0; i < ROWS; i++) t.add(idOf(i), data[i]);
        t.sealNow();
        return t;
    }

    private static ProcessBuilder child(String mode, Path index) {
        List<String> cmd = new ArrayList<>();
        cmd.add(Path.of(System.getProperty("java.home"), "bin", "java").toString());
        cmd.add("-Xmx256m");
        cmd.add("--add-modules");
        cmd.add("jdk.incubator.vector");
        cmd.add("-cp");
        cmd.add(System.getProperty("java.class.path"));
        cmd.add(TierLockChild.class.getName());
        cmd.add(mode);
        cmd.add(index.toString());
        return new ProcessBuilder(cmd).redirectErrorStream(true);
    }

    /** runs a child to its end and checks how it ended */
    private void run(String mode, Path index, int wantCode) throws Exception {
        Path log = Files.createTempFile("lock-child", ".log");
        try {
            Process p = child(mode, index).redirectOutput(log.toFile()).start();
            if (!p.waitFor(120, TimeUnit.SECONDS)) {
                p.destroyForcibly();
                fail("the child did not end within 120 s: " + Files.readString(log));
            }
            String said = Files.readString(log);
            assertEquals(wantCode, p.exitValue(), mode + ": " + said);
            if (wantCode == LOCKED_CODE) assertTrue(said.contains(index.getFileName().toString()), "the error names the directory: " + said);
        } finally {
            Files.delete(log);
        }
    }

    @Test
    void anotherProcessIsRefusedWhileATierHoldsTheDirectory() throws Exception {
        Path index = dir.resolve("index");
        TieredIndex t = track(small());
        t.checkpoint(index);
        run("open", index, LOCKED_CODE);
        run("checkpoint", index, LOCKED_CODE);
        assertEquals(1, Manifest.readLatest(index).generation(), "the refused checkpoint wrote nothing");
        t.checkpoint(index); // the owner goes on
        t.close();
        run("open", index, 0);
        run("checkpoint", index, 0);
    }

    @Test
    void theTiersOfOneProcessShareTheDirectoryAndTheLastOneReleasesIt() throws Exception {
        Path index = dir.resolve("index");
        TieredIndex first = track(small());
        first.checkpoint(index);
        TieredIndex second = track(TieredIndex.open(index, TierReopenTest.cfg()));
        first.close();
        first.close(); // a second close releases nothing more
        run("open", index, LOCKED_CODE);
        second.close();
        run("open", index, 0);
    }

    @Test
    void aCheckpointIntoAnotherDirectoryReleasesTheFirst() throws Exception {
        Path one = dir.resolve("one");
        Path two = dir.resolve("two");
        TieredIndex t = track(small());
        t.checkpoint(one);
        t.checkpoint(two);
        run("open", one, 0);
        run("open", two, LOCKED_CODE);
    }

    @Test
    void aFailedOpenAndAFailedCheckpointReleaseTheDirectory() throws Exception {
        Path empty = Files.createDirectory(dir.resolve("empty"));
        assertThrows(NoSuchFileException.class, () -> TieredIndex.open(empty, TierReopenTest.cfg()));
        run("checkpoint", empty, 0);

        Path cut = dir.resolve("cut");
        TieredIndex t = track(small());
        assertThrows(IOException.class, () -> t.checkpoint(cut, step -> {
            throw new IOException("crash after " + step);
        }));
        run("checkpoint", cut, 0);
    }

    @Test
    void anOpenThatRebuildsKeepsTheDirectory() throws Exception {
        Path index = Files.createDirectory(dir.resolve("index"));
        TieredIndex t = track(TieredIndex.open(index, TierReopenTest.DIM, Metric.L2, TierReopenTest.cfg(), List::of));
        assertNotNull(t.openFallbackCause());
        run("checkpoint", index, LOCKED_CODE);
        t.close();
        run("checkpoint", index, 0);
    }

    @Test
    void thisProcessIsRefusedWhileAnotherOneHoldsTheDirectory() throws Exception {
        Path index = dir.resolve("index");
        try (TieredIndex t = small()) {
            t.checkpoint(index);
        }
        Process p = child("hold", index).start();
        try {
            BufferedReader said = new BufferedReader(new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8));
            String line = CompletableFuture.supplyAsync(() -> {
                try {
                    return said.readLine();
                } catch (IOException e) {
                    throw new IllegalStateException(e);
                }
            }).get(120, TimeUnit.SECONDS);
            assertEquals(READY + " " + ROWS, line);

            DirectoryLockedException e = assertThrows(DirectoryLockedException.class, () -> TieredIndex.open(index, TierReopenTest.cfg()));
            assertEquals(index.toString(), e.directory());
            assertThrows(DirectoryLockedException.class,
                    () -> TieredIndex.open(index, TierReopenTest.DIM, Metric.L2, TierReopenTest.cfg(), List::of));
            FakeHost h = new FakeHost(TierReopenTest.DIM, 4, 7);
            assertThrows(DirectoryLockedException.class, () -> TieredIndex.open(index, TierReopenTest.DIM, Metric.L2, TierReopenTest.cfg(), h));
            TieredIndex other = track(small());
            assertThrows(DirectoryLockedException.class, () -> other.checkpoint(index));

            p.getOutputStream().close(); // the child closes its tier and ends
            assertTrue(p.waitFor(120, TimeUnit.SECONDS));
            assertEquals(0, p.exitValue());
            other.checkpoint(index);
            assertEquals(2, Manifest.readLatest(index).generation());
        } finally {
            p.destroyForcibly();
            p.waitFor(30, TimeUnit.SECONDS);
        }
    }
}
