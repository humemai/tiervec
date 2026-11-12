package ai.humem.tiervec;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** The directory flush of a publish: it works where the JDK can open a directory, does nothing on Windows, and hides no other failure. */
class FsyncDirTest {
    @TempDir
    Path tmp;

    private static boolean windows() {
        return System.getProperty("os.name", "").startsWith("Windows");
    }

    @Test
    void theHelperAgreesWithTheOperatingSystem() {
        assertEquals(windows(), SegmentFormat.WINDOWS);
    }

    @Test
    void anExistingDirectoryIsFlushedOnEveryPlatform() throws IOException {
        Files.writeString(tmp.resolve("a.bin"), "x");
        assertDoesNotThrow(() -> SegmentFormat.fsyncDir(tmp));
    }

    @Test
    void aMissingDirectoryStillThrowsWhereTheDirectoryIsOpened() {
        Assumptions.assumeFalse(windows(), "Windows does not open the directory, so there is nothing to fail");
        assertThrows(IOException.class, () -> SegmentFormat.fsyncDir(tmp.resolve("missing")));
    }

    @Test
    void aMissingDirectoryIsIgnoredOnWindows() {
        Assumptions.assumeTrue(windows(), "this is what holds on Windows only");
        assertDoesNotThrow(() -> SegmentFormat.fsyncDir(tmp.resolve("missing")));
    }
}
