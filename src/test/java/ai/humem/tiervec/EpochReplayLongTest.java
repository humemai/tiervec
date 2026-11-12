package ai.humem.tiervec;

import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** The randomized runs of EpochReplayStressTest, longer and with more seeds (issue #7). */
@Tag("slow")
class EpochReplayLongTest {
    @TempDir
    Path dir;

    @ParameterizedTest
    @ValueSource(longs = {11, 12, 13, 14})
    void crashAndReopenLoop(long seed) throws Exception {
        EpochReplayStressTest.crashLoop(seed, 250, dir);
    }

    @ParameterizedTest
    @ValueSource(longs = {21, 22, 23, 24})
    void aRebuildNeverLosesAWriteThatArrivesDuringIt(long seed) throws Exception {
        EpochReplayStressTest.rebuildRace(seed, 30, 1_000_000L);
    }
}
