package ai.humem.tiervec;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.condition.DisabledOnOs;
import org.junit.jupiter.api.condition.OS;

/**
 * The tests of {@link SegmentCorruptionTest} with the mapped reader (#27). Not run on Windows: they write into files right after a load,
 * while a file can still be mapped, and that was not checked there. {@link MappedVectorsTest} runs there.
 */
@DisabledOnOs(value = OS.WINDOWS, disabledReason = "these tests change files that can still be mapped, which was not checked on Windows")
class MappedSegmentCorruptionTest extends SegmentCorruptionTest {
    @BeforeAll
    static void useTheMappedReader() {
        TestReaders.storage = TieredIndex.VectorStorage.MAPPED;
    }
}
