package ai.humem.tiervec;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.condition.DisabledOnOs;
import org.junit.jupiter.api.condition.OS;

/**
 * The tests of {@link SegmentCorruptionTest} with the mapped reader (#27). Not on Windows, which does not delete a mapped file, so the
 * temporary directory of a test could not be removed; {@link MappedVectorsTest} runs there.
 */
@DisabledOnOs(value = OS.WINDOWS, disabledReason = "Windows does not delete a mapped file, so the temporary directory cannot be removed")
class MappedSegmentCorruptionTest extends SegmentCorruptionTest {
    @BeforeAll
    static void useTheMappedReader() {
        TestReaders.storage = TieredIndex.VectorStorage.MAPPED;
    }
}
