package ai.humem.tiervec;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.condition.DisabledOnOs;
import org.junit.jupiter.api.condition.OS;

/**
 * The tests of {@link TierCorruptionTest} with tiers that open their segments mapped (#27). Not on Windows, which does not delete a mapped
 * file, so these tests could not empty their directory between two cases; {@link MappedVectorsTest} runs there.
 */
@DisabledOnOs(value = OS.WINDOWS, disabledReason = "Windows does not delete a mapped file, and these tests delete segment files right after an open")
class MappedTierCorruptionTest extends TierCorruptionTest {
    @BeforeAll
    static void useTheMappedReader() {
        TestReaders.storage = TieredIndex.VectorStorage.MAPPED;
    }
}
