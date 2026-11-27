package ai.humem.tiervec;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.condition.DisabledOnOs;
import org.junit.jupiter.api.condition.OS;

/**
 * The tests of {@link TierHaltTest} where the open after the halt maps the segments (#27); the child writes the same files either way. Not
 * run on Windows: an open there can delete files while a file is still mapped, and that was not checked there.
 */
@DisabledOnOs(value = OS.WINDOWS, disabledReason = "these tests delete files that can still be mapped, which was not checked on Windows")
class MappedTierHaltTest extends TierHaltTest {
    @BeforeAll
    static void useTheMappedReader() {
        TestReaders.storage = TieredIndex.VectorStorage.MAPPED;
    }
}
