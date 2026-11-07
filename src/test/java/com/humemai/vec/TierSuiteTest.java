package com.humemai.vec;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * TieredIndex: every test of the TierTests program except the 10 s threaded one (that is TierConcurrentLongTest), one at a time.
 * The program selects a test by a substring of its name; each name below matches exactly one test.
 */
class TierSuiteTest {
    @BeforeAll
    static void setUp() {
        Checks.embedded();
    }

    @ParameterizedTest(name = "{0}")
    @CsvSource({
        "tailOnly, 14", "noGraphBelowSmall, 7", "segmentsRecall, 6", "metrics, 6", "swapInstantHooks, 7", "sealStress, 6",
        "merge, 24", "deleteReplace, 18", "rebuild, 10", "close, 2", "agentRate, 8"
    })
    void quick(String name, int minChecks) {
        int p0 = TierT.pass, f0 = TierT.fail;
        TierTests.main(new String[] {"quick", name});
        Checks.clean("TierTests " + name, TierT.pass - p0 + TierT.fail - f0, TierT.fail - f0, minChecks);
    }
}
