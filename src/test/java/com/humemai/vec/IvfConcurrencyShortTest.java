package com.humemai.vec;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** IVF index threaded program for 5 s with SQ8 (writers, readers, a retrain loop and a compaction loop). Own class: the program keeps static state. */
class IvfConcurrencyShortTest {
    @BeforeAll
    static void setUp() {
        Checks.embedded();
    }

    @Test
    void fiveSecondsSq8() throws Exception {
        int p0 = IvfT.pass, f0 = IvfT.fail;
        IvfConcurrencyTest.main(new String[] {"5", "SQ8"});
        Checks.clean("IvfConcurrencyTest 5 s SQ8", IvfT.pass - p0 + IvfT.fail - f0, IvfT.fail - f0, 9);
    }
}
