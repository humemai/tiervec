package com.humemai.vec;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** IVF index threaded program for 10 s with SQ8. Own class: the program keeps static state. */
@Tag("slow")
class IvfConcurrencyLongTest {
    @BeforeAll
    static void setUp() {
        Checks.embedded();
    }

    @Test
    void tenSecondsSq8() throws Exception {
        int p0 = IvfT.pass, f0 = IvfT.fail;
        IvfConcurrencyTest.main(new String[] {"10", "SQ8"});
        Checks.clean("IvfConcurrencyTest 10 s SQ8", IvfT.pass - p0 + IvfT.fail - f0, IvfT.fail - f0, 9);
    }
}
