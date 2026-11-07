package com.humemai.vec;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** TieredIndex: 8 writers and 8 readers for 10 s with seals and merges running (about 14 s wall time with the checks). */
@Tag("slow")
class TierConcurrentLongTest {
    @BeforeAll
    static void setUp() {
        Checks.embedded();
    }

    @Test
    void concurrent() {
        int p0 = TierT.pass, f0 = TierT.fail;
        TierTests.main(new String[] {"quick", "concurrent"});
        Checks.clean("TierTests concurrent", TierT.pass - p0 + TierT.fail - f0, TierT.fail - f0, 11);
    }
}
