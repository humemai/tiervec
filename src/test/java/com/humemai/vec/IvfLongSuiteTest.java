package com.humemai.vec;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** IVF index with the Vector API kernels at n = 20000: the single-threaded program and the drift program. */
@Tag("slow")
class IvfLongSuiteTest {
    @BeforeAll
    static void setUp() {
        Checks.embedded();
    }

    @Test
    void n20000() throws Exception {
        assertTrue(Kernels.VECTOR, "the Vector API kernels must be the ones under test (is --add-modules jdk.incubator.vector set?)");
        int p0 = IvfT.pass, f0 = IvfT.fail;
        IvfTest.main(new String[] {"20000"});
        Checks.clean("IvfTest n=20000", IvfT.pass - p0 + IvfT.fail - f0, IvfT.fail - f0, 6094);
    }

    @Test
    void drift20000() {
        int p0 = IvfT.pass, f0 = IvfT.fail;
        IvfDriftTest.main(new String[] {"20000"});
        Checks.clean("IvfDriftTest n=20000", IvfT.pass - p0 + IvfT.fail - f0, IvfT.fail - f0, 14);
    }
}
