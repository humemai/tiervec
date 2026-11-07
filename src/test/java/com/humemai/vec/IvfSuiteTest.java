package com.humemai.vec;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** IVF index with the Vector API kernels: the single-threaded program at n = 6000. */
class IvfSuiteTest {
    @BeforeAll
    static void setUp() {
        Checks.embedded();
    }

    @Test
    void n6000() throws Exception {
        assertTrue(Kernels.VECTOR, "the Vector API kernels must be the ones under test (is --add-modules jdk.incubator.vector set?)");
        int p0 = IvfT.pass, f0 = IvfT.fail;
        IvfTest.main(new String[] {"6000"});
        Checks.clean("IvfTest n=6000", IvfT.pass - p0 + IvfT.fail - f0, IvfT.fail - f0, 1948);
    }
}
