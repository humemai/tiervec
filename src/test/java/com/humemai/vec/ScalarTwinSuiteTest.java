package com.humemai.vec;

import static org.junit.jupiter.api.Assertions.assertFalse;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The same flat and IVF programs with the scalar twin kernels. Runs only in the surefire execution "scalar-twin", which sets
 * -Dhumemai.vec.scalar=true (the kernel choice is made once per JVM, at class initialization of Kernels).
 */
@Tag("scalar")
class ScalarTwinSuiteTest {
    @BeforeAll
    static void setUp() {
        Checks.embedded();
        assertFalse(Kernels.VECTOR, "this class needs -Dhumemai.vec.scalar=true, which the scalar-twin surefire execution sets");
    }

    @Test
    void flatCorrectness() {
        int p0 = T.pass, f0 = T.fail;
        FlatCorrectnessTest.main(new String[0]);
        Checks.clean("FlatCorrectnessTest (scalar twin)", T.pass - p0 + T.fail - f0, T.fail - f0, 10057);
    }

    @Test
    void ivfN6000() throws Exception {
        int p0 = IvfT.pass, f0 = IvfT.fail;
        IvfTest.main(new String[] {"6000"});
        Checks.clean("IvfTest n=6000 (scalar twin)", IvfT.pass - p0 + IvfT.fail - f0, IvfT.fail - f0, 1948);
    }
}
