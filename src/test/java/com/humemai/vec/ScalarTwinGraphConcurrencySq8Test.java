package com.humemai.vec;

import static org.junit.jupiter.api.Assertions.assertFalse;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Graph tier threaded program for 5 s, SQ8 codes, with the scalar twin kernels. Runs only in the surefire execution "scalar-twin". Own class: the program keeps static state. */
@Tag("scalar")
class ScalarTwinGraphConcurrencySq8Test {
    @BeforeAll
    static void setUp() {
        Checks.embedded();
        assertFalse(Kernels.VECTOR, "this class needs -Dhumemai.vec.scalar=true, which the scalar-twin surefire execution sets");
    }

    @Test
    void fiveSecondsSq8() throws Exception {
        int c0 = GraphT.checks, f0 = GraphT.failures;
        GraphConcurrencyTest.main(new String[] {"5", "sq8"});
        Checks.clean("GraphConcurrencyTest 5 s sq8 (scalar twin)", GraphT.checks - c0, GraphT.failures - f0, 12);
    }
}
