package com.humemai.vec;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Graph tier threaded program for 5 s: 8 writers, 8 readers and a background consolidate(). Own class: the program keeps static state. */
class GraphConcurrencyShortTest {
    @BeforeAll
    static void setUp() {
        Checks.embedded();
    }

    @Test
    void fiveSeconds() throws Exception {
        int c0 = GraphT.checks, f0 = GraphT.failures;
        GraphConcurrencyTest.main(new String[] {"5"});
        Checks.clean("GraphConcurrencyTest 5 s", GraphT.checks - c0, GraphT.failures - f0, 12);
    }
}
