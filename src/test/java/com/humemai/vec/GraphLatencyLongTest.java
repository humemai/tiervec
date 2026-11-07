package com.humemai.vec;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Graph tier: queries during a background consolidate() of a 30000-vector index with 40 percent deleted never return a deleted id, and every tombstone is recycled. */
@Tag("slow")
class GraphLatencyLongTest {
    @BeforeAll
    static void setUp() {
        Checks.embedded();
    }

    @Test
    void consolidateUnderQueries() throws Exception {
        int c0 = GraphT.checks, f0 = GraphT.failures;
        GraphLatencyTest.main(new String[] {"30000"});
        Checks.clean("GraphLatencyTest 30000", GraphT.checks - c0, GraphT.failures - f0, 2);
    }
}
