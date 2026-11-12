package ai.humem.tiervec;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Graph tier threaded program for 60 s, fp32 vectors. Own class: the program keeps static state. */
@Tag("slow")
class GraphConcurrencyLongTest {
    @BeforeAll
    static void setUp() {
        Checks.embedded();
    }

    @Test
    void sixtySeconds() throws Exception {
        int c0 = GraphT.checks, f0 = GraphT.failures;
        GraphConcurrencyTest.main(new String[] {"60"});
        Checks.clean("GraphConcurrencyTest 60 s", GraphT.checks - c0, GraphT.failures - f0, 12);
    }
}
