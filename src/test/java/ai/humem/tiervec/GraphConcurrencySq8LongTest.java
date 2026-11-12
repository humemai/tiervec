package ai.humem.tiervec;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Graph tier threaded program for 60 s, SQ8 codes. Own class: the program keeps static state. */
@Tag("slow")
class GraphConcurrencySq8LongTest {
    @BeforeAll
    static void setUp() {
        Checks.embedded();
    }

    @Test
    void sixtySecondsSq8() throws Exception {
        int c0 = GraphT.checks, f0 = GraphT.failures;
        GraphConcurrencyTest.main(new String[] {"60", "sq8"});
        Checks.clean("GraphConcurrencyTest 60 s sq8", GraphT.checks - c0, GraphT.failures - f0, 12);
    }
}
