package ai.humem.tiervec;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Graph tier: the functional program in its quick mode (kernels, read-your-writes, replace, metrics, filter, delete and consolidate, seal and fromSealed). */
class GraphSuiteTest {
    @BeforeAll
    static void setUp() {
        Checks.embedded();
    }

    @Test
    void quick() throws Exception {
        int c0 = GraphT.checks, f0 = GraphT.failures;
        GraphTest.main(new String[] {"quick"});
        Checks.clean("GraphTest quick", GraphT.checks - c0, GraphT.failures - f0, 2500);
    }
}
