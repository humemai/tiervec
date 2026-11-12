package ai.humem.tiervec;

import static org.junit.jupiter.api.Assertions.assertFalse;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Graph tier threaded program for 5 s with the scalar twin kernels. Runs only in the surefire execution "scalar-twin". Own class: the program keeps static state. */
@Tag("scalar")
class ScalarTwinGraphConcurrencyTest {
    @BeforeAll
    static void setUp() {
        Checks.embedded();
        assertFalse(Kernels.VECTOR, "this class needs -Dai.humem.tiervec.scalar=true, which the scalar-twin surefire execution sets");
    }

    @Test
    void fiveSeconds() throws Exception {
        int c0 = GraphT.checks, f0 = GraphT.failures;
        GraphConcurrencyTest.main(new String[] {"5"});
        Checks.clean("GraphConcurrencyTest 5 s (scalar twin)", GraphT.checks - c0, GraphT.failures - f0, 12);
    }
}
