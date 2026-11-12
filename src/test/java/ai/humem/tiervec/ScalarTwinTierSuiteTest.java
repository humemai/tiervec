package ai.humem.tiervec;

import static org.junit.jupiter.api.Assertions.assertFalse;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;

/** The tier program, one test at a time, with the scalar twin kernels. Runs only in the surefire execution "scalar-twin", which sets -Dai.humem.tiervec.scalar=true. The tests are the ones of TierSuiteTest. */
@Tag("scalar")
class ScalarTwinTierSuiteTest extends TierSuiteTest {
    @BeforeAll
    static void requireScalarTwin() {
        assertFalse(Kernels.VECTOR, "this class needs -Dai.humem.tiervec.scalar=true, which the scalar-twin surefire execution sets");
    }
}
