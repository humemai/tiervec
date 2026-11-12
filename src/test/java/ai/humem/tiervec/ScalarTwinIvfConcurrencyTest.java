package ai.humem.tiervec;

import static org.junit.jupiter.api.Assertions.assertFalse;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** IVF threaded program for 5 s with the scalar twin kernels. Runs only in the surefire execution "scalar-twin". Own class: the program keeps static state. */
@Tag("scalar")
class ScalarTwinIvfConcurrencyTest {
    @BeforeAll
    static void setUp() {
        Checks.embedded();
        assertFalse(Kernels.VECTOR, "this class needs -Dai.humem.tiervec.scalar=true, which the scalar-twin surefire execution sets");
    }

    @Test
    void fiveSecondsSq8() throws Exception {
        int p0 = IvfT.pass, f0 = IvfT.fail;
        IvfConcurrencyTest.main(new String[] {"5", "SQ8"});
        Checks.clean("IvfConcurrencyTest 5 s SQ8 (scalar twin)", IvfT.pass - p0 + IvfT.fail - f0, IvfT.fail - f0, 9);
    }
}
