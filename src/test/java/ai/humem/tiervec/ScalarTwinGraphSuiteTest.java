package ai.humem.tiervec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The graph kernels and the graph functional program with the scalar twin kernels. Runs only in the surefire execution "scalar-twin",
 * which sets -Dai.humem.tiervec.scalar=true (the kernel choice is made once per JVM, at class initialization of Kernels).
 */
@Tag("scalar")
class ScalarTwinGraphSuiteTest {
    @BeforeAll
    static void setUp() {
        Checks.embedded();
        assertFalse(Kernels.VECTOR, "this class needs -Dai.humem.tiervec.scalar=true, which the scalar-twin surefire execution sets");
    }

    @Test
    void kernUsesTheScalarTwinsInThisMode() {
        Random r = new Random(16);
        for (int n : new int[] {0, 1, 3, 4, 5, 33, 129, 768}) {
            float[] a = KernDiffTest.rnd(r, n), b = KernDiffTest.rnd(r, n);
            byte[] c = KernDiffTest.codes(r, n);
            String m = " n=" + n;
            assertEquals(Float.floatToIntBits(Kernels.dotScalar(a, 0, b, 0, n)), Float.floatToIntBits(Kern.dot(a, 0, b, 0, n)), "Kern.dot" + m);
            assertEquals(Float.floatToIntBits(Kernels.l2sqScalar(a, 0, b, 0, n)), Float.floatToIntBits(Kern.l2(a, 0, b, 0, n)), "Kern.l2" + m);
            assertEquals(Float.floatToIntBits(Kernels.dotF8Scalar(a, 0, c, 0, n)), Float.floatToIntBits(Kern.dotQ8(a, c, 0, n)), "Kern.dotQ8" + m);
        }
        assertEquals(0, Kern.lanes(), "no Vector API lanes in scalar mode");
        assertTrue(Kernels.scalarReason != null && Kernels.scalarReason.startsWith("forced by"), "scalar reason: " + Kernels.scalarReason);
    }

    @Test
    void graphQuick() throws Exception {
        int c0 = GraphT.checks, f0 = GraphT.failures;
        GraphTest.main(new String[] {"quick"});
        Checks.clean("GraphTest quick (scalar twin)", GraphT.checks - c0, GraphT.failures - f0, 2500);
    }
}
