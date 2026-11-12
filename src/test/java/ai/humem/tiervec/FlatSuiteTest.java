package ai.humem.tiervec;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/** Flat tier with the Vector API kernels: the correctness program, and the threaded program for 5 s per quantizer. */
class FlatSuiteTest {
    @BeforeAll
    static void setUp() {
        Checks.embedded();
    }

    @Test
    void correctness() {
        assertTrue(Kernels.VECTOR, "the Vector API kernels must be the ones under test (is --add-modules jdk.incubator.vector set?)");
        int p0 = T.pass, f0 = T.fail;
        FlatCorrectnessTest.main(new String[0]);
        Checks.clean("FlatCorrectnessTest", T.pass - p0 + T.fail - f0, T.fail - f0, 10057);
    }

    @ParameterizedTest(name = "concurrency {0}")
    @CsvSource({"NONE, 7", "SQ8, 6", "BINARY, 6"})
    void concurrency(String quant, int minChecks) throws Exception {
        int p0 = T.pass, f0 = T.fail;
        FlatConcurrencyTest.main(new String[] {"5", quant});
        Checks.clean("FlatConcurrencyTest " + quant, T.pass - p0 + T.fail - f0, T.fail - f0, minChecks);
    }
}
