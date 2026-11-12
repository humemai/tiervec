package ai.humem.tiervec;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/** Flat tier threaded program at its documented durations: 60 s without quantization, 20 s with SQ8, 20 s with BINARY. */
@Tag("slow")
class FlatLongSuiteTest {
    @BeforeAll
    static void setUp() {
        Checks.embedded();
    }

    @ParameterizedTest(name = "{0} s {1}")
    @CsvSource({"60, NONE, 7", "20, SQ8, 6", "20, BINARY, 6"})
    void concurrency(String seconds, String quant, int minChecks) throws Exception {
        int p0 = T.pass, f0 = T.fail;
        FlatConcurrencyTest.main(new String[] {seconds, quant});
        Checks.clean("FlatConcurrencyTest " + seconds + " s " + quant, T.pass - p0 + T.fail - f0, T.fail - f0, minChecks);
    }
}
