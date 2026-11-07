package com.humemai.vec;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * Record-id reuse, the slow variant at n = 20000: cosine with dimension 32 and L2 with dimension 128, one seed each, every case of {@link RidReuseProbe}. At this
 * size a segment holds thousands of rows, so the allow-one search ends in an exact scan of a large segment and a merge window is long enough for every operation
 * group. The other seeds of the pre-planned matrix are left out to keep the slow job well inside its time limit.
 */
@Tag("slow")
class RidReuseBigLongTest {
    @ParameterizedTest(name = "{0} seed {1}")
    @CsvSource({"n20000d32, 21", "n20000d128, 22"})
    void allCases(String cfg, long seed) {
        List<String> cases = RidReuseProbe.CASES;
        Map<String, Object> r = RidReuseProbe.runOneMap(cfg, seed, cases, true);
        RidReuseRig.assertClean(cfg + " seed " + seed, r, cases);
    }
}
