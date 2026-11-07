package com.humemai.vec;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * Record-id reuse, the slow variant at n = 3000: three configurations (L2 dimension 32, cosine dimension 128, cosine dimension 32) with three seeds each, every
 * case of {@link RidReuseProbe} including the background case, in which the worker seals and merges while one writer reuses, replaces and deletes ids and one
 * reader searches all the time. The reader counts a hit only for an id that was untouched for the whole search, so a vector deleted before a search started
 * must not appear in it.
 */
@Tag("slow")
class RidReuseLongTest {
    @ParameterizedTest(name = "{0} seed {1}")
    @CsvSource({
        "n3000d32, 11", "n3000d32, 12", "n3000d32, 13",
        "n3000d128, 11", "n3000d128, 12", "n3000d128, 13",
        "n3000d32c, 11", "n3000d32c, 12", "n3000d32c, 13"
    })
    void allCases(String cfg, long seed) {
        List<String> cases = RidReuseProbe.CASES;
        Map<String, Object> r = RidReuseProbe.runOneMap(cfg, seed, cases, true);
        RidReuseRig.assertClean(cfg + " seed " + seed, r, cases);
    }
}
