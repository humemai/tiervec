package com.humemai.vec;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * Record-id reuse and tombstone survival on a {@link TieredIndex} (issue #5), the fast variant: n = 3000, dimension 32 (L2) and 128 (cosine), two seeds each.
 * Each run goes through every deterministic case of {@link RidReuseProbe}: the old copy of a reused id in the active tail, in the tail and one segment, in sealed
 * segments, in a segment that a merge is rebuilding (re-add, delete, delete then re-add, delete then re-add then delete, replace, new ids, all inside the running
 * merge), after the merge, after four generations of the same ids moved the version floor up four times, and inside the publish window of a seal (before the
 * swap, with the frozen tail holding every old copy, and right after it). Every search runs plain, with the id alone allowed, and under
 * {@code Freshness.SEALED_ONLY}, with the id's current vector and with an older one as the query, and {@code size()} and {@code stats().live()} are compared with
 * a model next to the id checks. A window that is not reached fails the run. The larger matrix and the background case are in {@link RidReuseLongTest} and
 * {@link RidReuseBigLongTest}, the check that the probe fails on a broken tier is in {@link RidReuseMutantTest}.
 */
class RidReuseTest {
    @ParameterizedTest(name = "{0} seed {1}")
    @CsvSource({"n3000d32, 11", "n3000d32, 12", "n3000d128, 11", "n3000d128, 12"})
    void reuseAndSurvival(String cfg, long seed) {
        List<String> cases = RidReuseProbe.FAST_CASES;
        Map<String, Object> r = RidReuseProbe.runOneMap(cfg, seed, cases);
        RidReuseRig.assertClean(cfg + " seed " + seed, r, cases);
    }
}
