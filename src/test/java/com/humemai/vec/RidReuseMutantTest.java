package com.humemai.vec;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * The probe must fail on a broken tier, or it is not a test (issue #5, mutation check). {@link RidReuseRig} compiles a copy of {@code TieredIndex.java} with one
 * single-line fault into a temporary directory, loads it in a class loader that prefers that directory, and runs the probe class of the same loader against it.
 * This needs a JDK, not a JRE; without a compiler the class is skipped with a message (and fails when the CI variable is set). The fast variant: one seed, only
 * the case that carries each fault's signature, n = 3000.
 * <ul>
 * <li>control X0, no change: passes like the real class, which proves the loading path itself</li>
 * <li>M1, an add removes the floor entry of the id instead of raising it: an older vector comes back, {@code staleHits} above 0 in case b_sealed</li>
 * <li>M2, a delete takes a version but writes no floor entry: a deleted id comes back, {@code lostDeletes} above 0 in case b_sealed</li>
 * <li>M3, the floor cleanup after a merge drops every entry: a delete made inside the running merge is lost, a violation in case c2_delete_in_merge</li>
 * </ul>
 * The full matrix on the mutants is in {@link RidReuseMutantLongTest}.
 */
class RidReuseMutantTest {
    static final String CFG = "n3000d32c";
    static final long SEED = 11;
    static final List<String> SEALED = List.of("b_sealed");
    static final List<String> IN_MERGE = List.of("c2_delete_in_merge");

    static RidReuseRig.Mutants mutants;

    @BeforeAll
    static void setUp() throws Exception {
        RidReuseRig.requireCompiler();
        mutants = new RidReuseRig.Mutants();
    }

    @AfterAll
    static void tearDown() throws Exception {
        if (mutants != null) mutants.close();
    }

    @Test
    void controlWithoutChangePasses() throws Exception {
        Map<String, Object> r = mutants.run(RidReuseRig.X0, CFG, SEED, SEALED, false);
        RidReuseRig.assertClean("X0 (unchanged copy)", r, SEALED);
    }

    @Test
    void addThatRemovesTheFloorIsCaught() throws Exception {
        Map<String, Object> r = mutants.run(RidReuseRig.M1, CFG, SEED, SEALED, false);
        RidReuseRig.assertCaughtM1("M1", r, SEALED, true);
    }

    @Test
    void deleteWithoutFloorEntryIsCaught() throws Exception {
        Map<String, Object> r = mutants.run(RidReuseRig.M2, CFG, SEED, SEALED, false);
        RidReuseRig.assertCaughtM2("M2", r, SEALED, true);
    }

    @Test
    void floorCleanupThatDropsEverythingIsCaught() throws Exception {
        Map<String, Object> r = mutants.run(RidReuseRig.M3, CFG, SEED, IN_MERGE, false);
        RidReuseRig.assertCaughtM3("M3", r, IN_MERGE, true);
    }
}
