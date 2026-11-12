package ai.humem.tiervec;

import static org.junit.jupiter.api.Assertions.fail;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * The full probe (every case, n = 3000, cosine, dimension 32) on the unchanged copy X0 and on the mutants M1, M2 and M3 of {@link RidReuseMutantTest}, with three
 * seeds each. X0 must pass. M1 must give {@code staleHits} above 0 and a violation in case b_sealed, M2 {@code lostDeletes} above 0 and a violation in case b_sealed,
 * M3 a wrong answer and a violation in case c2_delete_in_merge, in every seed. A mutant run counts as a catch only when every merge window was reached.
 */
@Tag("slow")
class RidReuseMutantLongTest {
    static final String CFG = "n3000d32c";

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

    @ParameterizedTest(name = "{0} seed {1}")
    @CsvSource({
        "X0, 11", "X0, 12", "X0, 13",
        "M1, 11", "M1, 12", "M1, 13",
        "M2, 11", "M2, 12", "M2, 13",
        "M3, 11", "M3, 12", "M3, 13"
    })
    void fullProbe(String variant, long seed) throws Exception {
        List<String> cases = RidReuseProbe.FAST_CASES;
        String what = variant + " seed " + seed;
        switch (variant) {
            case "X0" -> RidReuseRig.assertClean(what, mutants.run(RidReuseRig.X0, CFG, seed, cases, false), cases);
            case "M1" -> {
                Map<String, Object> r = mutants.run(RidReuseRig.M1, CFG, seed, cases, false);
                RidReuseRig.assertCaughtM1(what, r, cases, false);
            }
            case "M2" -> {
                Map<String, Object> r = mutants.run(RidReuseRig.M2, CFG, seed, cases, false);
                RidReuseRig.assertCaughtM2(what, r, cases, false);
            }
            case "M3" -> {
                Map<String, Object> r = mutants.run(RidReuseRig.M3, CFG, seed, cases, false);
                RidReuseRig.assertCaughtM3(what, r, cases, false);
            }
            default -> fail("unknown variant " + variant);
        }
    }
}
