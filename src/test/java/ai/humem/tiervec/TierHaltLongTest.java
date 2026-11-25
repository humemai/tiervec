package ai.humem.tiervec;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * {@link TierHaltTest} in full: a child JVM halts itself at every publish step of each of its three checkpoints, with {@value #SEEDS} seeds
 * (other vectors and queries) for every pair, and the parent opens and compares. Every case must open the generation before the cut one or
 * the cut one with the child's answers; there is no tolerance for a failed open.
 */
@Tag("slow")
class TierHaltLongTest {
    static final int SEEDS = 8;

    @TempDir
    Path dir;

    @TempDir
    Path expect;

    static Stream<Arguments> points() {
        List<Arguments> l = new ArrayList<>();
        for (FaultHook.Step s : FaultHook.Step.values()) {
            for (int gen = 1; gen <= TierHaltTest.GENS; gen++) {
                for (int k = 0; k < SEEDS; k++) l.add(Arguments.of(s, gen, 20_000L + 1000L * s.ordinal() + 100L * gen + 2L * k));
            }
        }
        return l.stream();
    }

    @ParameterizedTest(name = "halt after {0} in checkpoint {1}, seed {2}")
    @MethodSource("points")
    void aHaltAtAnyStepOfAnyCheckpointLeavesTheGenerationBeforeOrTheOneBeingPublished(FaultHook.Step step, int gen, long seed) throws Exception {
        TierHaltTest.haltAndCheck(dir, expect, step, gen, seed);
    }
}
