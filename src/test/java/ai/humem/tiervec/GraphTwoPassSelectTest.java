package ai.humem.tiervec;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import java.util.Arrays;
import org.junit.jupiter.api.Test;

/**
 * The neighbor selection of {@link GraphIndex} with alpha above 1 runs in two passes (idea S4 of docs/SCREEN-IDEAS.md, issue #161): the first
 * pass keeps what the rule keeps at alpha 1, and only the room that is left goes to candidates that pass at the configured alpha.
 */
class GraphTwoPassSelectTest {
    /**
     * The rows 1 to 5, in the order they are added. L2 distances are squared, so alpha is squared too (1.2 gives 1.44). Seen from the origin
     * they are at about 1, 1.10, 1.21, 1.23, and 9.
     */
    private static final float[][] ROWS = {
        {1f, 0f},           // 1: the nearest, kept by every rule
        {1.05f, 0f},        // 2: beside row 1, which drops it at both values of alpha
        {0.6309f, 0.9011f}, // 3: row 1 drops it at alpha 1 and keeps it at alpha 1.2
        {0.6372f, 0.9101f}, // 4: as row 3 against row 1, and row 3 drops it at alpha 1.2
        {-3f, 0f},          // 5: far on the other side, kept at alpha 1
    };

    /** the ids on the layer-0 list of a row at the origin that is added last, so that no later write changes its list */
    private static long[] listOfTheLastRow(int m, float alpha) {
        GraphIndex g = new GraphIndex(2, Metric.L2, GraphIndex.Params.defaults().withM(m).withAlpha(alpha));
        for (int i = 0; i < ROWS.length; i++) g.add(i + 1, ROWS[i]);
        g.add(0, new float[] {0f, 0f});
        SealedGraph s = g.seal();
        for (int r = 0; r < s.n; r++) {
            if (s.ids[r] != 0) continue;
            long[] list = new long[s.off0[r + 1] - s.off0[r]];
            for (int i = 0; i < list.length; i++) list[i] = s.ids[s.edges0[s.off0[r] + i]];
            Arrays.sort(list);
            return list;
        }
        throw new AssertionError("the last row is not in the graph");
    }

    @Test
    void aFullListHoldsTheFarRowAndNotTheNearOnesThatOnlyAlphaKeeps() {
        // one pass at alpha 1.2 takes the rows 1 and 3 and never looks at row 5
        assertArrayEquals(new long[] {1, 5}, listOfTheLastRow(2, 1.2f));
    }

    @Test
    void theRoomThatIsLeftGoesToTheRowsThatPassAtAlpha() {
        // row 3 comes in the second pass; row 4 is dropped there by row 3, and row 2 by row 1 in the first
        assertArrayEquals(new long[] {1, 3, 5}, listOfTheLastRow(5, 1.2f));
    }

    @Test
    void atAlphaOneThereIsNoSecondPass() {
        assertArrayEquals(new long[] {1, 5}, listOfTheLastRow(2, 1.0f));
        assertArrayEquals(new long[] {1, 5}, listOfTheLastRow(5, 1.0f));
    }
}
