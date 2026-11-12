package ai.humem.tiervec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/**
 * The read-your-writes probe of {@link GraphConcurrencyTest} (issue #38) on a small index, single-threaded, so the fast job covers the
 * exact membership check and the self-search recall path of that program. At 3,000 vectors a search at ten times the default ef collects a third
 * of the index, so the tolerance here is zero. The second test is a control: a probe that cannot fail is not a test.
 */
class GraphReadYourWritesTest {
    static final int DIM = 128, N = 3000;

    @Test
    void everyInsertIsMemberAndFoundBySelfSearch() {
        GraphT.Gen gen = new GraphT.Gen(DIM, 31, 60, 0.4f);
        GraphIndex idx = new GraphIndex(DIM, Metric.COSINE, GraphIndex.Params.defaults());
        GraphConcurrencyTest.Ryw ryw = new GraphConcurrencyTest.Ryw();
        for (long id = 0; id < N; id++) {
            float[] v = gen.vec(id);
            idx.add(id, v);
            ryw.probe(idx, id, v);
        }
        System.out.println(ryw.line());
        assertEquals(N, ryw.probes.get(), "every insert was probed");
        assertEquals(0, ryw.absent.get(), "read-your-writes: id not present after insert: " + ryw.line());
        assertEquals(0, ryw.selfMiss.get(), "self-search recall at ef " + ryw.ef(idx) + ": " + ryw.line());
        assertEquals(10 * idx.params().efSearch(), ryw.ef(idx), "the self-search runs at ten times the default ef");
    }

    @Test
    void probeFailsWhenTheIdIsNotInTheIndex() {
        GraphT.Gen gen = new GraphT.Gen(DIM, 31, 60, 0.4f);
        GraphIndex idx = new GraphIndex(DIM, Metric.COSINE, GraphIndex.Params.defaults());
        for (long id = 0; id < 200; id++) idx.add(id, gen.vec(id));
        assertTrue(idx.containsForTest(5), "an inserted id is a member");
        assertFalse(idx.containsForTest(200), "an id that was never added is not a member");

        GraphConcurrencyTest.Ryw ryw = new GraphConcurrencyTest.Ryw();
        ryw.probe(idx, 200, gen.vec(200)); // never added
        assertEquals(1, ryw.absent.get(), "membership counts an id that was never added");
        assertEquals(1, ryw.selfMiss.get(), "self-search counts a vector whose id is not in the index");

        assertTrue(idx.delete(5), "delete returns true for a member");
        assertFalse(idx.containsForTest(5), "a deleted id is not a member");
        ryw.probe(idx, 5, gen.vec(5));
        assertEquals(2, ryw.absent.get(), "membership counts a deleted id");
        assertEquals(2, ryw.probes.get());
    }
}
