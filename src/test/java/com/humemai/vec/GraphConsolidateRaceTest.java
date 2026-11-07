package com.humemai.vec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.SplittableRandom;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

/**
 * Regression test for issue #3, without any timing. consolidate() and seal() take a snapshot {@code hi = nextOrd.get()} and size bitsets and arrays
 * from it, then read the entry point again. A concurrent add can allocate an ordinal at or above {@code hi} and publish that node as the entry point
 * in between, so the node index falls past the end of the bitset. These tests build exactly that view without threads: a single-threaded index of N
 * nodes, then {@code nextOrd} is lowered to HI (a multiple of 64 below N, so the bitset of a pass covers 0..HI-1 and nothing more) and the entry
 * point is set to node N-1, which is what a pass sees when the node was allocated after its snapshot. The fields are private, so the test reaches
 * them by reflection; it compiles against the code before the fix and fails there with the same ArrayIndexOutOfBoundsException as the threaded program.
 */
class GraphConsolidateRaceTest {
    static final int DIM = 16, N = 200, HI = 128;

    static float[] vec(long id) {
        SplittableRandom r = new SplittableRandom(id * 0x9E3779B97F4A7C15L + 17);
        float[] v = new float[DIM];
        for (int i = 0; i < DIM; i++) v[i] = (float) (r.nextDouble() - 0.5);
        return v;
    }

    static GraphIndex build() {
        GraphIndex g = new GraphIndex(DIM, Metric.COSINE);
        for (int i = 0; i < N; i++) g.add(i, vec(i));
        assertEquals(N, g.ordinalsAllocated(), "single-threaded adds allocate the ordinals 0..N-1 in order");
        return g;
    }

    /** The view of a pass whose snapshot predates node N-1. Close it to restore the index. */
    static AutoCloseable lateEntry(GraphIndex g) throws Exception {
        Field nextOrdField = GraphIndex.class.getDeclaredField("nextOrd");
        Field entryField = GraphIndex.class.getDeclaredField("entry");
        nextOrdField.setAccessible(true);
        entryField.setAccessible(true);
        AtomicInteger nextOrd = (AtomicInteger) nextOrdField.get(g);
        GraphIndex.Entry old = (GraphIndex.Entry) entryField.get(g);
        assertNotNull(old, "a built index has an entry point");
        entryField.set(g, new GraphIndex.Entry(N - 1, old.level()));
        nextOrd.set(HI);
        return () -> {
            nextOrd.set(N);
            entryField.set(g, old);
        };
    }

    @Test
    void bitPastTheEndIsNotSet() throws Exception {
        Method bit = GraphIndex.class.getDeclaredMethod("bit", long[].class, int.class);
        bit.setAccessible(true);
        long[] b = new long[2]; // covers the indexes 0..127
        b[1] = 1L << 63; // index 127
        assertTrue((boolean) bit.invoke(null, b, 127));
        assertFalse((boolean) bit.invoke(null, b, 126));
        assertFalse((boolean) bit.invoke(null, b, 128), "the first index past the end is not set");
        assertFalse((boolean) bit.invoke(null, b, 100_000), "an index far past the end is not set");
        assertFalse((boolean) bit.invoke(null, new long[0], 0), "an empty bitset has nothing set");
    }

    @Test
    void consolidateWithTombstonesAndAnEntryPointFromAfterTheSnapshot() throws Exception {
        GraphIndex g = build();
        for (int i = 0; i < 20; i++) assertTrue(g.delete(i));
        assertEquals(20, g.tombstoneCount());
        int recycled;
        try (AutoCloseable restore = lateEntry(g)) {
            recycled = g.consolidate();
        }
        assertEquals(20, recycled, "every tombstone below the snapshot is recycled");
        assertEquals(0, g.tombstoneCount());
        assertEquals(N - 20, g.size());
        Hit[] hits = g.search(vec(100), 10, null, 200);
        assertEquals(10, hits.length);
        for (Hit h : hits) assertTrue(h.id() >= 20, "a deleted id " + h.id() + " was returned");
    }

    @Test
    void consolidateWithNothingToRemoveAndAnEntryPointFromAfterTheSnapshot() throws Exception {
        GraphIndex g = build();
        int recycled;
        try (AutoCloseable restore = lateEntry(g)) {
            recycled = g.consolidate();
        }
        assertEquals(0, recycled);
        assertEquals(N, g.size());
    }

    @Test
    void sealWithAnEntryPointFromAfterTheSnapshot() throws Exception {
        GraphIndex g = build();
        SealedGraph sg;
        try (AutoCloseable restore = lateEntry(g)) {
            sg = g.seal();
        }
        assertEquals(HI, sg.size(), "the sealed graph holds the nodes below the snapshot");
        Hit[] hits = sg.search(vec(10), 5, null);
        assertEquals(5, hits.length);
    }
}
