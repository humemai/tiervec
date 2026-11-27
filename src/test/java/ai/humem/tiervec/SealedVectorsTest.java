package ai.humem.tiervec;

import static ai.humem.tiervec.SegmentTestSupport.graph;
import static ai.humem.tiervec.SegmentTestSupport.params;
import static ai.humem.tiervec.SegmentTestSupport.vectors;
import static ai.humem.tiervec.SegmentTestSupport.vectorsOf;
import static ai.humem.tiervec.SegmentTestSupport.versOf;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.function.LongPredicate;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * A sealed graph whose vectors are not a heap array searches, writes, and re-inflates like the same graph on the heap (#27). The stand-in
 * keeps the rows in a direct little-endian buffer and copies a row for every distance. The graph has no vector array of its own, so what
 * this checks is that {@link SealedVectors} is enough for every reader of a sealed segment's vectors.
 */
class SealedVectorsTest {
    @TempDir
    Path tmp;

    private static final class OffHeap implements SealedVectors {
        private final FloatBuffer b;
        private final int dim;

        OffHeap(float[] a, int dim) {
            this.b = ByteBuffer.allocateDirect(4 * a.length).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            this.b.put(0, a);
            this.dim = dim;
        }

        @Override
        public float l2(float[] q, int row) {
            return Kern.l2(q, 0, row(row), 0, dim);
        }

        @Override
        public float dot(float[] q, int row) {
            return Kern.dot(q, 0, row(row), 0, dim);
        }

        @Override
        public void copy(int row, float[] dst, int dstOff) {
            b.get(row * dim, dst, dstOff, dim);
        }

        @Override
        public float[] row(int row) {
            float[] r = new float[dim];
            copy(row, r, 0);
            return r;
        }

        @Override
        public int length() {
            return b.capacity();
        }

        @Override
        public void put(FloatBuffer dst, int from, int count) {
            dst.put(b.slice(from, count));
        }

        @Override
        public long heapBytes() {
            return 0;
        }
    }

    // dimension 512 makes the VEC section larger than the writer's 1 MiB chunk
    @ParameterizedTest(name = "{0} sq8={1} dim={2}")
    @CsvSource({"L2,false,24", "COSINE,false,24", "DOT,false,24", "L2,true,24", "COSINE,true,24", "L2,false,512"})
    void vectorsOffTheHeapBehaveLikeTheHeapArray(Metric metric, boolean sq8, int dim) throws IOException {
        final int n = 600;
        SealedGraph heap = graph(vectors(n, dim, 31), metric, params(sq8));
        SealedGraph off = new SealedGraph(heap.dim, heap.metric, heap.params, heap.n, new OffHeap(vectorsOf(heap), dim), heap.codes,
                heap.qbase, heap.qdelta, heap.qnorm, heap.ids, heap.level, heap.off0, heap.edges0, heap.upIdx, heap.upOff, heap.upEdges,
                heap.entryRow, heap.maxLevel, heap.sortedRows);

        // search: the walk, the fp32 rescoring of an SQ8 walk, and the exact scan of a filter
        LongPredicate few = id -> id % 7 == 0;
        for (float[] q : vectors(40, dim, 32)) {
            assertArrayEquals(heap.search(q, 10, null), off.search(q, 10, null));
            assertArrayEquals(heap.search(q, 10, few), off.search(q, 10, few));
            assertArrayEquals(heap.scan(q, 10, few, null), off.scan(q, 10, few, null));
        }
        assertArrayEquals(heap.rawVector(n - 1), off.rawVector(n - 1));
        assertEquals(heap.bytesHeld() - 4L * n * dim, off.bytesHeld());

        // the writer: the same file byte for byte, and it reads back with the vectors it was given
        Path a = tmp.resolve("heap.tvs");
        Path b = tmp.resolve("off.tvs");
        SegmentWriter.write(heap, versOf(n), a);
        SegmentWriter.write(off, versOf(n), b);
        assertArrayEquals(Files.readAllBytes(a), Files.readAllBytes(b));
        assertArrayEquals(vectorsOf(heap), vectorsOf(SealedGraph.load(b)));

        // re-inflation to a mutable graph keeps the vectors
        assertArrayEquals(vectorsOf(heap), vectorsOf(GraphIndex.fromSealed(off).seal()));
    }
}
