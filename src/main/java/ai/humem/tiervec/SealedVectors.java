package ai.humem.tiervec;

import java.nio.FloatBuffer;
import java.util.Arrays;

/**
 * The vectors of a sealed segment (section VEC in docs/SEGMENT-FORMAT.md): rows of {@code dim} floats, one after the other, never changed
 * after construction. {@link SealedGraph} reaches them only through this. {@link SegmentWriter} takes them from this, and the tier's merge
 * and {@link GraphIndex#fromSealed} read rows through the graph, so a reader that keeps the section in a mapped file can be another
 * implementation (#27). The only one in the library is {@link OnHeap}. The other sections of a segment are heap arrays of
 * {@code SealedGraph} and do not go through this.
 *
 * <p>An implementation is safe for concurrent readers. A row index is 0 until the number of rows; the callers keep it in range.
 */
interface SealedVectors {
    /** squared L2 distance between the query (its first {@code dim} floats) and a row */
    float l2(float[] q, int row);

    /** dot product of the query (its first {@code dim} floats) and a row */
    float dot(float[] q, int row);

    /** copies a row into {@code dst} from index {@code dstOff} on, without allocating */
    void copy(int row, float[] dst, int dstOff);

    /** a new array that holds a row */
    float[] row(int row);

    /** the number of floats, rows times {@code dim} */
    int length();

    /** puts {@code count} floats, from index {@code from} counted over all rows, at the position of {@code dst} */
    void put(FloatBuffer dst, int from, int count);

    /** heap bytes that the vectors hold */
    long heapBytes();

    /** one heap array, row after row: what {@link GraphIndex#seal} builds and what {@link SegmentReader} loads */
    final class OnHeap implements SealedVectors {
        private final float[] a;
        private final int dim;

        /** takes the array without a copy; the caller does not write to it again */
        OnHeap(float[] a, int dim) {
            this.a = a;
            this.dim = dim;
        }

        @Override
        public float l2(float[] q, int row) {
            return Kern.l2(q, 0, a, row * dim, dim);
        }

        @Override
        public float dot(float[] q, int row) {
            return Kern.dot(q, 0, a, row * dim, dim);
        }

        @Override
        public void copy(int row, float[] dst, int dstOff) {
            System.arraycopy(a, row * dim, dst, dstOff, dim);
        }

        @Override
        public float[] row(int row) {
            return Arrays.copyOfRange(a, row * dim, (row + 1) * dim);
        }

        @Override
        public int length() {
            return a.length;
        }

        @Override
        public void put(FloatBuffer dst, int from, int count) {
            dst.put(a, from, count);
        }

        @Override
        public long heapBytes() {
            return 4L * a.length;
        }
    }
}
