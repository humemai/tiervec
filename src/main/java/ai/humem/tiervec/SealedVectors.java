package ai.humem.tiervec;

import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

/**
 * The vectors of a sealed segment (section VEC in docs/SEGMENT-FORMAT.md): rows of {@code dim} floats, one after the other, never changed
 * after construction. {@link SealedGraph} reaches them only through this. {@link SegmentWriter} takes them from this, and the tier's merge
 * and {@link GraphIndex#fromSealed} read rows through the graph. There are two implementations: {@link OnHeap}, the default, and
 * {@link Mapped}, which leaves the section in the segment file ({@link TieredIndex.VectorStorage#MAPPED}, #27). The other sections of a
 * segment are heap arrays of {@code SealedGraph} and do not go through this.
 *
 * <p>An implementation is safe for concurrent readers. A row index is 0 until the number of rows; the callers keep it in range. A distance
 * can need a scratch array: each thread takes its own from {@link #newScratch} and passes it to every distance it asks for.
 */
interface SealedVectors {
    /** squared L2 distance between the query (its first {@code dim} floats) and a row; {@code scratch} is what {@link #newScratch} gave this thread */
    float l2(float[] q, int row, float[] scratch);

    /** dot product of the query (its first {@code dim} floats) and a row; {@code scratch} is what {@link #newScratch} gave this thread */
    float dot(float[] q, int row, float[] scratch);

    /** a new array for the distances of one thread, which no other thread may use at the same time, or null when the distances need none */
    float[] newScratch();

    /** copies a row into {@code dst} from index {@code dstOff} on, without allocating */
    void copy(int row, float[] dst, int dstOff);

    /** a new array that holds a row */
    float[] row(int row);

    /** the number of floats, rows times {@code dim} */
    int length();

    /** puts {@code count} floats, from index {@code from} counted over all rows, at the position of {@code dst} */
    void put(FloatBuffer dst, int from, int count);

    /** heap bytes that the vectors hold (the scratch arrays of the threads are not counted) */
    long heapBytes();

    /** one heap array, row after row: what {@link GraphIndex#seal} builds and what {@link SegmentReader} loads by default */
    final class OnHeap implements SealedVectors {
        private final float[] a;
        private final int dim;

        /** takes the array without a copy; the caller does not write to it again */
        OnHeap(float[] a, int dim) {
            this.a = a;
            this.dim = dim;
        }

        @Override
        public float l2(float[] q, int row, float[] scratch) {
            return Kern.l2(q, 0, a, row * dim, dim);
        }

        @Override
        public float dot(float[] q, int row, float[] scratch) {
            return Kern.dot(q, 0, a, row * dim, dim);
        }

        @Override
        public float[] newScratch() {
            return null;
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

    /**
     * The section left in the segment file, which is mapped read-only. On JDK 21 without preview features the kernels read a {@code float[]},
     * so a distance copies its row into the scratch array of the thread first. One mapping holds at most {@code Integer.MAX_VALUE} bytes, so a
     * larger section is mapped in several pieces: a power of two of whole rows each, the last one shorter. No row crosses a piece.
     *
     * <p>Nothing here unmaps. A search can still read a segment that the tier has replaced or that belongs to a closed tier, and a read of an
     * unmapped page ends the JVM. The JVM unmaps a piece some time after the garbage collector has found its buffer unreachable. Until then
     * Windows refuses to delete the file ({@link Manifest#gc} leaves it for a later call), and on other systems a deleted file keeps its disk
     * space. The file must not change while it is mapped: the checksum of the section is checked before it is mapped, not at a read.
     */
    final class Mapped implements SealedVectors {
        /** set when this process maps its first piece, and never cleared */
        private static volatile boolean used;

        private final FloatBuffer[] pieces; // piece i holds the rows from i << shift on
        private final int dim, shift, mask, length;

        private Mapped(FloatBuffer[] pieces, int dim, int shift, int length) {
            this.pieces = pieces;
            this.dim = dim;
            this.shift = shift;
            this.mask = (1 << shift) - 1;
            this.length = length;
        }

        /**
         * Maps {@code rows} rows of {@code dim} floats that start at byte {@code offset} of the file. The caller has checked that they lie
         * inside the file and that {@code rows * dim} fits an array. The mapping stays valid after the channel is closed.
         *
         * @param maxPieceBytes the most bytes of one piece; a piece holds one row at least
         */
        static Mapped map(FileChannel ch, long offset, int rows, int dim, int maxPieceBytes) throws IOException {
            final int rowBytes = 4 * dim;
            final int perPiece = Integer.highestOneBit(Math.max(1, maxPieceBytes / rowBytes));
            final int shift = Integer.numberOfTrailingZeros(perPiece);
            final FloatBuffer[] pieces = new FloatBuffer[(int) (((long) rows + perPiece - 1) >>> shift)];
            for (int i = 0; i < pieces.length; i++) {
                final int first = i << shift;
                final long bytes = (long) Math.min(perPiece, rows - first) * rowBytes;
                pieces[i] = ch.map(FileChannel.MapMode.READ_ONLY, offset + (long) first * rowBytes, bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
                used = true;
            }
            return new Mapped(pieces, dim, shift, rows * dim);
        }

        /** whether this process has mapped any segment, which may still be mapped */
        static boolean anyMapped() {
            return used;
        }

        /** the number of mappings that hold the section */
        int pieces() {
            return pieces.length;
        }

        private float[] load(int row, float[] scratch) {
            copy(row, scratch, 0);
            return scratch;
        }

        @Override
        public float l2(float[] q, int row, float[] scratch) {
            return Kern.l2(q, 0, load(row, scratch), 0, dim);
        }

        @Override
        public float dot(float[] q, int row, float[] scratch) {
            return Kern.dot(q, 0, load(row, scratch), 0, dim);
        }

        @Override
        public float[] newScratch() {
            return new float[dim];
        }

        @Override
        public void copy(int row, float[] dst, int dstOff) {
            pieces[row >>> shift].get((row & mask) * dim, dst, dstOff, dim); // an absolute get: it moves no position, so readers share the buffer
        }

        @Override
        public float[] row(int row) {
            float[] r = new float[dim];
            copy(row, r, 0);
            return r;
        }

        @Override
        public int length() {
            return length;
        }

        @Override
        public void put(FloatBuffer dst, int from, int count) {
            final int perPiece = (mask + 1) * dim;
            for (int at = from, end = from + count; at < end; ) {
                final int in = at % perPiece;
                final int k = Math.min(end - at, perPiece - in);
                dst.put(pieces[at / perPiece].slice(in, k));
                at += k;
            }
        }

        @Override
        public long heapBytes() {
            return 0;
        }
    }
}
