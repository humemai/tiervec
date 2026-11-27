package ai.humem.tiervec;

import ai.humem.tiervec.SegmentFormat.CorruptException;
import ai.humem.tiervec.SegmentFormat.UnsupportedException;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.zip.CRC32C;

/**
 * The tail file of a generation, {@code tail-<generation>.tvt}: the rows that the tier held outside its segments when {@code checkpoint} ran
 * (see docs/SEGMENT-FORMAT.md, "Tail file"). All values are little-endian. The file is a header, then the id of every row, then the version of
 * every row, then the vector of every row, as the host passed it. The rows are in the order of their versions, which is the order in which
 * they arrived. A generation with an empty tail has no tail file.
 *
 * <p>Every byte is covered by a CRC32C: the header by the one in its last four bytes, the rows by the one that the manifest holds. The file
 * has no checksum of its own after the rows, because a CRC32C over bytes followed by their own CRC32C is the same constant for any bytes, and
 * a manifest checksum over such a file would not tie the manifest to the content.
 */
final class TailFile {
    private TailFile() {}

    static final byte[] MAGIC = {'T', 'V', 'T', 'A', 'I', 'L', '0', '1'};
    /** a reader refuses a larger major and opens a larger minor, like the segment format */
    static final int MAJOR = 1, MINOR = 0;
    /** bytes of the header in this version; the last 4 are its CRC32C. A newer minor may make it longer, up to {@link #MAX_HEADER} */
    static final int HEADER_LEN = 40;
    static final int MAX_HEADER = 4096;
    /** byte offsets of the header fields */
    static final int H_MAJOR = 8, H_MINOR = 10, H_LEN = 12, H_GEN = 16, H_ROWS = 24, H_DIM = 32;
    private static final int BUFFER = 1 << 16;

    /** the rows of a tail: row {@code i} has {@code ids[i]}, {@code vers[i]} and {@code vecs[i]}; versions increase strictly with the row */
    record Rows(long[] ids, long[] vers, float[][] vecs) {
        static final Rows NONE = new Rows(new long[0], new long[0], new float[0][]);

        int size() {
            return ids.length;
        }
    }

    /** what the manifest records of a tail file that was written: its length and the CRC32C of the rows (every byte after the header) */
    record Written(long length, int crc) {}

    /** the bytes of one row in the file: an id, a version, and {@code dim} floats */
    private static long rowBytes(int dim) {
        return 16 + 4L * dim;
    }

    // ---- write ----

    /** writes the rows to {@code file}, which is replaced when it is there, and flushes it; there must be at least one row */
    static Written write(Path file, long generation, int dim, Rows rows) throws IOException {
        final int n = rows.size();
        if (n == 0) throw new IllegalArgumentException("an empty tail has no file");
        if (rows.vers().length != n || rows.vecs().length != n) throw new IllegalArgumentException("the arrays of the tail rows differ in length");
        ByteBuffer b = ByteBuffer.allocate(BUFFER).order(ByteOrder.LITTLE_ENDIAN);
        b.put(MAGIC).putShort((short) MAJOR).putShort((short) MINOR).putInt(HEADER_LEN).putLong(generation).putLong(n).putInt(dim);
        b.putInt(SegmentFormat.crc32c(b.array(), 0, HEADER_LEN - 4));
        final CRC32C crc = new CRC32C();
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE)) {
            drain(ch, b, null); // the header, which carries its own CRC
            for (long id : rows.ids()) {
                if (b.remaining() < 8) drain(ch, b, crc);
                b.putLong(id);
            }
            for (long ver : rows.vers()) {
                if (b.remaining() < 8) drain(ch, b, crc);
                b.putLong(ver);
            }
            for (float[] v : rows.vecs()) {
                if (v.length != dim) throw new IllegalArgumentException("dim " + v.length + " != " + dim);
                for (float x : v) {
                    if (b.remaining() < 4) drain(ch, b, crc);
                    b.putInt(Float.floatToRawIntBits(x)); // the bits as they are, a NaN payload included
                }
            }
            drain(ch, b, crc);
            ch.force(true);
        }
        return new Written(HEADER_LEN + n * rowBytes(dim), (int) crc.getValue());
    }

    /** writes what the buffer holds, adds it to the CRC when there is one, and empties the buffer */
    private static void drain(FileChannel ch, ByteBuffer b, CRC32C crc) throws IOException {
        b.flip();
        if (crc != null) crc.update(b.array(), 0, b.limit());
        while (b.hasRemaining()) ch.write(b);
        b.clear();
    }

    // ---- read ----

    /**
     * The check that {@code Manifest.readLatest} makes, which reads the header only: the file is there with the length that the manifest holds,
     * its header is intact and in a format this reader understands, and it belongs to this manifest. The rows are not read.
     */
    static void check(Path file, Manifest m) throws IOException {
        try (FileChannel ch = open(file)) {
            header(ch, file, m);
        }
    }

    /**
     * Reads the rows of the tail file that {@code m} names, after the checks of {@link #check}. The CRC32C of the rows is compared with the
     * manifest's before anything is returned, and the versions must increase strictly from at least 1 to at most the version counter of the
     * manifest. Whether the rows agree with the segments is the tier's check.
     */
    static Rows read(Path file, Manifest m) throws IOException {
        final int dim = m.fingerprint().dim();
        try (FileChannel ch = open(file)) {
            final int headerLen = header(ch, file, m);
            final int n = Math.toIntExact(m.tailRows());
            final long[] ids = new long[n];
            final long[] vers = new long[n];
            final float[][] vecs = new float[n][dim];
            final In in = new In(ch, headerLen, m.tailLength());
            for (int i = 0; i < n; i++) ids[i] = in.longValue();
            for (int i = 0; i < n; i++) vers[i] = in.longValue();
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < dim; j++) vecs[i][j] = Float.intBitsToFloat(in.intValue());
            }
            if ((int) in.crc.getValue() != m.tailCrc()) throw new CorruptException("CRC32C mismatch in " + file.getFileName(), "TAIL");
            long last = 0;
            for (long ver : vers) {
                if (ver <= last || ver > m.versionCounter()) {
                    throw new CorruptException("tail version " + ver + " after " + last + " does not fit the version counter " + m.versionCounter(), "TAIL");
                }
                last = ver;
            }
            return new Rows(ids, vers, vecs);
        }
    }

    private static FileChannel open(Path file) throws IOException {
        try {
            return FileChannel.open(file, StandardOpenOption.READ);
        } catch (NoSuchFileException x) {
            throw new CorruptException("tail file " + file.getFileName() + " is missing", "MANIFEST");
        }
    }

    /**
     * Checks the header and the length of the file and returns the header length. The frame is that of a segment header (magic, version, header
     * length at offset 12, a CRC32C in the last 4 bytes of the header), so the CRC is checked before the version is believed: a damaged version
     * field is damage, and only an intact header can say that the file is in a newer format.
     */
    private static int header(FileChannel ch, Path file, Manifest m) throws IOException {
        final long size = ch.size();
        if (size != m.tailLength()) {
            throw new CorruptException("tail file " + file.getFileName() + " has " + size + " bytes, the manifest says " + m.tailLength(), "TAIL");
        }
        if (size < HEADER_LEN) throw new CorruptException("tail file " + file.getFileName() + " is too short", "TAIL");
        ByteBuffer h = ByteBuffer.allocate((int) Math.min(size, MAX_HEADER)).order(ByteOrder.LITTLE_ENDIAN);
        readFully(ch, h, 0);
        for (int i = 0; i < MAGIC.length; i++) {
            if (h.get(i) != MAGIC[i]) throw new CorruptException("bad tail magic in " + file.getFileName(), "TAIL");
        }
        final long headerLen = Integer.toUnsignedLong(h.getInt(H_LEN));
        if (headerLen < HEADER_LEN || headerLen > h.capacity()) throw new CorruptException("bad tail header length " + headerLen, "TAIL");
        final int hl = (int) headerLen;
        if (SegmentFormat.crc32c(h.array(), 0, hl - 4) != h.getInt(hl - 4)) throw new CorruptException("CRC32C mismatch in the tail header", "TAIL");
        final int major = h.getShort(H_MAJOR) & 0xFFFF;
        if (major < 1 || major > MAJOR) throw new UnsupportedException("tail format major version " + major + " is not supported (this reader supports " + MAJOR + ")");
        final long gen = h.getLong(H_GEN);
        final long rows = h.getLong(H_ROWS);
        final long dim = Integer.toUnsignedLong(h.getInt(H_DIM));
        if (gen != m.generation() || rows != m.tailRows() || dim != m.fingerprint().dim()) {
            throw new CorruptException("tail file " + file.getFileName() + " does not belong to this manifest: generation " + gen + ", " + rows
                    + " rows of dimension " + dim, "TAIL");
        }
        // the rows fill the file after the header exactly; written as a division so that no row count can overflow
        final long body = size - hl;
        final long perRow = rowBytes(m.fingerprint().dim());
        if (rows < 1 || rows > Integer.MAX_VALUE - 8 || body % perRow != 0 || body / perRow != rows) {
            throw new CorruptException("tail file " + file.getFileName() + " has " + body + " bytes of rows, which are not " + rows + " rows", "TAIL");
        }
        return hl;
    }

    private static void readFully(FileChannel ch, ByteBuffer b, long position) throws IOException {
        long p = position;
        while (b.hasRemaining()) {
            int r = ch.read(b, p);
            if (r < 0) throw new CorruptException("unexpected end of the tail file at byte " + p, "TAIL");
            p += r;
        }
    }

    /** reads the bytes after the header from front to back through one buffer, and keeps their CRC32C */
    private static final class In {
        private final FileChannel ch;
        private final long end;
        private long position;
        private final ByteBuffer b = ByteBuffer.allocate(BUFFER).order(ByteOrder.LITTLE_ENDIAN);
        final CRC32C crc = new CRC32C();

        In(FileChannel ch, long from, long end) {
            this.ch = ch;
            this.position = from;
            this.end = end;
            b.limit(0);
        }

        private void need(int bytes) throws IOException {
            if (b.remaining() >= bytes) return;
            b.compact();
            final int kept = b.position();
            b.limit((int) Math.min(b.capacity(), kept + (end - position)));
            final int fresh = b.remaining();
            readFully(ch, b, position);
            position += fresh;
            crc.update(b.array(), kept, fresh);
            b.flip();
            if (b.remaining() < bytes) throw new CorruptException("unexpected end of the tail file at byte " + position, "TAIL");
        }

        long longValue() throws IOException {
            need(8);
            return b.getLong();
        }

        int intValue() throws IOException {
            need(4);
            return b.getInt();
        }
    }
}
