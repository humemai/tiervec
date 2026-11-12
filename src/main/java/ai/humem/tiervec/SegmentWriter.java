package ai.humem.tiervec;

import static ai.humem.tiervec.SegmentFormat.*;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.CRC32C;

/**
 * Writes one {@link SealedGraph} as one immutable segment file. The file is created with CREATE_NEW (a segment is never overwritten),
 * streamed front to back (the header holds nothing that depends on the section bytes; the CRCs go into the footer, which is written last,
 * followed by the locator) and flushed to disk before the method returns. A file that was cut short never validates,
 * because the locator is the last thing written.
 */
@Internal
public final class SegmentWriter {
    private SegmentWriter() {}

    /** an additional section, used by the compatibility tests to build a file with a section this reader does not know */
    record Extra(int id, boolean required, byte[] data) {}

    /**
     * Writes the graph. {@code vers} is the per-row version of the tier (aligned with the rows of {@code g}) or null for a standalone graph
     * (then the file has no VERS section). Refuses a graph with soft-deleted rows: this version stores no tombstones inside a segment,
     * and writing such a graph would bring the deleted rows back on load.
     *
     * @return what the file's header, footer and locator say
     */
    public static SegmentInfo write(SealedGraph g, long[] vers, Path file) throws IOException {
        return write(g, vers, file, MAJOR, MINOR, List.of());
    }

    static SegmentInfo write(SealedGraph g, long[] vers, Path file, int major, int minor, List<Extra> extras) throws IOException {
        final int n = g.n;
        final int dim = g.dim;
        if (vers != null && vers.length != n) throw new IllegalArgumentException("vers has " + vers.length + " entries for " + n + " rows");
        if (g.size() != n) throw new IllegalStateException("the graph has deleted rows; this version does not store tombstones in a segment");
        if (g.vec.length != n * dim) throw new IllegalStateException("vector array has " + g.vec.length + " floats, expected " + (long) n * dim);
        final boolean sq8 = g.codes != null;
        if (sq8 && (g.qbase == null || g.qdelta == null || g.qnorm == null)) throw new IllegalStateException("SQ8 codes without their parameters");
        final GraphIndex.Params p = g.params;
        long verMin = 0, verMax = 0;
        if (vers != null && n > 0) {
            verMin = vers[0];
            verMax = vers[0];
            for (int i = 1; i < n; i++) {
                verMin = Math.min(verMin, vers[i]);
                verMax = Math.max(verMax, vers[i]);
            }
        }
        // opened outside the try below: when the name is taken, the existing file must not be deleted by the cleanup
        FileChannel ch = FileChannel.open(file, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE);
        try (ch) {
            Sink s = new Sink(ch);
            byte[] head = headerBlock(major, minor, dim, n, g.metric);
            s.raw(head, 0, head.length);
            final int headerCrc = ByteBuffer.wrap(head).order(ByteOrder.LITTLE_ENDIAN).getInt(HEADER_LEN - 4);

            s.begin();
            s.longs(g.ids, n);
            s.end(S_IDS, FLAG_REQUIRED, 8, n);
            if (vers != null) {
                s.begin();
                s.longs(vers, n);
                s.end(S_VERS, 0, 8, n);
            }
            s.begin();
            s.ints(g.sortedRows, n);
            s.end(S_BYID, FLAG_REQUIRED, 4, n);
            s.begin();
            s.floats(g.vec, n * dim);
            s.end(S_VEC, FLAG_REQUIRED, 4, (long) n * dim);
            s.begin();
            s.bytes(g.level, n);
            s.end(S_LEVEL, FLAG_REQUIRED, 1, n);
            s.begin();
            s.ints(g.off0, n + 1);
            s.end(S_L0_OFF, FLAG_REQUIRED, 4, n + 1);
            s.begin();
            s.ints(g.edges0, g.edges0.length);
            s.end(S_L0_EDGES, FLAG_REQUIRED, 4, g.edges0.length);
            s.begin();
            s.ints(g.upIdx, n);
            s.end(S_UP_IDX, FLAG_REQUIRED, 4, n);
            s.begin();
            s.ints(g.upOff, g.upOff.length);
            s.end(S_UP_OFF, FLAG_REQUIRED, 4, g.upOff.length);
            s.begin();
            s.ints(g.upEdges, g.upEdges.length);
            s.end(S_UP_EDGES, FLAG_REQUIRED, 4, g.upEdges.length);
            if (sq8) {
                s.begin();
                s.bytes(g.codes, n * dim);
                s.end(S_SQ8_CODES, 0, 1, (long) n * dim);
                s.begin();
                s.floats(g.qbase, n);
                s.floats(g.qdelta, n);
                s.floats(g.qnorm, n);
                s.end(S_SQ8_PARAMS, 0, 4, 3L * n);
            }
            for (Extra x : extras) {
                s.begin();
                s.bytes(x.data(), x.data().length);
                s.end(x.id(), x.required() ? FLAG_REQUIRED : 0, 1, x.data().length);
            }

            // footer, then the locator: the last bytes of the file
            final long footerOff = s.pos;
            ByteBuffer f = ByteBuffer.allocate(F_META_LEN + ENTRY_LEN * s.sections.size()).order(ByteOrder.LITTLE_ENDIAN);
            f.putInt(g.entryRow).putInt(g.maxLevel).putInt(p.M()).putInt(p.efConstruction()).putInt(p.efSearch()).putFloat(p.alpha());
            f.putLong(p.seed()).putLong(verMin).putLong(verMax);
            f.putInt(p.sq8() ? 1 : 0).putInt(s.sections.size()).putInt(F_META_LEN).putInt(0);
            for (SegmentInfo.Section sec : s.sections) {
                f.putShort((short) sec.id()).putShort((short) sec.flags()).putInt(sec.width()).putLong(sec.count());
                f.putLong(sec.offset()).putLong(sec.length()).putInt(sec.crc()).putInt(0);
            }
            if (f.position() != f.capacity()) throw new IllegalStateException("footer size " + f.position() + " != " + f.capacity());
            byte[] footer = f.array();
            final int footerCrc = crc32c(footer, 0, footer.length);
            s.raw(footer, 0, footer.length);
            final long fileLength = footerOff + footer.length + LOCATOR_LEN;
            ByteBuffer loc = ByteBuffer.allocate(LOCATOR_LEN).order(ByteOrder.LITTLE_ENDIAN);
            loc.put(LOCATOR_MAGIC).putLong(footerOff).putInt(footer.length).putInt(footerCrc).putLong(fileLength);
            s.raw(loc.array(), 0, LOCATOR_LEN);
            if (s.pos != fileLength) throw new IllegalStateException("file length " + s.pos + " != " + fileLength);
            ch.force(true);
            return new SegmentInfo(major, minor, dim, g.metric, n, fileLength, headerCrc, verMin, verMax, p, g.entryRow, g.maxLevel, s.sections);
        } catch (IOException | RuntimeException e) {
            try {
                Files.deleteIfExists(file); // a half-written file is never useful; a crash (no catch) leaves it as an orphan that open removes
            } catch (IOException suppressed) {
                e.addSuppressed(suppressed);
            }
            throw e;
        }
    }

    private static byte[] headerBlock(int major, int minor, int dim, int n, Metric metric) {
        byte[] b = new byte[HEADER_BLOCK];
        ByteBuffer h = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN);
        h.put(MAGIC);
        h.putShort((short) major).putShort((short) minor);
        h.putInt(HEADER_LEN);
        h.putInt(ENDIAN_PROBE);
        h.putInt(0); // flags, none defined yet
        h.putInt(dim);
        h.putInt(dim); // stride_floats: rows are not padded in this version
        h.putLong(n);
        h.put((byte) metricCode(metric)).put((byte) ID_WIDTH).put((byte) LAYOUT_CSR).put((byte) 0);
        h.putInt(ALIGN);
        if (h.position() != HEADER_LEN - 4) throw new IllegalStateException("header fields end at " + h.position());
        h.putInt(crc32c(b, 0, HEADER_LEN - 4));
        return b;
    }

    /** a position-tracking writer with a reusable little-endian chunk buffer and a CRC32C that restarts for every section */
    private static final class Sink {
        final FileChannel ch;
        final ByteBuffer buf = ByteBuffer.allocate(1 << 20).order(ByteOrder.LITTLE_ENDIAN);
        final CRC32C crc = new CRC32C();
        final List<SegmentInfo.Section> sections = new ArrayList<>();
        long pos;
        long start;

        Sink(FileChannel ch) {
            this.ch = ch;
        }

        void raw(byte[] b, int off, int len) throws IOException {
            ByteBuffer bb = ByteBuffer.wrap(b, off, len);
            while (bb.hasRemaining()) ch.write(bb);
            pos += len;
        }

        /** zero padding up to the next aligned offset (not part of any section CRC), then a fresh CRC */
        void begin() throws IOException {
            long aligned = alignUp(pos);
            if (aligned > pos) {
                byte[] pad = new byte[(int) (aligned - pos)];
                raw(pad, 0, pad.length);
            }
            crc.reset();
            start = pos;
        }

        void end(int id, int flags, int width, long count) {
            sections.add(new SegmentInfo.Section(id, flags, width, count, start, pos - start, (int) crc.getValue()));
        }

        /** writes buf[0, bytes) */
        private void flush(int bytes) throws IOException {
            crc.update(buf.array(), 0, bytes);
            buf.position(0);
            buf.limit(bytes);
            while (buf.hasRemaining()) ch.write(buf);
            pos += bytes;
        }

        void longs(long[] a, int count) throws IOException {
            final int per = buf.capacity() / 8;
            for (int i = 0; i < count; ) {
                int c = Math.min(per, count - i);
                buf.clear();
                buf.asLongBuffer().put(a, i, c);
                flush(c * 8);
                i += c;
            }
        }

        void ints(int[] a, int count) throws IOException {
            final int per = buf.capacity() / 4;
            for (int i = 0; i < count; ) {
                int c = Math.min(per, count - i);
                buf.clear();
                buf.asIntBuffer().put(a, i, c);
                flush(c * 4);
                i += c;
            }
        }

        void floats(float[] a, int count) throws IOException {
            final int per = buf.capacity() / 4;
            for (int i = 0; i < count; ) {
                int c = Math.min(per, count - i);
                buf.clear();
                buf.asFloatBuffer().put(a, i, c);
                flush(c * 4);
                i += c;
            }
        }

        void bytes(byte[] a, int count) throws IOException {
            final int per = buf.capacity();
            for (int i = 0; i < count; ) {
                int c = Math.min(per, count - i);
                buf.clear();
                buf.put(a, i, c);
                flush(c);
                i += c;
            }
        }
    }
}
