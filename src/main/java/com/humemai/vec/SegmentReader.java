package com.humemai.vec;

import static com.humemai.vec.SegmentFormat.*;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.CRC32C;

/**
 * Reads a segment file written by {@link SegmentWriter}. The read is copy-based: every section is read into the heap arrays that
 * {@link SealedGraph} already uses (no memory mapping in this version).
 *
 * <p>{@link #inspect} reads the header block, the locator and the footer only: a constant amount of work whatever the file size. A file that is
 * cut short, has bytes appended, or has a damaged header, footer or locator fails there, before any section is read. {@link #read} then
 * loads the sections; with {@code verifyAll} it also checks the CRC32C of each section against the footer.
 */
public final class SegmentReader {
    private SegmentReader() {}

    /** a loaded segment: the graph, the per-row versions (null when the file has no VERS section) and what the metadata said */
    public record Loaded(SealedGraph graph, long[] vers, SegmentInfo info) {}

    /** header, footer and locator only; never reads a section */
    public static SegmentInfo inspect(Path file) throws IOException {
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
            return inspect(ch);
        }
    }

    /**
     * Loads the graph. The stored id order (section BYID) is used as is, so the load does not sort. With verifyAll false the section bytes
     * are not checksummed (the header, footer and locator always are); a caller that wants every byte checked passes true.
     */
    public static SealedGraph read(Path file, boolean verifyAll) throws IOException {
        return readFull(file, verifyAll).graph();
    }

    /** like {@link #read} but also returns the per-row versions and the file's metadata */
    public static Loaded readFull(Path file, boolean verifyAll) throws IOException {
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
            SegmentInfo info = inspect(ch);
            return load(ch, info, verifyAll);
        }
    }

    private static ByteBuffer readAt(FileChannel ch, long pos, int len) throws IOException {
        ByteBuffer b = ByteBuffer.allocate(len).order(ByteOrder.LITTLE_ENDIAN);
        long p = pos;
        while (b.hasRemaining()) {
            int r = ch.read(b, p);
            if (r < 0) throw new CorruptException("unexpected end of file at byte " + p);
            p += r;
        }
        b.flip();
        return b;
    }

    static SegmentInfo inspect(FileChannel ch) throws IOException {
        final long size = ch.size();
        if (size < MAGIC.length) throw new CorruptException("file too short for a segment (" + size + " bytes)", "HEADER");
        ByteBuffer h = readAt(ch, 0, (int) Math.min(size, HEADER_BLOCK));
        for (int i = 0; i < MAGIC.length; i++) {
            if (h.get(i) != MAGIC[i]) throw new UnsupportedException("not a tiervec segment file (bad magic)");
        }
        if (size < HEADER_BLOCK + LOCATOR_LEN) throw new CorruptException("file too short for a segment (" + size + " bytes)", "HEADER");

        // header block
        final int major = h.getShort(H_MAJOR) & 0xFFFF;
        final int minor = h.getShort(H_MINOR) & 0xFFFF;
        if (major < 1 || major > MAJOR) {
            throw new UnsupportedException("segment format major version " + major + " is not supported (this reader supports " + MAJOR + ")");
        }
        final long headerLen = Integer.toUnsignedLong(h.getInt(H_LEN));
        if (headerLen < HEADER_LEN || headerLen > HEADER_BLOCK) throw new CorruptException("bad header length " + headerLen, "HEADER");
        final int hl = (int) headerLen;
        final int headerCrc = h.getInt(hl - 4);
        if (crc32c(h.array(), 0, hl - 4) != headerCrc) throw new CorruptException("CRC32C mismatch", "HEADER");
        if (h.getInt(H_PROBE) != ENDIAN_PROBE) throw new UnsupportedException("the byte-order probe does not read back as little-endian");
        final long dimL = Integer.toUnsignedLong(h.getInt(H_DIM));
        final long strideL = Integer.toUnsignedLong(h.getInt(H_STRIDE));
        final long nL = h.getLong(H_N);
        if (dimL < 1 || dimL > (1 << 20)) throw new CorruptException("bad dimension " + dimL, "HEADER");
        if (strideL != dimL) throw new UnsupportedException("row stride " + strideL + " differs from the dimension " + dimL + "; padded rows are not supported in this version");
        if (nL < 0 || nL * dimL > MAX_ARRAY) throw new UnsupportedException("row count " + nL + " is too large for one array");
        final Metric metric = metricOf(h.get(H_METRIC) & 0xFF);
        if ((h.get(H_IDW) & 0xFF) != ID_WIDTH) throw new UnsupportedException("id width " + (h.get(H_IDW) & 0xFF) + " is not supported");
        if ((h.get(H_LAYOUT) & 0xFF) != LAYOUT_CSR) throw new UnsupportedException("layer 0 layout " + (h.get(H_LAYOUT) & 0xFF) + " is not supported");
        final long align = Integer.toUnsignedLong(h.getInt(H_ALIGN));
        if (align < 8 || align > (1L << 30) || (align & (align - 1)) != 0) throw new CorruptException("bad section alignment " + align, "HEADER");
        final int dim = (int) dimL;
        final int n = (int) nL;

        // locator: constant-size, at the very end
        ByteBuffer l = readAt(ch, size - LOCATOR_LEN, LOCATOR_LEN);
        for (int i = 0; i < LOCATOR_MAGIC.length; i++) {
            if (l.get(i) != LOCATOR_MAGIC[i]) throw new CorruptException("locator not found at the end of the file (truncated, or bytes appended)", "LOCATOR");
        }
        final long footerOff = l.getLong(8);
        final long footerLen = Integer.toUnsignedLong(l.getInt(16));
        final int footerCrc = l.getInt(20);
        final long fileLen = l.getLong(24);
        if (fileLen != size) throw new CorruptException("the locator says " + fileLen + " bytes, the file has " + size, "LOCATOR");
        if (footerOff < HEADER_BLOCK || footerLen < F_META_LEN || footerLen > MAX_FOOTER || footerOff + footerLen + LOCATOR_LEN != size) {
            throw new CorruptException("footer position " + footerOff + " length " + footerLen + " does not fit a file of " + size + " bytes", "LOCATOR");
        }

        // footer
        final int fl = (int) footerLen;
        ByteBuffer f = readAt(ch, footerOff, fl);
        if (crc32c(f.array(), 0, fl) != footerCrc) throw new CorruptException("CRC32C mismatch", "FOOTER");
        final int entryRow = f.getInt(F_ENTRY_ROW);
        final int maxLevel = f.getInt(F_MAX_LEVEL);
        final GraphIndex.Params params = new GraphIndex.Params(f.getInt(F_M), f.getInt(F_EFC), f.getInt(F_EFS), f.getFloat(F_ALPHA),
                (f.getInt(F_FLAGS) & 1) != 0, f.getLong(F_SEED));
        final long verMin = f.getLong(F_VER_MIN);
        final long verMax = f.getLong(F_VER_MAX);
        final long count = Integer.toUnsignedLong(f.getInt(F_COUNT));
        final long entriesOff = Integer.toUnsignedLong(f.getInt(F_ENTRIES_OFF));
        if (entriesOff < F_META_LEN || count > 4096 || entriesOff + count * ENTRY_LEN > fl) throw new CorruptException("bad section table", "FOOTER");
        if (n > 0 && (entryRow < 0 || entryRow >= n || maxLevel < 0 || maxLevel > 127)) throw new CorruptException("bad entry row or top level", "FOOTER");

        List<SegmentInfo.Section> sections = new ArrayList<>();
        Map<Integer, SegmentInfo.Section> byId = new HashMap<>();
        long prevEnd = HEADER_BLOCK;
        for (int i = 0; i < count; i++) {
            int at = (int) (entriesOff + (long) i * ENTRY_LEN);
            int id = f.getShort(at) & 0xFFFF;
            int flags = f.getShort(at + 2) & 0xFFFF;
            long width = Integer.toUnsignedLong(f.getInt(at + 4));
            long cnt = f.getLong(at + 8);
            long off = f.getLong(at + 16);
            long len = f.getLong(at + 24);
            int crc = f.getInt(at + 32);
            String name = sectionName(id);
            if (cnt < 0 || off < prevEnd || len < 0 || off % align != 0 || off + len > footerOff || width < 1 || width > 8 || cnt > Long.MAX_VALUE / width || cnt * width != len) {
                throw new CorruptException("bad section table entry (offset " + off + ", length " + len + ")", name);
            }
            prevEnd = off + len;
            int known = widthOf(id);
            if (known == 0) {
                // a section this reader does not know: ignored unless the writer marked it required
                if ((flags & FLAG_REQUIRED) != 0) throw new UnsupportedException("the file has a required section " + name + " that this reader does not know");
            } else {
                if (width != known) throw new CorruptException("element width " + width + ", expected " + known, name);
                long expect = expectedCount(id, n, dim);
                if (expect >= 0 && cnt != expect) throw new CorruptException("element count " + cnt + ", expected " + expect, name);
                if (cnt > MAX_ARRAY) throw new UnsupportedException("section " + name + " is too large for one array");
            }
            SegmentInfo.Section s = new SegmentInfo.Section(id, flags, (int) width, cnt, off, len, crc);
            if (byId.put(id, s) != null) throw new CorruptException("section listed twice", name);
            sections.add(s);
        }
        for (int id : new int[] {S_IDS, S_BYID, S_VEC, S_LEVEL, S_L0_OFF, S_L0_EDGES, S_UP_IDX, S_UP_OFF, S_UP_EDGES}) {
            if (!byId.containsKey(id)) throw new CorruptException("required section is missing", sectionName(id));
        }
        if (byId.containsKey(S_SQ8_CODES) != byId.containsKey(S_SQ8_PARAMS)) {
            throw new CorruptException("SQ8 codes and SQ8 parameters must both be present or both absent", "SQ8_PARAMS");
        }
        return new SegmentInfo(major, minor, dim, metric, n, size, headerCrc, verMin, verMax, params, entryRow, maxLevel, sections);
    }

    /** the element count a known section must have, or -1 when it is free (edge lists, upper offsets) */
    private static long expectedCount(int id, int n, int dim) {
        switch (id) {
            case S_IDS:
            case S_VERS:
            case S_BYID:
            case S_LEVEL:
            case S_UP_IDX:
                return n;
            case S_VEC:
            case S_SQ8_CODES:
                return (long) n * dim;
            case S_L0_OFF:
                return n + 1L;
            case S_SQ8_PARAMS:
                return 3L * n;
            default:
                return -1;
        }
    }

    private static Loaded load(FileChannel ch, SegmentInfo info, boolean verify) throws IOException {
        Src src = new Src(ch, verify);
        final int n = info.n();
        long[] ids = src.longs(info.section(S_IDS));
        long[] vers = info.section(S_VERS) != null ? src.longs(info.section(S_VERS)) : null;
        int[] byId = src.ints(info.section(S_BYID));
        float[] vec = src.floats(info.section(S_VEC));
        byte[] level = src.bytes(info.section(S_LEVEL));
        int[] off0 = src.ints(info.section(S_L0_OFF));
        int[] edges0 = src.ints(info.section(S_L0_EDGES));
        int[] upIdx = src.ints(info.section(S_UP_IDX));
        int[] upOff = src.ints(info.section(S_UP_OFF));
        int[] upEdges = src.ints(info.section(S_UP_EDGES));
        byte[] codes = null;
        float[] qbase = null, qdelta = null, qnorm = null;
        if (info.section(S_SQ8_CODES) != null) {
            codes = src.bytes(info.section(S_SQ8_CODES));
            float[] prm = src.floats(info.section(S_SQ8_PARAMS));
            qbase = Arrays.copyOfRange(prm, 0, n);
            qdelta = Arrays.copyOfRange(prm, n, 2 * n);
            qnorm = Arrays.copyOfRange(prm, 2 * n, 3 * n);
        }
        // cheap structure checks that hold whether or not the CRCs were verified
        if (off0[0] != 0 || off0[n] != edges0.length) throw new CorruptException("layer 0 offsets disagree with the edge count", "L0_OFF");
        for (int i = 0; i < n; i++) {
            int r = byId[i];
            if (r < 0 || r >= n) throw new CorruptException("row " + r + " out of range", "BYID");
            if (i > 0 && ids[r] <= ids[byId[i - 1]]) throw new CorruptException("rows are not in strictly increasing id order", "BYID");
        }
        SealedGraph g = new SealedGraph(info.dim(), info.metric(), info.params(), n, vec, codes, qbase, qdelta, qnorm, ids, level, off0, edges0,
                upIdx, upOff, upEdges, info.entryRow(), info.maxLevel(), byId);
        return new Loaded(g, vers, info);
    }

    /** reads sections into typed arrays through a reusable little-endian chunk buffer, checksumming when asked */
    private static final class Src {
        final FileChannel ch;
        final boolean verify;
        final ByteBuffer buf = ByteBuffer.allocate(1 << 20).order(ByteOrder.LITTLE_ENDIAN);
        final CRC32C crc = new CRC32C();

        Src(FileChannel ch, boolean verify) {
            this.ch = ch;
            this.verify = verify;
        }

        /** reads exactly {@code bytes} bytes at {@code pos} into buf, which is then positioned at 0 with that limit */
        private void fill(long pos, int bytes) throws IOException {
            buf.clear();
            buf.limit(bytes);
            long p = pos;
            while (buf.hasRemaining()) {
                int r = ch.read(buf, p);
                if (r < 0) throw new CorruptException("unexpected end of file at byte " + p);
                p += r;
            }
            buf.flip();
            if (verify) crc.update(buf.array(), 0, bytes);
        }

        private void check(SegmentInfo.Section s) throws CorruptException {
            if (verify && (int) crc.getValue() != s.crc()) throw new CorruptException("CRC32C mismatch", sectionName(s.id()));
        }

        long[] longs(SegmentInfo.Section s) throws IOException {
            final int count = (int) s.count();
            long[] a = new long[count];
            crc.reset();
            final int per = buf.capacity() / 8;
            long pos = s.offset();
            for (int i = 0; i < count; ) {
                int c = Math.min(per, count - i);
                fill(pos, c * 8);
                buf.asLongBuffer().get(a, i, c);
                pos += (long) c * 8;
                i += c;
            }
            check(s);
            return a;
        }

        int[] ints(SegmentInfo.Section s) throws IOException {
            final int count = (int) s.count();
            int[] a = new int[count];
            crc.reset();
            final int per = buf.capacity() / 4;
            long pos = s.offset();
            for (int i = 0; i < count; ) {
                int c = Math.min(per, count - i);
                fill(pos, c * 4);
                buf.asIntBuffer().get(a, i, c);
                pos += (long) c * 4;
                i += c;
            }
            check(s);
            return a;
        }

        float[] floats(SegmentInfo.Section s) throws IOException {
            final int count = (int) s.count();
            float[] a = new float[count];
            crc.reset();
            final int per = buf.capacity() / 4;
            long pos = s.offset();
            for (int i = 0; i < count; ) {
                int c = Math.min(per, count - i);
                fill(pos, c * 4);
                buf.asFloatBuffer().get(a, i, c);
                pos += (long) c * 4;
                i += c;
            }
            check(s);
            return a;
        }

        byte[] bytes(SegmentInfo.Section s) throws IOException {
            final int count = (int) s.count();
            byte[] a = new byte[count];
            crc.reset();
            final int per = buf.capacity();
            long pos = s.offset();
            for (int i = 0; i < count; ) {
                int c = Math.min(per, count - i);
                fill(pos, c);
                buf.get(a, i, c);
                pos += c;
                i += c;
            }
            check(s);
            return a;
        }
    }
}
