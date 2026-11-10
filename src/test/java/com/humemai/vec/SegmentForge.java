package com.humemai.vec;

import static com.humemai.vec.SegmentFormat.F_COUNT;
import static com.humemai.vec.SegmentFormat.F_ENTRIES_OFF;
import static com.humemai.vec.SegmentFormat.HEADER_BLOCK;
import static com.humemai.vec.SegmentFormat.H_LEN;
import static com.humemai.vec.SegmentFormat.LOCATOR_LEN;
import static com.humemai.vec.SegmentFormat.LOCATOR_MAGIC;
import static com.humemai.vec.SegmentFormat.crc32c;
import static org.junit.jupiter.api.Assertions.fail;

import com.humemai.vec.SegmentFormat.CorruptException;
import com.humemai.vec.SegmentFormat.UnsupportedException;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.opentest4j.AssertionFailedError;

/**
 * Takes a good segment file apart and puts it together again with chosen fields changed. {@link #build} recomputes every CRC and length
 * that sits above the change (the header CRC, the section CRCs of changed sections, the footer CRC and the locator), so the file that comes
 * out is damaged in exactly one place and a reader can only find that damage through the check that is meant to find it. A test that wants a
 * CRC to stay wrong changes the bytes of the built file instead.
 */
final class SegmentForge {
    /** one entry of the section table, as the footer holds it */
    static final class Entry {
        int id;
        int flags;
        long width;
        long count;
        long offset;
        long length;
        int crc;
    }

    /** the 4096-byte header block */
    final byte[] head;
    /** the bytes from the end of the header block to the start of the footer: the sections and the zero fill between them */
    byte[] body;
    /** the footer metadata that precedes the section entries */
    byte[] meta;
    final List<Entry> entries = new ArrayList<>();
    /** recompute the header CRC (at header length minus 4) in {@link #build} */
    boolean fixHeaderCrc = true;
    /** write the entry count and the first entry offset into the footer metadata in {@link #build}; false keeps whatever {@link #meta} holds */
    boolean syncMeta = true;
    /** values for the locator fields that replace the true ones, null keeps the true value */
    Long locFooterOff, locFooterLen, locFileLen;

    private SegmentForge(byte[] file) {
        ByteBuffer b = le(file);
        final int size = file.length;
        final int footerOff = (int) b.getLong(size - LOCATOR_LEN + 8);
        final int entriesOff = b.getInt(footerOff + F_ENTRIES_OFF);
        final int count = b.getInt(footerOff + F_COUNT);
        head = Arrays.copyOfRange(file, 0, HEADER_BLOCK);
        body = Arrays.copyOfRange(file, HEADER_BLOCK, footerOff);
        meta = Arrays.copyOfRange(file, footerOff, footerOff + entriesOff);
        for (int i = 0; i < count; i++) {
            int at = footerOff + entriesOff + i * SegmentFormat.ENTRY_LEN;
            Entry e = new Entry();
            e.id = b.getShort(at) & 0xFFFF;
            e.flags = b.getShort(at + 2) & 0xFFFF;
            e.width = Integer.toUnsignedLong(b.getInt(at + 4));
            e.count = b.getLong(at + 8);
            e.offset = b.getLong(at + 16);
            e.length = b.getLong(at + 24);
            e.crc = b.getInt(at + 32);
            entries.add(e);
        }
    }

    /** takes apart a file that {@link SegmentWriter} wrote */
    static SegmentForge of(byte[] file) {
        return new SegmentForge(file);
    }

    static ByteBuffer le(byte[] a) {
        return ByteBuffer.wrap(a).order(ByteOrder.LITTLE_ENDIAN);
    }

    /** the header block for in-place changes of header fields */
    ByteBuffer headBuf() {
        return le(head);
    }

    /** the footer metadata for in-place changes */
    ByteBuffer metaBuf() {
        return le(meta);
    }

    Entry entry(int id) {
        for (Entry e : entries) if (e.id == id) return e;
        throw new IllegalArgumentException("no section " + id);
    }

    byte[] sectionBytes(int id) {
        Entry e = entry(id);
        return Arrays.copyOfRange(body, (int) e.offset - HEADER_BLOCK, (int) (e.offset + e.length) - HEADER_BLOCK);
    }

    /** replaces the bytes of a section (same length) and puts the CRC of the new bytes into its entry */
    void setSectionBytes(int id, byte[] data) {
        Entry e = entry(id);
        if (data.length != e.length) throw new IllegalArgumentException("length " + data.length + " != " + e.length);
        System.arraycopy(data, 0, body, (int) e.offset - HEADER_BLOCK, data.length);
        e.crc = crc32c(data, 0, data.length);
    }

    /** appends a section after the last one, at the next aligned offset, with a correct CRC, and lists it in the table */
    Entry appendSection(int id, int flags, int width, byte[] data) {
        long end = HEADER_BLOCK + body.length;
        long off = SegmentFormat.alignUp(end);
        byte[] nb = Arrays.copyOf(body, (int) (off - HEADER_BLOCK) + data.length);
        System.arraycopy(data, 0, nb, (int) (off - HEADER_BLOCK), data.length);
        body = nb;
        Entry e = new Entry();
        e.id = id;
        e.flags = flags;
        e.width = width;
        e.count = data.length / width;
        e.offset = off;
        e.length = data.length;
        e.crc = crc32c(data, 0, data.length);
        entries.add(e);
        return e;
    }

    /** the file, with the CRCs and the lengths above the changes recomputed */
    byte[] build() {
        if (fixHeaderCrc) {
            int hl = headBuf().getInt(H_LEN);
            if (hl >= 8 && hl <= HEADER_BLOCK) headBuf().putInt(hl - 4, crc32c(head, 0, hl - 4));
        }
        if (syncMeta) {
            metaBuf().putInt(F_COUNT, entries.size());
            metaBuf().putInt(F_ENTRIES_OFF, meta.length);
        }
        ByteBuffer f = ByteBuffer.allocate(meta.length + SegmentFormat.ENTRY_LEN * entries.size()).order(ByteOrder.LITTLE_ENDIAN);
        f.put(meta);
        for (Entry e : entries) {
            f.putShort((short) e.id).putShort((short) e.flags).putInt((int) e.width).putLong(e.count).putLong(e.offset).putLong(e.length);
            f.putInt(e.crc).putInt(0);
        }
        byte[] footer = f.array();
        final long footerOff = HEADER_BLOCK + body.length;
        final long fileLen = footerOff + footer.length + LOCATOR_LEN;
        ByteBuffer out = ByteBuffer.allocate((int) fileLen).order(ByteOrder.LITTLE_ENDIAN);
        out.put(head).put(body).put(footer);
        out.put(LOCATOR_MAGIC).putLong(locFooterOff != null ? locFooterOff : footerOff).putInt((int) (locFooterLen != null ? locFooterLen : footer.length));
        out.putInt(crc32c(footer, 0, footer.length)).putLong(locFileLen != null ? locFileLen : fileLen);
        return out.array();
    }

    /**
     * Loads the file the way a host does ({@link SealedGraph#load}, every section checksummed) and returns the exception that refused it.
     * Fails the test when a graph comes back, and when anything but a {@link CorruptException} or an {@link UnsupportedException} comes out:
     * a runtime exception, an error or another IOException is an uncontrolled failure.
     */
    static IOException refused(Path file, String what) {
        return refused(file, true, what);
    }

    static IOException refused(Path file, boolean verifyAll, String what) {
        try {
            SealedGraph g = SegmentReader.read(file, verifyAll);
            fail(what + ": a graph with " + g.n + " rows came back from a damaged file");
            return null;
        } catch (CorruptException | UnsupportedException e) {
            return e;
        } catch (AssertionFailedError e) {
            throw e;
        } catch (Throwable t) {
            throw new AssertionFailedError(what + ": an uncontrolled failure instead of a named exception: " + t, t);
        }
    }

    /** the same as {@link #refused} for a file that is passed as bytes (written to {@code file} first) */
    static IOException refused(Path file, byte[] bytes, String what) throws IOException {
        java.nio.file.Files.write(file, bytes);
        return refused(file, what);
    }
}
