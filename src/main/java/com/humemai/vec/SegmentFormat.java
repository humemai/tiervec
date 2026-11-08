package com.humemai.vec;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.zip.CRC32C;

/**
 * Constants and small helpers of the segment file format (see docs/SEGMENT-FORMAT.md for the byte layout). All multi-byte values are
 * little-endian. A segment file is: a 4096-byte header block, sections that each start at a multiple of the alignment, a footer with the
 * section table, and a 32-byte locator at the very end that names the footer and the exact file length.
 */
public final class SegmentFormat {
    private SegmentFormat() {}

    /** first 8 bytes of a segment file */
    public static final byte[] MAGIC = {'T', 'V', 'S', 'E', 'G', '0', '0', '1'};
    /** first 8 bytes of the locator */
    public static final byte[] LOCATOR_MAGIC = {'T', 'V', 'S', 'E', 'G', 'E', 'N', 'D'};
    /** a reader refuses a larger major; a larger minor opens and ignores what it does not know */
    public static final int MAJOR = 1, MINOR = 0;
    public static final int HEADER_BLOCK = 4096;
    public static final int ALIGN = 4096;
    /** bytes of the header block that carry fields (the header CRC is the last 4 of them) */
    public static final int HEADER_LEN = 52;
    public static final int LOCATOR_LEN = 32;
    public static final int ENDIAN_PROBE = 0x01020304;
    public static final int ID_WIDTH = 8;
    /** layer 0 stored as CSR (offsets plus edges); the only layout of this version */
    public static final int LAYOUT_CSR = 0;
    /** section flag: a reader that does not know the section id must refuse the file */
    public static final int FLAG_REQUIRED = 1;

    // section ids, never reused
    public static final int S_IDS = 1, S_VERS = 2, S_BYID = 3, S_VEC = 4, S_LEVEL = 5, S_L0_OFF = 6, S_L0_EDGES = 7, S_UP_IDX = 8,
            S_UP_OFF = 9, S_UP_EDGES = 10, S_SQ8_CODES = 11, S_SQ8_PARAMS = 12;

    // byte offsets of the header fields
    static final int H_MAJOR = 8, H_MINOR = 10, H_LEN = 12, H_PROBE = 16, H_FLAGS = 20, H_DIM = 24, H_STRIDE = 28, H_N = 32,
            H_METRIC = 40, H_IDW = 41, H_LAYOUT = 42, H_ALIGN = 44;
    // byte offsets of the footer fields; the section entries start at the offset stored in F_ENTRIES_OFF
    static final int F_ENTRY_ROW = 0, F_MAX_LEVEL = 4, F_M = 8, F_EFC = 12, F_EFS = 16, F_ALPHA = 20, F_SEED = 24, F_VER_MIN = 32,
            F_VER_MAX = 40, F_FLAGS = 48, F_COUNT = 52, F_ENTRIES_OFF = 56, F_META_LEN = 64;
    /** bytes of one section entry in the footer: id u16, flags u16, width u32, count u64, offset u64, length u64, crc u32, reserved u32 */
    static final int ENTRY_LEN = 40;
    static final int MAX_FOOTER = 1 << 20;
    static final int MAX_ARRAY = Integer.MAX_VALUE - 8;

    /** An integrity failure: a CRC, length or structure check failed. {@link #section()} names the section when one is known. */
    public static class CorruptException extends IOException {
        private static final long serialVersionUID = 1L;
        private final String section;

        public CorruptException(String message) {
            this(message, (String) null);
        }

        public CorruptException(String message, String section) {
            super(section == null ? message : message + " (section " + section + ")");
            this.section = section;
        }

        public CorruptException(String message, Throwable cause) {
            super(message, cause);
            this.section = null;
        }

        /** name of the damaged section (or HEADER, FOOTER, LOCATOR, MANIFEST), null when the failure is not tied to one */
        public String section() {
            return section;
        }
    }

    /** The file is intact as far as is known but this reader does not understand it: a larger major version, an unknown required section, another byte order. */
    public static class UnsupportedException extends IOException {
        private static final long serialVersionUID = 1L;

        public UnsupportedException(String message) {
            super(message);
        }
    }

    /** The persisted segments belong to another index definition (dimension, metric or graph parameters differ from the ones asked for). */
    public static class MismatchException extends IOException {
        private static final long serialVersionUID = 1L;

        public MismatchException(String message) {
            super(message);
        }
    }

    /** CRC32C of b[off, off + len) as a signed int (the stored form) */
    public static int crc32c(byte[] b, int off, int len) {
        CRC32C c = new CRC32C();
        c.update(b, off, len);
        return (int) c.getValue();
    }

    /** the smallest multiple of {@link #ALIGN} that is at least x */
    static long alignUp(long x) {
        return (x + ALIGN - 1) / ALIGN * ALIGN;
    }

    /** explicit numeric metric codes (never the enum ordinal, which may change when a constant is added) */
    static int metricCode(Metric m) {
        switch (m) {
            case COSINE: return 0;
            case DOT: return 1;
            case L2: return 2;
            default: throw new IllegalArgumentException("metric " + m);
        }
    }

    static Metric metricOf(int code) throws IOException {
        switch (code) {
            case 0: return Metric.COSINE;
            case 1: return Metric.DOT;
            case 2: return Metric.L2;
            default: throw new UnsupportedException("unknown metric code " + code);
        }
    }

    /** element width in bytes of a section this version knows, 0 for an unknown id */
    static int widthOf(int id) {
        switch (id) {
            case S_IDS:
            case S_VERS:
                return 8;
            case S_BYID:
            case S_VEC:
            case S_L0_OFF:
            case S_L0_EDGES:
            case S_UP_IDX:
            case S_UP_OFF:
            case S_UP_EDGES:
            case S_SQ8_PARAMS:
                return 4;
            case S_LEVEL:
            case S_SQ8_CODES:
                return 1;
            default:
                return 0;
        }
    }

    public static String sectionName(int id) {
        switch (id) {
            case S_IDS: return "IDS";
            case S_VERS: return "VERS";
            case S_BYID: return "BYID";
            case S_VEC: return "VEC";
            case S_LEVEL: return "LEVEL";
            case S_L0_OFF: return "L0_OFF";
            case S_L0_EDGES: return "L0_EDGES";
            case S_UP_IDX: return "UP_IDX";
            case S_UP_OFF: return "UP_OFF";
            case S_UP_EDGES: return "UP_EDGES";
            case S_SQ8_CODES: return "SQ8_CODES";
            case S_SQ8_PARAMS: return "SQ8_PARAMS";
            default: return "UNKNOWN_" + id;
        }
    }

    /** true on Windows, decided once; the JDK cannot open a directory there, so a directory cannot be flushed from Java */
    static final boolean WINDOWS = System.getProperty("os.name", "").startsWith("Windows");

    /**
     * flush a directory's entries to disk (new names, renames). Linux and macOS open a directory for reading and force it. On Windows
     * {@code FileChannel.open} on a directory throws {@code AccessDeniedException} and no other way exists in the JDK, so this does nothing
     * there: file contents are still forced by the callers with {@code FileChannel.force}, and the directory entry (the new name or the
     * rename) is left to NTFS, so a power loss right after a rename can lose the rename. Every other failure on every platform is thrown.
     */
    static void fsyncDir(Path dir) throws IOException {
        if (WINDOWS) return;
        try (FileChannel ch = FileChannel.open(dir, StandardOpenOption.READ)) {
            ch.force(true);
        }
    }
}
