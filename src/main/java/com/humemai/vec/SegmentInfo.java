package com.humemai.vec;

import java.util.List;

/**
 * What the header, the footer and the locator of a segment file say, read without touching the sections. {@code headerCrc} is the stored
 * CRC32C of the header fields (the manifest records it). {@code verMin} and {@code verMax} are the smallest and largest per-row version
 * (both 0 when the file has no VERS section).
 */
@Internal
public record SegmentInfo(int major, int minor, int dim, Metric metric, int n, long fileLength, int headerCrc, long verMin, long verMax,
                          GraphIndex.Params params, int entryRow, int maxLevel, List<Section> sections) {

    /** one entry of the section table. {@code crc} is the CRC32C over exactly the section bytes (the padding before it is not covered). */
    public record Section(int id, int flags, int width, long count, long offset, long length, int crc) {
        public boolean required() {
            return (flags & SegmentFormat.FLAG_REQUIRED) != 0;
        }
    }

    public SegmentInfo {
        sections = List.copyOf(sections);
    }

    /** the section with this id, or null */
    public Section section(int id) {
        for (Section s : sections) if (s.id() == id) return s;
        return null;
    }
}
