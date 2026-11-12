package ai.humem.tiervec;

import static ai.humem.tiervec.SegmentForge.refused;
import static ai.humem.tiervec.SegmentTestSupport.graph;
import static ai.humem.tiervec.SegmentTestSupport.params;
import static ai.humem.tiervec.SegmentTestSupport.vectors;
import static ai.humem.tiervec.SegmentTestSupport.versOf;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.humem.tiervec.SegmentFormat.CorruptException;
import ai.humem.tiervec.SegmentFormat.UnsupportedException;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Segment files that are damaged in exactly one place with every CRC above the damage recomputed (see {@link SegmentForge}), so that a reader
 * can find the damage only through the rule that is meant to find it: the version rule, the byte-order probe, the section table checks, the
 * locator checks and the checks of the stored id order and the layer 0 offsets. {@link SegmentCorruptionTest} covers the bytes that a CRC
 * covers; this class covers the rules that hold when the CRCs are fine.
 */
class SegmentForgedFileTest {
    private static final int N = 300, DIM = 4;
    private static final float[][] QS = vectors(40, DIM, 123);

    @TempDir
    Path tmp;

    private SealedGraph orig;
    private byte[] good;
    private Path f;

    @BeforeEach
    void write() throws IOException {
        orig = graph(vectors(N, DIM, 17), Metric.L2, params(true));
        Path g = tmp.resolve("good.tvs");
        SegmentWriter.write(orig, versOf(N), g);
        good = Files.readAllBytes(g);
        f = tmp.resolve("forged.tvs");
    }

    private SegmentForge forge() {
        return SegmentForge.of(good);
    }

    private void assertServedLikeTheOriginal(Path p) throws IOException {
        SealedGraph h = SealedGraph.load(p);
        assertArrayEquals(orig.ids, h.ids);
        assertArrayEquals(orig.vec, h.vec);
        assertArrayEquals(orig.level, h.level);
        assertArrayEquals(orig.off0, h.off0);
        assertArrayEquals(orig.edges0, h.edges0);
        assertArrayEquals(orig.codes, h.codes);
        for (float[] q : QS) assertArrayEquals(orig.search(q, 10, null), h.search(q, 10, null));
    }

    // ---- the version rule ----

    @ParameterizedTest(name = "major {0}")
    @ValueSource(ints = {0, 2, 3, 255, 256, 4097, 65535})
    void aMajorVersionThatIsNotOneIsRefusedAsUnsupportedWithValidChecksums(int major) throws IOException {
        SegmentForge forge = forge();
        forge.headBuf().putShort(SegmentFormat.H_MAJOR, (short) major);
        Files.write(f, forge.build());
        assertInstanceOf(UnsupportedException.class, refused(f, "major " + major));
        assertInstanceOf(UnsupportedException.class, refused(f, false, "major " + major));
        try {
            SegmentReader.inspect(f);
            throw new AssertionError("inspect returned for major " + major);
        } catch (UnsupportedException expected) {
            assertTrue(expected.getMessage().contains("major version " + major), expected.getMessage());
        }
    }

    @ParameterizedTest(name = "minor {0}")
    @ValueSource(ints = {1, 2, 255, 65535})
    void aLargerMinorVersionOpensAndServesTheSameGraph(int minor) throws IOException {
        SegmentForge forge = forge();
        forge.headBuf().putShort(SegmentFormat.H_MINOR, (short) minor);
        Files.write(f, forge.build());
        assertEquals(minor, SegmentReader.inspect(f).minor());
        assertServedLikeTheOriginal(f);
    }

    @Test
    void aLargerMinorMayAppendHeaderFieldsAndFooterMetadata() throws IOException {
        // the header length says where the header CRC is, the offset of the first section entry says where the table starts
        SegmentForge forge = forge();
        ByteBuffer h = forge.headBuf();
        h.putShort(SegmentFormat.H_MINOR, (short) 3);
        h.putInt(SegmentFormat.H_LEN, 60);
        h.putInt(52, 0x5EED5EED); // a field of the newer minor; the old CRC slot (48 to 51) is a field now, the CRC moves to 56
        forge.meta = java.util.Arrays.copyOf(forge.meta, 96); // 32 bytes of newer footer metadata
        for (int i = SegmentFormat.F_META_LEN; i < 96; i++) forge.meta[i] = (byte) (i * 7);
        Files.write(f, forge.build());
        SegmentInfo si = SegmentReader.inspect(f);
        assertEquals(3, si.minor());
        assertEquals(12, si.sections().size());
        assertServedLikeTheOriginal(f);
        // and the header CRC that moved is still checked
        byte[] b = Files.readAllBytes(f);
        b[53] ^= 0x01;
        assertEquals("HEADER", assertInstanceOf(CorruptException.class, refused(f, b, "newer header field")).section());
    }

    @ParameterizedTest(name = "unknown optional section id {0}")
    @ValueSource(ints = {13, 900, 4096, 65535})
    void aLargerMinorWithAnUnknownOptionalSectionIsOpenedAndTheSectionIsIgnored(int id) throws IOException {
        SegmentForge forge = forge();
        forge.headBuf().putShort(SegmentFormat.H_MINOR, (short) 1);
        SegmentForge.Entry e = forge.appendSection(id, 0, 1, new byte[5000]);
        Files.write(f, forge.build());
        SegmentInfo si = SegmentReader.inspect(f);
        assertNotNull(si.section(id));
        assertEquals(false, si.section(id).required());
        assertEquals(13, si.sections().size());
        assertServedLikeTheOriginal(f);
        // the reader does not look at a section it does not know, so damage there is not an integrity failure of what it serves
        byte[] b = Files.readAllBytes(f);
        b[(int) (e.offset + 2500)] ^= 0x20;
        Files.write(f, b);
        assertServedLikeTheOriginal(f);
    }

    @ParameterizedTest(name = "unknown required section id {0}, minor {1}")
    @MethodSource("requiredUnknown")
    void anUnknownRequiredSectionIsRefusedAsUnsupportedEvenWithValidChecksums(int id, int minor) throws IOException {
        SegmentForge forge = forge();
        forge.headBuf().putShort(SegmentFormat.H_MINOR, (short) minor);
        forge.appendSection(id, SegmentFormat.FLAG_REQUIRED, 1, new byte[100]);
        Files.write(f, forge.build());
        for (boolean verify : new boolean[] {true, false}) {
            UnsupportedException e = assertInstanceOf(UnsupportedException.class, refused(f, verify, "required section " + id));
            assertTrue(e.getMessage().contains("required section"), e.getMessage());
        }
        try {
            SegmentReader.inspect(f);
            throw new AssertionError("inspect returned");
        } catch (UnsupportedException expected) {
            // refused in constant time, before any section is read
        }
    }

    static Stream<Arguments> requiredUnknown() {
        return Stream.of(Arguments.of(13, 1), Arguments.of(901, 1), Arguments.of(901, 0), Arguments.of(901, 9), Arguments.of(65535, 1));
    }

    // ---- the byte-order probe ----

    @Test
    void aProbeThatReadsBackSwappedIsRefusedAsUnsupported() throws IOException {
        SegmentForge forge = forge();
        forge.headBuf().putInt(SegmentFormat.H_PROBE, 0x04030201);
        Files.write(f, forge.build());
        UnsupportedException e = assertInstanceOf(UnsupportedException.class, refused(f, "swapped probe"));
        assertTrue(e.getMessage().contains("byte-order"), e.getMessage());
        assertInstanceOf(UnsupportedException.class, refused(f, false, "swapped probe"));
    }

    @ParameterizedTest(name = "probe bit {0}")
    @MethodSource("thirtyTwoBits")
    void everySingleBitOfTheProbeIsRefusedAsUnsupportedWithValidChecksums(int bit) throws IOException {
        SegmentForge forge = forge();
        forge.headBuf().putInt(SegmentFormat.H_PROBE, SegmentFormat.ENDIAN_PROBE ^ (1 << bit));
        Files.write(f, forge.build());
        assertInstanceOf(UnsupportedException.class, refused(f, "probe bit " + bit));
    }

    static IntStream thirtyTwoBits() {
        return IntStream.range(0, 32);
    }

    @Test
    void aHeaderWrittenInBigEndianIsRefusedAsUnsupported() throws IOException {
        // every multi-byte field of the header written the other way round, and a CRC that matches the bytes
        SegmentForge forge = forge();
        ByteBuffer le = forge.headBuf();
        byte[] be = new byte[SegmentFormat.HEADER_BLOCK];
        ByteBuffer b = ByteBuffer.wrap(be).order(ByteOrder.BIG_ENDIAN);
        System.arraycopy(forge.head, 0, be, 0, 8);
        b.putShort(8, le.getShort(8)).putShort(10, le.getShort(10)).putInt(12, le.getInt(12)).putInt(16, le.getInt(16));
        b.putInt(20, le.getInt(20)).putInt(24, le.getInt(24)).putInt(28, le.getInt(28)).putLong(32, le.getLong(32));
        System.arraycopy(forge.head, 40, be, 40, 4);
        b.putInt(44, le.getInt(44));
        b.putInt(48, SegmentFormat.crc32c(be, 0, 48));
        System.arraycopy(be, 0, forge.head, 0, SegmentFormat.HEADER_LEN);
        forge.fixHeaderCrc = false;
        Files.write(f, forge.build());
        assertEquals(0x0100, ByteBuffer.wrap(forge.head).order(ByteOrder.LITTLE_ENDIAN).getShort(8), "the major version reads as 256 in little-endian");
        // the first thing the reader trusts is the major version, and a big-endian writer's 1 reads as 256
        UnsupportedException e = assertInstanceOf(UnsupportedException.class, refused(f, "big-endian header"));
        assertTrue(e.getMessage().contains("major version 256"), e.getMessage());
    }

    // ---- the header fields, with a valid header CRC ----

    static Stream<Arguments> headerFields() {
        return Stream.of(
                header("row stride differs from the dimension", b -> b.putInt(SegmentFormat.H_STRIDE, DIM + 4), UnsupportedException.class, null),
                header("metric code 3", b -> b.put(SegmentFormat.H_METRIC, (byte) 3), UnsupportedException.class, null),
                header("metric code 255", b -> b.put(SegmentFormat.H_METRIC, (byte) 255), UnsupportedException.class, null),
                header("id width 4", b -> b.put(SegmentFormat.H_IDW, (byte) 4), UnsupportedException.class, null),
                header("layer 0 layout 1", b -> b.put(SegmentFormat.H_LAYOUT, (byte) 1), UnsupportedException.class, null),
                header("dimension zero", b -> b.putInt(SegmentFormat.H_DIM, 0), CorruptException.class, "HEADER"),
                header("dimension above 2^20", b -> b.putInt(SegmentFormat.H_DIM, (1 << 20) + 1), CorruptException.class, "HEADER"),
                header("dimension one above", b -> {
                    b.putInt(SegmentFormat.H_DIM, DIM + 1);
                    b.putInt(SegmentFormat.H_STRIDE, DIM + 1);
                }, CorruptException.class, "VEC"),
                header("row count one above", b -> b.putLong(SegmentFormat.H_N, N + 1), CorruptException.class, "IDS"),
                header("row count zero", b -> b.putLong(SegmentFormat.H_N, 0), CorruptException.class, "IDS"),
                header("row count negative", b -> b.putLong(SegmentFormat.H_N, -1), IOException.class, null),
                header("row count 100 million", b -> b.putLong(SegmentFormat.H_N, 100_000_000L), CorruptException.class, "IDS"),
                header("row count 2^31 - 9", b -> b.putLong(SegmentFormat.H_N, Integer.MAX_VALUE - 9L), IOException.class, null),
                header("row count 2^62", b -> b.putLong(SegmentFormat.H_N, 1L << 62), IOException.class, null),
                header("alignment 3000, not a power of two", b -> b.putInt(SegmentFormat.H_ALIGN, 3000), CorruptException.class, "HEADER"),
                header("alignment 4", b -> b.putInt(SegmentFormat.H_ALIGN, 4), CorruptException.class, "HEADER"),
                header("alignment 2^31", b -> b.putInt(SegmentFormat.H_ALIGN, 1 << 31), CorruptException.class, "HEADER"),
                header("alignment 8192 while the sections start at multiples of 4096", b -> b.putInt(SegmentFormat.H_ALIGN, 8192), CorruptException.class, "IDS"),
                header("header length 51", b -> b.putInt(SegmentFormat.H_LEN, 51), CorruptException.class, "HEADER"),
                header("header length 4097", b -> b.putInt(SegmentFormat.H_LEN, 4097), CorruptException.class, "HEADER"),
                header("header length 2^32 - 1", b -> b.putInt(SegmentFormat.H_LEN, -1), CorruptException.class, "HEADER"));
    }

    private static Arguments header(String name, java.util.function.Consumer<ByteBuffer> change, Class<? extends IOException> type, String section) {
        return Arguments.of(name, change, type, section);
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("headerFields")
    void aHeaderFieldThatThisReaderCannotUseIsRefusedWithValidChecksums(String name, java.util.function.Consumer<ByteBuffer> change,
            Class<? extends IOException> type, String section) throws IOException {
        SegmentForge forge = forge();
        change.accept(forge.headBuf());
        Files.write(f, forge.build());
        IOException e = refused(f, name);
        assertInstanceOf(type, e, name);
        if (section != null) assertEquals(section, ((CorruptException) e).section(), name);
    }

    // ---- the section table, the footer metadata and the locator, with valid CRCs ----

    static Stream<Arguments> structure() {
        final int footerMeta = SegmentFormat.F_META_LEN;
        return Stream.of(
                table("section offset not aligned", fg -> fg.entry(SegmentFormat.S_IDS).offset += 8, "IDS"),
                table("section offset inside the header block", fg -> fg.entry(SegmentFormat.S_IDS).offset = 0, "IDS"),
                table("two sections swapped in the file order", fg -> {
                    long a = fg.entries.get(1).offset;
                    fg.entries.get(1).offset = fg.entries.get(2).offset;
                    fg.entries.get(2).offset = a;
                }, "BYID"),
                table("a section starts where the one before it starts", fg -> fg.entry(SegmentFormat.S_BYID).offset = fg.entry(SegmentFormat.S_VERS).offset, "BYID"),
                table("a section runs past the footer", fg -> fg.entry(SegmentFormat.S_SQ8_PARAMS).offset += 1 << 20, "SQ8_PARAMS"),
                table("length is not width times count", fg -> fg.entry(SegmentFormat.S_VEC).length += 4, "VEC"),
                table("element width zero", fg -> fg.entry(SegmentFormat.S_IDS).width = 0, "IDS"),
                table("element width above 8", fg -> {
                    SegmentForge.Entry e = fg.entry(SegmentFormat.S_IDS);
                    e.width = 16;
                    e.count = e.length / 16;
                }, "IDS"),
                table("element width 4 for an i64 section", fg -> {
                    SegmentForge.Entry e = fg.entry(SegmentFormat.S_IDS);
                    e.width = 4;
                    e.count = e.length / 4;
                }, "IDS"),
                table("element count one short with a matching length", fg -> {
                    SegmentForge.Entry e = fg.entry(SegmentFormat.S_IDS);
                    e.count = N - 1;
                    e.length = 8L * (N - 1);
                }, "IDS"),
                table("negative element count", fg -> fg.entry(SegmentFormat.S_IDS).count = -1, "IDS"),
                table("negative section length", fg -> fg.entry(SegmentFormat.S_IDS).length = -8, "IDS"),
                table("SQ8 codes without their parameters", fg -> fg.entries.remove(fg.entry(SegmentFormat.S_SQ8_PARAMS)), "SQ8_PARAMS"),
                table("SQ8 parameters without their codes", fg -> fg.entries.remove(fg.entry(SegmentFormat.S_SQ8_CODES)), "SQ8_PARAMS"),
                table("a section listed twice", fg -> fg.appendSection(SegmentFormat.S_IDS, SegmentFormat.FLAG_REQUIRED, 8, new byte[8 * N]), "IDS"),
                footer("more entries than the footer holds", fg -> fg.metaBuf().putInt(SegmentFormat.F_COUNT, fg.entries.size() + 1), "FOOTER"),
                footer("more than 4096 entries", fg -> fg.metaBuf().putInt(SegmentFormat.F_COUNT, 5000), "FOOTER"),
                footer("entry count 2^32 - 1", fg -> fg.metaBuf().putInt(SegmentFormat.F_COUNT, -1), "FOOTER"),
                footer("first entry inside the metadata", fg -> fg.metaBuf().putInt(SegmentFormat.F_ENTRIES_OFF, footerMeta - 4), "FOOTER"),
                footer("first entry beyond the footer", fg -> fg.metaBuf().putInt(SegmentFormat.F_ENTRIES_OFF, 100_000), "FOOTER"),
                footer("entry row equal to the row count", fg -> fg.metaBuf().putInt(SegmentFormat.F_ENTRY_ROW, N), "FOOTER"),
                footer("negative entry row", fg -> fg.metaBuf().putInt(SegmentFormat.F_ENTRY_ROW, -1), "FOOTER"),
                footer("top level 128", fg -> fg.metaBuf().putInt(SegmentFormat.F_MAX_LEVEL, 128), "FOOTER"),
                footer("negative top level", fg -> fg.metaBuf().putInt(SegmentFormat.F_MAX_LEVEL, -1), "FOOTER"),
                loc("footer offset one too high", fg -> fg.locFooterOff = 4096L + fg.body.length + 1, "LOCATOR"),
                loc("footer offset inside the header block", fg -> fg.locFooterOff = 100L, "LOCATOR"),
                loc("footer offset zero", fg -> fg.locFooterOff = 0L, "LOCATOR"),
                loc("footer length one too long", fg -> fg.locFooterLen = fg.meta.length + 40L * fg.entries.size() + 1, "LOCATOR"),
                loc("footer length zero", fg -> fg.locFooterLen = 0L, "LOCATOR"),
                loc("footer length above one MiB", fg -> fg.locFooterLen = (1L << 20) + 1, "LOCATOR"),
                loc("file length one too long", fg -> fg.locFileLen = 4096L + fg.body.length + fg.meta.length + 40L * fg.entries.size() + 33, "LOCATOR"),
                loc("file length zero", fg -> fg.locFileLen = 0L, "LOCATOR"),
                loc("file length 2^63 - 1", fg -> fg.locFileLen = Long.MAX_VALUE, "LOCATOR"),
                data("a row listed twice in the id order", fg -> {
                    byte[] a = fg.sectionBytes(SegmentFormat.S_BYID);
                    System.arraycopy(a, 0, a, 4, 4); // row at position 1 := row at position 0
                    fg.setSectionBytes(SegmentFormat.S_BYID, a);
                }, "BYID"),
                data("a row number outside the rows in the id order", fg -> {
                    byte[] a = fg.sectionBytes(SegmentFormat.S_BYID);
                    SegmentForge.le(a).putInt(0, N);
                    fg.setSectionBytes(SegmentFormat.S_BYID, a);
                }, "BYID"),
                data("a negative row number in the id order", fg -> {
                    byte[] a = fg.sectionBytes(SegmentFormat.S_BYID);
                    SegmentForge.le(a).putInt(4 * (N - 1), -1);
                    fg.setSectionBytes(SegmentFormat.S_BYID, a);
                }, "BYID"),
                data("layer 0 offsets that end above the edge count", fg -> {
                    byte[] a = fg.sectionBytes(SegmentFormat.S_L0_OFF);
                    ByteBuffer b = SegmentForge.le(a);
                    b.putInt(4 * N, b.getInt(4 * N) + 1);
                    fg.setSectionBytes(SegmentFormat.S_L0_OFF, a);
                }, "L0_OFF"),
                data("layer 0 offsets that do not start at zero", fg -> {
                    byte[] a = fg.sectionBytes(SegmentFormat.S_L0_OFF);
                    SegmentForge.le(a).putInt(0, 1);
                    fg.setSectionBytes(SegmentFormat.S_L0_OFF, a);
                }, "L0_OFF"));
    }

    private static Arguments table(String name, java.util.function.Consumer<SegmentForge> change, String section) {
        return Arguments.of("table: " + name, change, section);
    }

    private static Arguments footer(String name, java.util.function.Consumer<SegmentForge> change, String section) {
        return Arguments.of("footer: " + name, (java.util.function.Consumer<SegmentForge>) fg -> {
            fg.syncMeta = false;
            change.accept(fg);
        }, section);
    }

    private static Arguments loc(String name, java.util.function.Consumer<SegmentForge> change, String section) {
        return Arguments.of("locator: " + name, change, section);
    }

    private static Arguments data(String name, java.util.function.Consumer<SegmentForge> change, String section) {
        return Arguments.of("data: " + name, change, section);
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("structure")
    void aStructureThatBreaksTheDocumentedRulesIsRefusedWithValidChecksums(String name, java.util.function.Consumer<SegmentForge> change,
            String section) throws IOException {
        SegmentForge forge = forge();
        change.accept(forge);
        Files.write(f, forge.build());
        CorruptException e = assertInstanceOf(CorruptException.class, refused(f, name), name);
        assertEquals(section, e.section(), name);
        // the checks of the table, the footer and the locator also hold without the section CRCs; the data checks need the sections loaded
        if (!name.startsWith("data:")) assertEquals(section, assertInstanceOf(CorruptException.class, refused(f, false, name), name).section(), name);
    }

    @ParameterizedTest(name = "required section id {0} missing")
    @ValueSource(ints = {1, 3, 4, 5, 6, 7, 8, 9, 10})
    void aMissingRequiredSectionIsRefusedWithValidChecksums(int id) throws IOException {
        SegmentForge forge = forge();
        forge.entries.remove(forge.entry(id));
        Files.write(f, forge.build());
        for (boolean verify : new boolean[] {true, false}) {
            CorruptException e = assertInstanceOf(CorruptException.class, refused(f, verify, "section " + id + " missing"));
            assertEquals(SegmentFormat.sectionName(id), e.section());
            assertTrue(e.getMessage().contains("required section is missing"), e.getMessage());
        }
    }

    @Test
    void theOptionalSectionsMayBeAbsentTogether() throws IOException {
        // SQ8 codes and parameters are optional as a pair; the version section is optional too (a standalone graph has none)
        SegmentForge forge = forge();
        forge.entries.remove(forge.entry(SegmentFormat.S_SQ8_CODES));
        forge.entries.remove(forge.entry(SegmentFormat.S_SQ8_PARAMS));
        forge.entries.remove(forge.entry(SegmentFormat.S_VERS));
        Files.write(f, forge.build());
        SegmentReader.Loaded l = SegmentReader.readFull(f, true);
        assertEquals(null, l.vers());
        assertEquals(null, l.graph().codes);
        assertArrayEquals(orig.ids, l.graph().ids);
    }
}
