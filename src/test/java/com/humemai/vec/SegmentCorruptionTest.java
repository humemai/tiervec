package com.humemai.vec;

import static com.humemai.vec.SegmentForge.refused;
import static com.humemai.vec.SegmentTestSupport.graph;
import static com.humemai.vec.SegmentTestSupport.params;
import static com.humemai.vec.SegmentTestSupport.vectors;
import static com.humemai.vec.SegmentTestSupport.versOf;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.humemai.vec.SegmentFormat.CorruptException;
import com.humemai.vec.SegmentFormat.UnsupportedException;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Damage to a segment file, byte by byte. A file that is cut short, or has any bit of its header, footer, locator or sections flipped, is
 * refused with a {@link CorruptException} (an {@link UnsupportedException} where the damaged bytes are the major version; for the magic either
 * of the two named exceptions), never loaded, and never fails in any other way. The sweeps are exhaustive over a small file: every cut length, and every byte that a
 * CRC or a length check covers. The only bytes whose damage is not detected are the zero fill that no CRC covers (the rest of the header
 * block and the gaps between sections), and a test below shows that damage there changes nothing a reader serves.
 *
 * <p>The code reads files with plain {@code FileChannel} reads into heap arrays (no memory mapping and no native memory in this version),
 * so a damaged file cannot take the JVM down; what the tests rule out is every other uncontrolled failure: an exception that is not one of
 * the two named ones, an out-of-memory error from a length read out of the file, or a graph that is returned.
 */
class SegmentCorruptionTest {
    private static final int N = 300, DIM = 4;

    @TempDir
    Path tmp;

    private SealedGraph orig;
    private byte[] good;
    private SegmentInfo info;
    private Path scratch;

    @BeforeEach
    void write() throws IOException {
        orig = graph(vectors(N, DIM, 17), Metric.L2, params(true)); // SQ8 on: all twelve sections are in the file
        Path f = tmp.resolve("good.tvs");
        info = SegmentWriter.write(orig, versOf(N), f);
        good = Files.readAllBytes(f);
        scratch = tmp.resolve("scratch.tvs");
        Files.write(scratch, good);
        assertEquals(12, info.sections().size());
        for (SegmentInfo.Section s : info.sections()) assertTrue(s.length() > 0, "section " + SegmentFormat.sectionName(s.id()) + " has bytes to damage");
    }

    private int footerOff() {
        return good.length - SegmentFormat.LOCATOR_LEN - (SegmentFormat.F_META_LEN + SegmentFormat.ENTRY_LEN * info.sections().size());
    }

    /** flips one bit of the scratch file in place; returns the byte to give back to {@link #restore} */
    private byte flip(FileChannel ch, int pos, int bit) throws IOException {
        byte old = good[pos];
        ch.write(ByteBuffer.wrap(new byte[] {(byte) (old ^ (1 << bit))}), pos);
        return old;
    }

    private void restore(FileChannel ch, int pos) throws IOException {
        ch.write(ByteBuffer.wrap(new byte[] {good[pos]}), pos);
    }

    // ---- truncation ----

    @Test
    void everyCutLengthIsRefusedWithACorruptException() throws IOException {
        // the scratch file is cut one byte at a time from its full length down to nothing; a cut inside the header block leaves too little for a
        // segment, every other cut leaves a file whose last 32 bytes are not the locator
        try (FileChannel ch = FileChannel.open(scratch, StandardOpenOption.WRITE)) {
            for (int len = good.length - 1; len >= 0; len--) {
                ch.truncate(len);
                CorruptException e = assertInstanceOf(CorruptException.class, refused(scratch, "cut at " + len), "cut at " + len);
                assertEquals(len < SegmentFormat.HEADER_BLOCK + SegmentFormat.LOCATOR_LEN ? "HEADER" : "LOCATOR", e.section(), "cut at " + len);
            }
        }
        assertEquals(0, Files.size(scratch));
    }

    @ParameterizedTest(name = "section id {0}")
    @ValueSource(ints = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
    void aFileCutInTheMiddleOfASectionOrAtItsLastByteFailsInConstantTime(int id) throws IOException {
        SegmentInfo.Section sec = info.section(id);
        for (long cut : new long[] {sec.offset() + sec.length() / 2, sec.offset() + sec.length() - 1}) {
            Path p = tmp.resolve("cut-" + id + "-" + cut + ".tvs");
            Files.write(p, java.util.Arrays.copyOf(good, (int) cut));
            // inspect reads the header block and the last 32 bytes only, so no section is read
            CorruptException e = assertInstanceOf(CorruptException.class, assertThrowsIo(() -> SegmentReader.inspect(p), "cut at " + cut));
            assertEquals("LOCATOR", e.section());
            assertEquals("LOCATOR", assertInstanceOf(CorruptException.class, refused(p, false, "cut at " + cut)).section());
            assertEquals("LOCATOR", assertInstanceOf(CorruptException.class, refused(p, true, "cut at " + cut)).section());
        }
    }

    @Test
    void aFileCutInsideTheFooterOrTheLocatorOrAtTheEndOfTheSectionsIsRefused() throws IOException {
        final int footerOff = footerOff();
        final int[] cuts = {footerOff, footerOff + 1, footerOff + SegmentFormat.F_META_LEN, good.length - SegmentFormat.LOCATOR_LEN - 1,
            good.length - SegmentFormat.LOCATOR_LEN, good.length - SegmentFormat.LOCATOR_LEN + 1, good.length - 8, good.length - 1};
        for (int cut : cuts) {
            Path p = tmp.resolve("tail-" + cut + ".tvs");
            Files.write(p, java.util.Arrays.copyOf(good, cut));
            assertEquals("LOCATOR", assertInstanceOf(CorruptException.class, refused(p, "cut at " + cut)).section(), "cut at " + cut);
        }
    }

    @Test
    void bytesAppendedToAGoodFileAreRefused() throws IOException {
        for (int extra : new int[] {1, 7, 32, 4096}) {
            byte[] b = java.util.Arrays.copyOf(good, good.length + extra); // zeros after the locator: it is no longer the last 32 bytes
            Path p = tmp.resolve("appended-" + extra + ".tvs");
            Files.write(p, b);
            assertEquals("LOCATOR", assertInstanceOf(CorruptException.class, refused(p, extra + " bytes appended")).section());
        }
    }

    // ---- bit flips ----

    @Test
    void everyBitOfTheHeaderIsGuarded() throws IOException {
        // bytes 0 to 9 are the magic and the major version: a flipped bit there names a file this reader does not understand. Bytes 10 to 51
        // (minor version, header length, probe, flags, dimensions, enum codes, alignment, the CRC itself) are covered by the header CRC.
        try (FileChannel ch = FileChannel.open(scratch, StandardOpenOption.WRITE)) {
            for (int pos = 0; pos < SegmentFormat.HEADER_LEN; pos++) {
                for (int bit = 0; bit < 8; bit++) {
                    flip(ch, pos, bit);
                    String what = "bit " + bit + " of header byte " + pos;
                    for (boolean verify : new boolean[] {true, false}) {
                        IOException e = refused(scratch, verify, what);
                        if (pos >= 8 && pos < 10) assertInstanceOf(UnsupportedException.class, e, what);
                        else if (pos >= 10) assertEquals("HEADER", assertInstanceOf(CorruptException.class, e, what).section(), what);
                    }
                    restore(ch, pos);
                }
            }
        }
    }

    @Test
    void everyBitOfTheFooterIsGuarded() throws IOException {
        // the footer CRC in the locator covers the metadata and every section entry. The section table, with its offsets, widths, counts and
        // CRCs, therefore cannot change without notice, and the check runs before any section is read (verifyAll false below).
        final int footerOff = footerOff(), footerEnd = good.length - SegmentFormat.LOCATOR_LEN;
        try (FileChannel ch = FileChannel.open(scratch, StandardOpenOption.WRITE)) {
            for (int pos = footerOff; pos < footerEnd; pos++) {
                for (int bit = 0; bit < 8; bit++) {
                    flip(ch, pos, bit);
                    String what = "bit " + bit + " of footer byte " + (pos - footerOff);
                    for (boolean verify : new boolean[] {true, false}) {
                        assertEquals("FOOTER", assertInstanceOf(CorruptException.class, refused(scratch, verify, what), what).section(), what);
                    }
                    restore(ch, pos);
                }
            }
        }
    }

    @Test
    void everyBitOfTheLocatorIsGuarded() throws IOException {
        // magic (0 to 7), footer offset (8 to 15), footer length (16 to 19) and file length (24 to 31) are checked against the file itself; the
        // footer CRC (20 to 23) is checked against the footer
        final int locOff = good.length - SegmentFormat.LOCATOR_LEN;
        try (FileChannel ch = FileChannel.open(scratch, StandardOpenOption.WRITE)) {
            for (int rel = 0; rel < SegmentFormat.LOCATOR_LEN; rel++) {
                for (int bit = 0; bit < 8; bit++) {
                    flip(ch, locOff + rel, bit);
                    String what = "bit " + bit + " of locator byte " + rel;
                    for (boolean verify : new boolean[] {true, false}) {
                        CorruptException e = assertInstanceOf(CorruptException.class, refused(scratch, verify, what), what);
                        assertEquals(rel >= 20 && rel < 24 ? "FOOTER" : "LOCATOR", e.section(), what);
                    }
                    restore(ch, locOff + rel);
                }
            }
        }
    }

    @Test
    void everyByteOfEverySectionIsCoveredByItsCrc() throws IOException {
        // every byte of every section has one bit flipped (the bit changes from byte to byte), and the first, the middle and the last byte of each
        // section have all eight flipped. The metadata is intact in all of these, so the CRC of exactly the damaged section must be what fails.
        try (FileChannel ch = FileChannel.open(scratch, StandardOpenOption.WRITE)) {
            for (SegmentInfo.Section sec : info.sections()) {
                final String name = SegmentFormat.sectionName(sec.id());
                final int first = (int) sec.offset(), last = (int) (sec.offset() + sec.length() - 1), mid = (int) (sec.offset() + sec.length() / 2);
                for (int pos = first; pos <= last; pos++) {
                    boolean all = pos == first || pos == mid || pos == last;
                    for (int bit = all ? 0 : pos % 8; bit < (all ? 8 : pos % 8 + 1); bit++) {
                        flip(ch, pos, bit);
                        String what = "bit " + bit + " of byte " + (pos - first) + " of section " + name;
                        CorruptException e = assertInstanceOf(CorruptException.class, refused(scratch, what), what);
                        assertEquals(name, e.section(), what);
                        restore(ch, pos);
                    }
                }
            }
        }
    }

    @Test
    void aFlippedSectionBitIsFoundOnlyWhenTheSectionIsChecksummed() throws IOException {
        // the contract of read(file, false): header, footer and locator are checked, the section bytes are not. A host that wants every byte
        // checked uses SealedGraph.load or TieredIndex.open, which always do.
        SegmentInfo.Section vec = info.section(SegmentFormat.S_VEC);
        try (FileChannel ch = FileChannel.open(scratch, StandardOpenOption.WRITE)) {
            flip(ch, (int) (vec.offset() + 5), 3);
        }
        SegmentReader.read(scratch, false); // loads: the flip is in a float of the vectors
        assertEquals("VEC", assertInstanceOf(CorruptException.class, refused(scratch, "bit in VEC")).section());
        assertEquals("VEC", assertInstanceOf(CorruptException.class, assertThrowsIo(() -> SealedGraph.load(scratch), "SealedGraph.load")).section());
    }

    // ---- the zero fill that no CRC covers ----

    @Test
    void damageInTheZeroFillIsNotDetectedAndChangesNothingThatIsServed() throws IOException {
        // the doc says so: header bytes 52 to 4095 and the gaps between sections are zero and covered by no CRC. A flip there is not an
        // integrity failure because no reader looks at those bytes; the test pins that the loaded graph is the same one.
        List<int[]> gaps = new ArrayList<>(); // [from, to)
        gaps.add(new int[] {SegmentFormat.HEADER_LEN, SegmentFormat.HEADER_BLOCK});
        for (int i = 0; i + 1 < info.sections().size(); i++) {
            SegmentInfo.Section s = info.sections().get(i), next = info.sections().get(i + 1);
            if (s.offset() + s.length() < next.offset()) gaps.add(new int[] {(int) (s.offset() + s.length()), (int) next.offset()});
        }
        SegmentInfo.Section lastSec = info.sections().get(info.sections().size() - 1);
        gaps.add(new int[] {(int) (lastSec.offset() + lastSec.length()), footerOff()});
        assertTrue(gaps.size() >= 12, "the gaps found: " + gaps.size());
        float[][] qs = vectors(30, DIM, 99);
        try (FileChannel ch = FileChannel.open(scratch, StandardOpenOption.WRITE)) {
            for (int[] g : gaps) {
                for (int pos : new int[] {g[0], (g[0] + g[1]) / 2, g[1] - 1}) {
                    if (pos < g[0] || pos >= g[1]) continue;
                    for (int bit : new int[] {0, 7}) {
                        flip(ch, pos, bit);
                        SealedGraph h = SealedGraph.load(scratch);
                        assertArrayEquals(orig.ids, h.ids, "zero fill byte " + pos);
                        assertArrayEquals(orig.vec, h.vec, "zero fill byte " + pos);
                        for (float[] q : qs) assertArrayEquals(orig.search(q, 10, null), h.search(q, 10, null), "zero fill byte " + pos);
                        restore(ch, pos);
                    }
                }
            }
        }
        // and the gaps really are zero in a file the writer wrote
        for (int[] g : gaps) {
            for (int pos = g[0]; pos < g[1]; pos++) assertEquals(0, good[pos], "byte " + pos + " of the fill");
        }
    }

    // ---- wrong magic and files that are no segment ----

    @Test
    void aWrongMagicIsRefusedWhicheverByteIsWrong() throws IOException {
        for (int pos = 0; pos < SegmentFormat.MAGIC.length; pos++) {
            for (int x : new int[] {0x01, 0x80, 0xFF}) {
                byte[] b = good.clone();
                b[pos] ^= (byte) x;
                Path p = tmp.resolve("magic-" + pos + "-" + x + ".tvs");
                // the magic is checked before any CRC; which of the two named exceptions says "not a segment file" is not part of the claim
                refused(p, b, "magic byte " + pos + " ^ " + x);
            }
        }
        byte[] zeroed = good.clone();
        java.util.Arrays.fill(zeroed, 0, 8, (byte) 0);
        refused(tmp.resolve("zeroed-magic.tvs"), zeroed, "zeroed magic");
        // the magic of a manifest, a tombstone file and the locator are not the magic of a segment
        for (String other : new String[] {"TVMAN001", "TVTOMB01", "TVSEGEND", "tvseg001", "TVSEG002"}) {
            byte[] b = good.clone();
            System.arraycopy(other.getBytes(java.nio.charset.StandardCharsets.US_ASCII), 0, b, 0, 8);
            refused(tmp.resolve("other-" + other + ".tvs"), b, other);
        }
    }

    @Test
    void filesThatAreNoSegmentAtAllAreRefused() throws IOException {
        Random r = new Random(5);
        byte[] noise = new byte[100_000];
        r.nextBytes(noise);
        refused(tmp.resolve("noise.tvs"), noise, "random bytes");
        byte[] text = "this is a text file, not a segment, but it is longer than the header".getBytes(java.nio.charset.StandardCharsets.US_ASCII);
        refused(tmp.resolve("text.tvs"), text, "text");
        // shorter than the magic: too short to be anything
        for (int len : new int[] {0, 1, 7}) {
            assertInstanceOf(CorruptException.class, refused(tmp.resolve("short" + len + ".tvs"), new byte[len], len + " bytes"), len + " bytes");
        }
        // zeros: no magic
        refused(tmp.resolve("zeros.tvs"), new byte[HEADER_AND_LOCATOR + 10], "zeros");
        // the right magic followed by nothing, and by random bytes
        byte[] magicOnly = java.util.Arrays.copyOf(SegmentFormat.MAGIC, 8);
        assertInstanceOf(CorruptException.class, refused(tmp.resolve("magic-only.tvs"), magicOnly, "magic only"));
        byte[] magicNoise = noise.clone();
        System.arraycopy(SegmentFormat.MAGIC, 0, magicNoise, 0, 8);
        refused(tmp.resolve("magic-noise.tvs"), magicNoise, "magic then random bytes"); // some named exception, never a graph
    }

    private static final int HEADER_AND_LOCATOR = SegmentFormat.HEADER_BLOCK + SegmentFormat.LOCATOR_LEN;

    private interface IoCall {
        void run() throws IOException;
    }

    private static IOException assertThrowsIo(IoCall c, String what) {
        try {
            c.run();
        } catch (IOException e) {
            return e;
        }
        throw new AssertionError(what + ": no exception");
    }
}
