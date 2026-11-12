package ai.humem.tiervec;

import static ai.humem.tiervec.SegmentTestSupport.exactTopK;
import static ai.humem.tiervec.SegmentTestSupport.graph;
import static ai.humem.tiervec.SegmentTestSupport.idSet;
import static ai.humem.tiervec.SegmentTestSupport.params;
import static ai.humem.tiervec.SegmentTestSupport.recall;
import static ai.humem.tiervec.SegmentTestSupport.vectors;
import static ai.humem.tiervec.SegmentTestSupport.versOf;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.humem.tiervec.SegmentFormat.CorruptException;
import ai.humem.tiervec.SegmentFormat.UnsupportedException;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

/** The segment file: round trip, section damage, versioning, and sizes against the in-memory counter. */
class SegmentFormatTest {
    @TempDir
    Path tmp;

    private int counter;

    private Path file(String name) {
        return tmp.resolve(name);
    }

    private Path write(byte[] bytes) throws IOException {
        Path p = file("damaged" + (counter++) + ".tvs");
        Files.write(p, bytes);
        return p;
    }

    private static void assertSameArrays(SealedGraph g, SealedGraph h) {
        assertEquals(g.dim, h.dim);
        assertEquals(g.metric, h.metric);
        assertEquals(g.params, h.params);
        assertEquals(g.n, h.n);
        assertEquals(g.entryRow, h.entryRow);
        assertEquals(g.maxLevel, h.maxLevel);
        assertArrayEquals(g.ids, h.ids);
        assertArrayEquals(g.vec, h.vec);
        assertArrayEquals(g.level, h.level);
        assertArrayEquals(g.off0, h.off0);
        assertArrayEquals(g.edges0, h.edges0);
        assertArrayEquals(g.upIdx, h.upIdx);
        assertArrayEquals(g.upOff, h.upOff);
        assertArrayEquals(g.upEdges, h.upEdges);
        assertArrayEquals(g.sortedIds, h.sortedIds);
        assertArrayEquals(g.sortedRows, h.sortedRows);
        assertEquals(g.codes == null, h.codes == null);
        if (g.codes != null) {
            assertArrayEquals(g.codes, h.codes);
            assertArrayEquals(g.qbase, h.qbase);
            assertArrayEquals(g.qdelta, h.qdelta);
            assertArrayEquals(g.qnorm, h.qnorm);
        }
    }

    // ---- U1: round trip ----

    @ParameterizedTest(name = "{0} sq8={1}")
    @CsvSource({"L2,false", "COSINE,false", "DOT,false", "L2,true", "COSINE,true"})
    void roundTrip(Metric metric, boolean sq8) throws IOException {
        final int n = 1500, dim = 16;
        float[][] data = vectors(n, dim, 11);
        SealedGraph g = graph(data, metric, params(sq8));
        assertEquals(sq8, g.codes != null);
        long[] vers = versOf(n);
        Path f = file("a.tvs");
        SegmentInfo wrote = SegmentWriter.write(g, vers, f);
        assertEquals(wrote, SegmentReader.inspect(f));
        SegmentReader.Loaded l = SegmentReader.readFull(f, true);
        assertEquals(wrote, l.info());
        assertArrayEquals(vers, l.vers());
        assertSameArrays(g, l.graph());
        assertEquals(g.bytesHeld(), l.graph().bytesHeld());

        float[][] qs = vectors(200, dim, 99);
        int sameSets = 0;
        double recallG = 0, recallH = 0;
        for (float[] q : qs) {
            Hit[] a = g.search(q, 10, null);
            Hit[] b = l.graph().search(q, 10, null);
            if (idSet(a).equals(idSet(b))) sameSets++;
            Set<Long> exact = exactTopK(data, q, 10, metric);
            recallG += recall(a, exact);
            recallH += recall(b, exact);
        }
        assertTrue(sameSets >= 198, "top-10 sets equal for " + sameSets + " of 200 queries");
        assertEquals(Math.round(recallG / qs.length * 1e4), Math.round(recallH / qs.length * 1e4), "recall at four decimals");
    }

    @Test
    void serializeAndLoadOnTheGraph() throws IOException {
        SealedGraph g = graph(vectors(600, 12, 4), Metric.L2, params(false));
        Path f = file("g.tvs");
        g.serialize(f);
        SealedGraph h = SealedGraph.load(f);
        assertSameArrays(g, h);
        assertEquals(g.size(), h.size());
        assertNull(SegmentReader.readFull(f, true).vers(), "a standalone graph file has no version section");
        // the id lookup built from the stored order works: a delete finds its row
        long id = g.idAt(5);
        assertTrue(h.delete(id));
        assertEquals(g.size() - 1, h.size());
        assertTrue(h.isDeleted(5));

        // a segment is never overwritten
        byte[] before = Files.readAllBytes(f);
        assertThrows(FileAlreadyExistsException.class, () -> g.serialize(f));
        assertArrayEquals(before, Files.readAllBytes(f));

        // a graph with deleted rows would come back with them alive: refused, and no file is left
        assertTrue(g.delete(g.idAt(3)));
        Path d = file("deleted.tvs");
        assertThrows(IllegalStateException.class, () -> g.serialize(d));
        assertFalse(Files.exists(d));
    }

    // ---- sizes ----

    @ParameterizedTest(name = "dim {0} sq8={1}")
    @CsvSource({"16,false", "48,false", "16,true", "48,true"})
    void sectionBytesMatchTheInMemoryCounter(int dim, boolean sq8) throws IOException {
        final int n = 600;
        SealedGraph g = graph(vectors(n, dim, 3), Metric.COSINE, params(sq8));
        Path f = file("s.tvs");
        SegmentInfo info = SegmentWriter.write(g, versOf(n), f);
        long payload = 0;
        for (SegmentInfo.Section s : info.sections()) payload += s.length();
        // the file holds VERS and BYID where the counter holds the two sorted-id arrays (the same 12 bytes per row) and no deletion bitmap
        assertEquals(g.bytesHeld() - 8L * ((n + 63) / 64), payload);
        // the file length from the section table: header block, sections at aligned offsets, footer, locator
        long pos = SegmentFormat.HEADER_BLOCK;
        for (SegmentInfo.Section s : info.sections()) pos = SegmentFormat.alignUp(pos) + s.length();
        long footer = SegmentFormat.F_META_LEN + (long) SegmentFormat.ENTRY_LEN * info.sections().size();
        assertEquals(pos + footer + SegmentFormat.LOCATOR_LEN, Files.size(f));
        assertEquals(Files.size(f), info.fileLength());
        for (SegmentInfo.Section s : info.sections()) assertEquals(0, s.offset() % SegmentFormat.ALIGN);
    }

    // ---- U2: every section, damaged or cut ----

    @ParameterizedTest(name = "section id {0}")
    @ValueSource(ints = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
    void damageInASectionIsDetected(int id) throws IOException {
        final int n = 800, dim = 16;
        boolean sq8 = id >= SegmentFormat.S_SQ8_CODES;
        SealedGraph g = graph(vectors(n, dim, 5), Metric.L2, params(sq8));
        Path good = file("good.tvs");
        SegmentInfo info = SegmentWriter.write(g, versOf(n), good);
        SegmentInfo.Section sec = info.section(id);
        assertNotNull(sec, "section " + id + " is in the file");
        assertTrue(sec.length() > 1, "section " + id + " has bytes to damage");
        byte[] bytes = Files.readAllBytes(good);
        final String name = SegmentFormat.sectionName(id);

        // one flipped bit in the middle of the section: the metadata is intact, the CRC of exactly this section fails
        byte[] flipped = bytes.clone();
        flipped[(int) (sec.offset() + sec.length() / 2)] ^= 0x10;
        Path fp = write(flipped);
        SegmentReader.inspect(fp);
        CorruptException e = assertThrows(CorruptException.class, () -> SegmentReader.read(fp, true));
        assertEquals(name, e.section());

        // the file cut in the middle of the section, and one byte before its end
        for (long cut : new long[] {sec.offset() + sec.length() / 2, sec.offset() + sec.length() - 1}) {
            Path tp = write(Arrays.copyOf(bytes, (int) cut));
            assertThrows(CorruptException.class, () -> SegmentReader.inspect(tp));
            assertThrows(CorruptException.class, () -> SegmentReader.read(tp, false));
        }
    }

    // ---- U4: header, footer and locator are checked in constant time ----

    @Test
    void damageToTheMetadataIsDetectedBeforeAnySectionIsRead() throws IOException {
        final int n = 500;
        SealedGraph g = graph(vectors(n, 8, 6), Metric.L2, params(false));
        Path good = file("good.tvs");
        SegmentInfo info = SegmentWriter.write(g, versOf(n), good);
        byte[] bytes = Files.readAllBytes(good);
        final int size = bytes.length;
        final int footerLen = SegmentFormat.F_META_LEN + SegmentFormat.ENTRY_LEN * info.sections().size();
        final int footerOff = size - SegmentFormat.LOCATOR_LEN - footerLen;

        // verifyAll is false in every read below: no section CRC is computed, so only the header, footer and locator checks can fire
        int[] spots = {0, 8, 10, 24, 32, 40, footerOff + 1, footerOff + 20, footerOff + SegmentFormat.F_META_LEN + 5, size - 32, size - 24,
            size - 16, size - 12, size - 1};
        for (int at : spots) {
            byte[] b = bytes.clone();
            b[at] ^= 0x01;
            Path p = write(b);
            assertThrows(IOException.class, () -> SegmentReader.inspect(p), "bit flip at byte " + at);
            assertThrows(IOException.class, () -> SegmentReader.read(p, false), "bit flip at byte " + at);
        }

        // the footer is covered by its own CRC, the locator by its magic and the lengths it states
        byte[] footerFlip = bytes.clone();
        footerFlip[footerOff + 3] ^= 0x40;
        Path pf = write(footerFlip);
        assertEquals("FOOTER", assertThrows(CorruptException.class, () -> SegmentReader.inspect(pf)).section());
        byte[] locFlip = bytes.clone();
        locFlip[size - 32] ^= 0x01;
        Path pl = write(locFlip);
        assertEquals("LOCATOR", assertThrows(CorruptException.class, () -> SegmentReader.inspect(pl)).section());

        // a byte appended and a file cut anywhere in the tail
        Path appended = write(Arrays.copyOf(bytes, size + 1));
        assertEquals("LOCATOR", assertThrows(CorruptException.class, () -> SegmentReader.inspect(appended)).section());
        for (int cut : new int[] {0, 5, 100, SegmentFormat.HEADER_BLOCK, SegmentFormat.HEADER_BLOCK + 100, size - 33, size - 1}) {
            Path p = write(Arrays.copyOf(bytes, cut));
            assertThrows(IOException.class, () -> SegmentReader.inspect(p), "cut at " + cut);
            assertThrows(IOException.class, () -> SegmentReader.read(p, false), "cut at " + cut);
        }
        // control: the undamaged file passes the same calls
        assertEquals(info, SegmentReader.inspect(good));
        SegmentReader.read(good, false);
    }

    // ---- U3: versioning ----

    @Test
    void versionRules() throws IOException {
        final int n = 400;
        SealedGraph g = graph(vectors(n, 8, 21), Metric.L2, params(false));
        long[] vers = versOf(n);
        Path ok = file("ok.tvs");
        SegmentWriter.write(g, vers, ok);

        // wrong magic
        byte[] b = Files.readAllBytes(ok);
        b[0] = 'X';
        Path bad = write(b);
        assertThrows(UnsupportedException.class, () -> SegmentReader.inspect(bad));

        // a larger major is refused
        Path major = file("major2.tvs");
        SegmentWriter.write(g, vers, major, 2, 0, List.of());
        assertThrows(UnsupportedException.class, () -> SegmentReader.read(major, true));

        // a larger minor opens
        Path minor = file("minor7.tvs");
        SegmentWriter.write(g, vers, minor, SegmentFormat.MAJOR, 7, List.of());
        assertEquals(7, SegmentReader.inspect(minor).minor());
        assertSameArrays(g, SegmentReader.read(minor, true));

        // a larger minor with a section this reader does not know: ignored when optional
        Path optional = file("optional.tvs");
        SegmentWriter.write(g, vers, optional, SegmentFormat.MAJOR, 1, List.of(new SegmentWriter.Extra(900, false, new byte[5000])));
        SegmentInfo oi = SegmentReader.inspect(optional);
        assertNotNull(oi.section(900));
        assertFalse(oi.section(900).required());
        assertSameArrays(g, SegmentReader.read(optional, true));

        // refused when it is marked required
        Path required = file("required.tvs");
        SegmentWriter.write(g, vers, required, SegmentFormat.MAJOR, 1, List.of(new SegmentWriter.Extra(901, true, new byte[10])));
        assertThrows(UnsupportedException.class, () -> SegmentReader.inspect(required));

        // a byte-order probe that reads back swapped (header CRC fixed up so only the probe is wrong)
        byte[] p = Files.readAllBytes(ok);
        ByteBuffer bb = ByteBuffer.wrap(p).order(ByteOrder.LITTLE_ENDIAN);
        bb.putInt(SegmentFormat.H_PROBE, 0x04030201);
        bb.putInt(SegmentFormat.HEADER_LEN - 4, SegmentFormat.crc32c(p, 0, SegmentFormat.HEADER_LEN - 4));
        Path swapped = write(p);
        assertThrows(UnsupportedException.class, () -> SegmentReader.inspect(swapped));

        // control: the file all of these were made from opens
        assertSameArrays(g, SegmentReader.read(ok, true));
    }
}
