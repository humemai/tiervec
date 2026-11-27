package ai.humem.tiervec;

import static ai.humem.tiervec.SegmentTestSupport.vectors;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import ai.humem.tiervec.SegmentFormat.CorruptException;
import ai.humem.tiervec.SegmentFormat.UnsupportedException;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.function.Consumer;
import java.util.stream.Stream;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Damage to the tail file of an index directory, found by {@code TieredIndex.open} (#158): a file of another length, a flipped bit in any
 * byte, a missing file, and forged files with valid checksums. {@code open} throws the exception that names the damage and changes nothing,
 * and never serves a generation with a part of its tail. As for a segment, damage that the check of {@code readLatest} sees (the file, its
 * length, its header) leaves the generation before, when that is still there; damaged rows are found when they are read, and refused.
 */
class TierTailCorruptionTest {
    static final int DIM = TierReopenTest.DIM;
    /** rows of each of the two segments, ids 0 to 79 */
    static final int ROWS = 40;
    /** the tail of generation 1: id 5 replaced (first), then ids 80 to 89 */
    static final int TAIL = 11;
    static final int HL = TailFile.HEADER_LEN;
    static final float[][] DATA = vectors(120, DIM, 1593);
    static final float[][] QUERIES = vectors(20, DIM, 1594);

    /** generation 1 alone */
    @TempDir
    static Path single;
    /** a crash after the rename of generation 2: both generations, each with its tail file */
    @TempDir
    static Path twoGen;

    static Hit[][] want1;
    static int size1;

    @TempDir
    Path dir;

    static TieredIndex.Config cfg() {
        return TierReopenTest.cfg();
    }

    @BeforeAll
    static void build() throws IOException {
        try (TieredIndex t = new TieredIndex(DIM, Metric.L2, cfg())) {
            for (int i = 0; i < 2 * ROWS; i++) {
                t.add(i, DATA[i]);
                if (i % ROWS == ROWS - 1) t.sealNow();
            }
            assertTrue(t.awaitIdle(60_000));
            t.add(5, DATA[80]);
            for (int i = 80; i < 90; i++) t.add(i, DATA[i + 1]);
            t.delete(45);
            t.checkpoint(twoGen);
            copy(twoGen, single);
            want1 = results(t);
            size1 = t.size();
            for (int i = 90; i < 95; i++) t.add(i, DATA[i + 1]);
            t.delete(81);
            assertThrows(IOException.class, () -> t.checkpoint(twoGen, step -> {
                if (step == FaultHook.Step.DIR_SYNCED_AFTER_RENAME) throw new IOException("simulated crash after the rename and before the cleanup");
            }));
        }
        assertEquals(TAIL, Manifest.readLatest(single).tailRows());
        assertEquals(2, Manifest.readLatest(twoGen).generation());
        assertTrue(Files.exists(twoGen.resolve(Manifest.tailName(1))) && Files.exists(twoGen.resolve(Manifest.tailName(2))));
    }

    private static Hit[][] results(TieredIndex t) {
        Hit[][] out = new Hit[QUERIES.length][];
        for (int i = 0; i < QUERIES.length; i++) out[i] = t.search(QUERIES[i], 10, null);
        return out;
    }

    private static void copy(Path from, Path to) throws IOException {
        try (Stream<Path> s = Files.list(from)) {
            for (Path p : (Iterable<Path>) s::iterator) {
                if (!p.getFileName().toString().equals(DirectoryLock.FILE)) Files.copy(p, to.resolve(p.getFileName().toString()));
            }
        }
    }

    private void fresh(Path template) throws IOException {
        try (Stream<Path> s = Files.list(dir)) {
            for (Path p : (Iterable<Path>) s::iterator) Files.delete(p);
        }
        copy(template, dir);
    }

    private static Map<String, byte[]> snapshot(Path d) throws IOException {
        Map<String, byte[]> m = new TreeMap<>();
        try (Stream<Path> s = Files.list(d)) {
            for (Path p : (Iterable<Path>) s::iterator) {
                if (!p.getFileName().toString().equals(DirectoryLock.FILE)) m.put(p.getFileName().toString(), Files.readAllBytes(p));
            }
        }
        return m;
    }

    private static void assertUnchanged(Map<String, byte[]> before, Path d, String what) throws IOException {
        Map<String, byte[]> after = snapshot(d);
        assertEquals(before.keySet(), after.keySet(), what + ": the files in the directory");
        for (Map.Entry<String, byte[]> e : before.entrySet()) assertArrayEquals(e.getValue(), after.get(e.getKey()), what + ": " + e.getKey());
    }

    /** opens the directory and returns the named exception that refused it; fails when an index comes back */
    private IOException openRefused(String what) throws IOException {
        try (TieredIndex u = TieredIndex.open(dir, cfg())) {
            return fail(what + ": open returned an index with " + u.size() + " rows from a damaged directory");
        } catch (CorruptStateException e) {
            return assertInstanceOf(CorruptException.class, e.getCause(), what);
        } catch (UnsupportedException e) {
            return e;
        }
    }

    private void assertRefused(String section, String what) throws IOException {
        Map<String, byte[]> before = snapshot(dir);
        CorruptException e = assertInstanceOf(CorruptException.class, openRefused(what), what);
        // when no manifest validates, readLatest wraps the failure of the newest one
        Throwable first = e.section() == null ? e.getCause() : e;
        assertEquals(section, assertInstanceOf(CorruptException.class, first, what).section(), what + ": " + e.getMessage());
        assertUnchanged(before, dir, what);
    }

    private void assertServesGenerationOne(String what) throws IOException {
        try (TieredIndex u = TieredIndex.open(dir, cfg())) {
            assertNull(u.openFallbackCause(), what);
            assertEquals(size1, u.size(), what);
            assertEquals(TAIL, u.pendingTail(), what);
            Hit[][] got = results(u);
            for (int i = 0; i < QUERIES.length; i++) assertArrayEquals(want1[i], got[i], what + ", query " + i);
        }
        assertEquals(snapshot(single).keySet(), snapshot(dir).keySet(), what + ": the files of generation 1 and no other");
    }

    private static byte[] flip(byte[] raw, int at, int bit) {
        byte[] b = raw.clone();
        b[at] ^= (byte) (1 << bit);
        return b;
    }

    @Test
    void aTailFileOfAnyOtherLengthIsRefused() throws IOException {
        fresh(single);
        Path file = dir.resolve(Manifest.tailName(1));
        byte[] raw = Files.readAllBytes(file);
        assertEquals(HL + TAIL * (16 + 4 * DIM), raw.length);
        for (int cut = 0; cut < raw.length; cut++) {
            Files.write(file, Arrays.copyOf(raw, cut));
            assertRefused("TAIL", "cut at " + cut);
        }
        for (int longer : new int[] {1, 16 + 4 * DIM}) {
            Files.write(file, Arrays.copyOf(raw, raw.length + longer));
            assertRefused("TAIL", longer + " bytes appended");
        }
        Files.delete(file);
        assertRefused("MANIFEST", "no tail file");
        Files.write(file, raw);
        assertServesGenerationOne("the file put back");
    }

    @Test
    void aFlippedBitInAnyByteOfTheTailFileIsRefused() throws IOException {
        fresh(single);
        Path file = dir.resolve(Manifest.tailName(1));
        byte[] raw = Files.readAllBytes(file);
        for (int at = 0; at < raw.length; at++) {
            // every bit of the header, and one bit of every byte of the rows
            for (int bit = 0; bit < 8; bit++) {
                if (at >= HL && bit != at % 8) continue;
                Files.write(file, flip(raw, at, bit));
                assertRefused("TAIL", "bit " + bit + " of byte " + at);
            }
        }
        // the open that holds the records builds from them
        List<TieredIndex.IdVector> records = new ArrayList<>();
        for (int i = 0; i < 20; i++) records.add(new TieredIndex.IdVector(i, DATA[i]));
        for (int at : new int[] {TailFile.H_DIM, raw.length - 1}) {
            Files.write(file, flip(raw, at, 0));
            Map<String, byte[]> before = snapshot(dir);
            try (TieredIndex r = TieredIndex.open(dir, DIM, Metric.L2, cfg(), () -> records)) {
                assertInstanceOf(CorruptException.class, r.openFallbackCause());
                assertEquals(records.size(), r.size());
            }
            assertUnchanged(before, dir, "the fallback, byte " + at);
        }
    }

    @Test
    void damageThatReadLatestSeesLeavesTheGenerationBeforeWithItsWholeTail() throws IOException {
        final int length = (int) Files.size(twoGen.resolve(Manifest.tailName(2)));
        for (int cut : new int[] {-1, 0, 1, HL - 1, HL, length / 2, length - 1}) {
            fresh(twoGen);
            Path file = dir.resolve(Manifest.tailName(2));
            if (cut < 0) Files.delete(file);
            else Files.write(file, Arrays.copyOf(Files.readAllBytes(file), cut));
            assertServesGenerationOne(cut < 0 ? "no tail file" : "cut at " + cut);
        }
        for (int at = 0; at < HL; at++) {
            fresh(twoGen);
            Path file = dir.resolve(Manifest.tailName(2));
            Files.write(file, flip(Files.readAllBytes(file), at, at % 8));
            assertServesGenerationOne("a bit of header byte " + at);
        }
    }

    @Test
    void aFlippedRowBitInTheNewestGenerationIsRefusedNotHiddenByTheOlderOne() throws IOException {
        final int length = (int) Files.size(twoGen.resolve(Manifest.tailName(2)));
        for (int at : new int[] {HL, HL + 8 * 9, length / 2, length - 1}) {
            fresh(twoGen);
            Path file = dir.resolve(Manifest.tailName(2));
            Files.write(file, flip(Files.readAllBytes(file), at, 3));
            assertRefused("TAIL", "byte " + at + " of the newest tail");
        }
    }

    // ---- forged files: every checksum is right ----

    private static ByteBuffer le(byte[] b) {
        return ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN);
    }

    /** a manifest that names {@code tail} as the tail file of {@code m}, with {@code rows} rows, the epoch and the high-water id given */
    private static byte[] manifestFor(Manifest m, long epoch, long hwm, long rows, byte[] tail) {
        int hl = le(tail).getInt(TailFile.H_LEN);
        return new Manifest(m.generation(), epoch, hwm, m.versionCounter(), m.nextSegId(), m.fingerprint(), m.entries(), m.tombGeneration(),
                m.tombLength(), m.tombCrc(), rows, tail.length, SegmentFormat.crc32c(tail, hl, tail.length - hl))
                .encode(Manifest.FORMAT_MAJOR, Manifest.FORMAT_MINOR);
    }

    /** changes the tail file of the directory, then makes the checksum of its header and the manifest agree with the result */
    private void forge(Consumer<ByteBuffer> change) throws IOException {
        Manifest m = Manifest.readLatest(dir);
        Path file = dir.resolve(Manifest.tailName(m.generation()));
        byte[] raw = Files.readAllBytes(file);
        ByteBuffer b = le(raw);
        change.accept(b);
        int hl = b.getInt(TailFile.H_LEN);
        b.putInt(hl - 4, SegmentFormat.crc32c(raw, 0, hl - 4));
        Files.write(file, raw);
        Files.write(dir.resolve(Manifest.manifestName(m.generation())), manifestFor(m, m.epoch(), m.hwm(), m.tailRows(), raw));
    }

    private static int idAt(int row) {
        return HL + 8 * row;
    }

    private static int verAt(int row) {
        return HL + 8 * TAIL + 8 * row;
    }

    @Test
    void aForgedTailIsRefused() throws IOException {
        fresh(single);
        final Manifest m = Manifest.readLatest(dir);
        record Case(String what, Consumer<ByteBuffer> change) {}
        List<Case> cases = List.of(
                new Case("another magic", b -> b.put(0, (byte) 'X')),
                new Case("another generation", b -> b.putLong(TailFile.H_GEN, 9)),
                new Case("another dimension", b -> b.putInt(TailFile.H_DIM, DIM + 1)),
                new Case("another row count", b -> b.putLong(TailFile.H_ROWS, TAIL + 1)),
                new Case("an id twice", b -> b.putLong(idAt(2), b.getLong(idAt(1)))),
                new Case("an id that a segment shows", b -> b.putLong(idAt(1), 20)),
                new Case("versions out of order", b -> {
                    long a = b.getLong(verAt(1));
                    b.putLong(verAt(1), b.getLong(verAt(2))).putLong(verAt(2), a);
                }),
                new Case("a version above the counter", b -> b.putLong(verAt(TAIL - 1), m.versionCounter() + 1)),
                new Case("version 0", b -> b.putLong(verAt(0), 0)),
                new Case("a row older than the segment copy of its id", b -> b.putLong(verAt(0), 1)));
        for (Case c : cases) {
            fresh(single);
            forge(c.change());
            assertRefused("TAIL", c.what());
        }

        // the manifest and the header agree on one row less than the file holds
        fresh(single);
        Path file = dir.resolve(Manifest.tailName(1));
        byte[] raw = Files.readAllBytes(file);
        le(raw).putLong(TailFile.H_ROWS, TAIL - 1).putInt(HL - 4, 0);
        le(raw).putInt(HL - 4, SegmentFormat.crc32c(raw, 0, HL - 4));
        Files.write(file, raw);
        Files.write(dir.resolve(Manifest.manifestName(1)), manifestFor(m, m.epoch(), m.hwm(), TAIL - 1, raw));
        assertRefused("TAIL", "a row count that does not fill the file");

        // inside an epoch no id of the tail is above the high-water id of the manifest
        fresh(single);
        Files.write(dir.resolve(Manifest.manifestName(1)), manifestFor(m, 5, 88, TAIL, Files.readAllBytes(file)));
        assertRefused("TAIL", "an id above the high-water id");
        Files.write(dir.resolve(Manifest.manifestName(1)), manifestFor(m, 5, 89, TAIL, Files.readAllBytes(file)));
        TieredIndex.open(dir, cfg()).close();

        // a manifest that names a tail file and no rows
        fresh(single);
        Path manifest = dir.resolve(Manifest.manifestName(1));
        byte[] bytes = Files.readAllBytes(manifest);
        le(bytes).putLong(96, 0);
        le(bytes).putInt(bytes.length - 4, SegmentFormat.crc32c(bytes, 0, bytes.length - 4));
        Files.write(manifest, bytes);
        assertRefused("MANIFEST", "no rows and a length");
    }

    @Test
    void aTailOfANewerMajorIsRefusedAndNotReplaced() throws IOException {
        for (Path template : new Path[] {single, twoGen}) {
            fresh(template);
            forge(b -> b.putShort(TailFile.H_MAJOR, (short) (TailFile.MAJOR + 1)));
            Map<String, byte[]> before = snapshot(dir);
            assertThrows(UnsupportedException.class, () -> Manifest.readLatest(dir), "never skipped for an older generation");
            assertInstanceOf(UnsupportedException.class, openRefused("tail major 2"));
            try (TieredIndex t = new TieredIndex(DIM, Metric.L2, cfg())) {
                assertThrows(UnsupportedException.class, () -> t.checkpoint(dir));
            }
            assertUnchanged(before, dir, "tail major 2");
        }
    }

    @Test
    void aTailOfALargerMinorWithALongerHeaderOpens() throws IOException {
        fresh(single);
        Manifest m = Manifest.readLatest(dir);
        Path file = dir.resolve(Manifest.tailName(1));
        byte[] raw = Files.readAllBytes(file);
        final int longer = HL + 8;
        byte[] grown = new byte[raw.length + 8];
        System.arraycopy(raw, 0, grown, 0, HL - 4);
        System.arraycopy(raw, HL, grown, longer, raw.length - HL);
        le(grown).putShort(TailFile.H_MINOR, (short) 4).putInt(TailFile.H_LEN, longer).putInt(HL - 4, 0x5EED5EED);
        le(grown).putInt(longer - 4, SegmentFormat.crc32c(grown, 0, longer - 4));
        Files.write(file, grown);
        Files.write(dir.resolve(Manifest.manifestName(1)), manifestFor(m, m.epoch(), m.hwm(), TAIL, grown));
        assertServesGenerationOne("a larger minor");
    }
}
