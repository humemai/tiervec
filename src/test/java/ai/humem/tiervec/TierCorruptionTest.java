package ai.humem.tiervec;

import static ai.humem.tiervec.SegmentTestSupport.idOf;
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
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.stream.Stream;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.opentest4j.AssertionFailedError;

/**
 * Damage to an index directory, found by {@code TieredIndex.open}. The segment file rules are tested byte by byte in
 * {@link SegmentCorruptionTest} and {@link SegmentForgedFileTest}; this class checks what the directory does with them: a damaged segment, manifest
 * or tombstone file makes {@code open} throw a named exception (never return an index), a failed open deletes and changes nothing, the overload that
 * holds the records falls back to a rebuild and serves them, a newer format is refused and not skipped, and a larger minor version opens and serves
 * exactly what the writer's tier served.
 */
class TierCorruptionTest {
    static final int DIM = TierReopenTest.DIM;
    static final int ROWS = 400;

    /** the directory of the checkpoint of generation 1: two segments, seg-000001 and seg-000002 */
    @TempDir
    static Path single;
    /** a directory after a crash after the rename of generation 2: generation 1 (segments 1 and 2) and generation 2 (segments 1, 2 and 3) */
    @TempDir
    static Path twoGen;

    static TieredIndex tier;
    static float[][] data, qs;
    static Hit[][] want1;
    static List<TieredIndex.IdVector> records1;

    @TempDir
    Path dir;

    /** SQ8 on, so that a segment file has all twelve sections */
    static TieredIndex.Config cfg() {
        return TierReopenTest.cfg().withGraph(GraphIndex.Params.defaults().withEfConstruction(64).withSq8(true));
    }

    @BeforeAll
    static void build() throws IOException {
        data = vectors(3 * ROWS, DIM, 91);
        qs = vectors(40, DIM, 92);
        tier = new TieredIndex(DIM, Metric.L2, cfg());
        batch(0);
        batch(1);
        tier.checkpoint(twoGen); // generation 1
        want1 = results(tier);
        copy(twoGen, single);
        records1 = new ArrayList<>();
        for (int i = 0; i < 2 * ROWS; i++) records1.add(new TieredIndex.IdVector(idOf(i), data[i]));
        batch(2);
        assertThrows(IOException.class, () -> tier.checkpoint(twoGen, step -> {
            if (step == FaultHook.Step.DIR_SYNCED_AFTER_RENAME) throw new IOException("simulated crash after the rename and before the cleanup");
        }));
        assertEquals(2, Manifest.readLatest(twoGen).generation());
        assertEquals(3, Manifest.readLatest(twoGen).entries().size());
        assertEquals(12, SegmentReader.inspect(single.resolve(Manifest.segName(1))).sections().size());
        assertEquals(1, Manifest.readLatest(single).generation());
        assertEquals(2, Manifest.readLatest(single).entries().size());
        assertTrue(Files.exists(twoGen.resolve(Manifest.manifestName(1))) && Files.exists(twoGen.resolve(Manifest.manifestName(2))));
    }

    @AfterAll
    static void close() {
        tier.close();
    }

    private static void batch(int b) {
        for (int i = b * ROWS; i < (b + 1) * ROWS; i++) tier.add(idOf(i), data[i]);
        tier.sealNow();
        assertTrue(tier.awaitIdle(60_000));
    }

    private static Hit[][] results(TieredIndex t) {
        Hit[][] out = new Hit[qs.length][];
        for (int i = 0; i < qs.length; i++) out[i] = t.search(qs[i], 10, null);
        return out;
    }

    private static void assertServes(TieredIndex u, Hit[][] want) {
        for (int i = 0; i < qs.length; i++) assertArrayEquals(want[i], u.search(qs[i], 10, null), "query " + i);
    }

    private static void copy(Path from, Path to) throws IOException {
        try (Stream<Path> s = Files.list(from)) {
            for (Path p : (Iterable<Path>) s::iterator) {
                if (!p.getFileName().toString().equals(DirectoryLock.FILE)) Files.copy(p, to.resolve(p.getFileName().toString()));
            }
        }
    }

    private Path seg(long id) {
        return dir.resolve(Manifest.segName(id));
    }

    /** empties the test's directory so that the next copy of a template can go in */
    private void clear() throws IOException {
        try (Stream<Path> s = Files.list(dir)) {
            for (Path p : (Iterable<Path>) s::iterator) Files.delete(p);
        }
    }

    /** the files of a directory, without the lock file (it holds nothing, and Windows does not let a locked file be read) */
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

    /** opens the directory; returns the named exception that refused it, and fails when an index comes back or anything else is thrown */
    private static IOException openRefused(Path d, String what) {
        try {
            TieredIndex u = TieredIndex.open(d, cfg());
            int size = u.size();
            u.close();
            fail(what + ": open returned an index with " + size + " rows from a damaged directory");
            return null;
        } catch (CorruptStateException e) { // damage is unchecked at open, with the exception that names it as its cause
            return assertInstanceOf(CorruptException.class, e.getCause(), what);
        } catch (UnsupportedException e) {
            return e;
        } catch (AssertionFailedError e) {
            throw e;
        } catch (Throwable t) {
            throw new AssertionFailedError(what + ": an uncontrolled failure instead of a named exception: " + t, t);
        }
    }

    /** the same for {@link Manifest#readLatest}, which throws the checked exceptions */
    private static IOException latestRefused(Path d, String what) {
        try {
            Manifest m = Manifest.readLatest(d);
            fail(what + ": readLatest returned generation " + (m == null ? "null" : m.generation()) + " from a damaged manifest");
            return null;
        } catch (CorruptException | UnsupportedException e) {
            return e;
        } catch (AssertionFailedError e) {
            throw e;
        } catch (Throwable t) {
            throw new AssertionFailedError(what + ": an uncontrolled failure instead of a named exception: " + t, t);
        }
    }

    // ---- a damaged segment file ----

    /** every damage here is corruption; a newer format is forged with matching checksums, further down */
    enum Expect { CORRUPT }

    /** a way to damage the bytes of one segment file; {@code section} is the section the failure must name, when one is known */
    record Damage(String name, BiFunction<byte[], SegmentInfo, byte[]> apply, Expect expect, String section) {
        @Override
        public String toString() {
            return name;
        }
    }

    private static byte[] cut(byte[] b, long len) {
        return Arrays.copyOf(b, (int) len);
    }

    private static byte[] flip(byte[] b, long pos) {
        byte[] c = b.clone();
        c[(int) pos] ^= (byte) (1 << (pos % 8));
        return c;
    }

    static Stream<Damage> segmentDamage() {
        List<Damage> l = new ArrayList<>();
        for (int id = 1; id <= 12; id++) {
            final int sid = id;
            final String name = SegmentFormat.sectionName(id);
            l.add(new Damage("cut in the middle of " + name, (b, i) -> cut(b, i.section(sid).offset() + i.section(sid).length() / 2), Expect.CORRUPT, null));
            l.add(new Damage("cut at the last byte of " + name, (b, i) -> cut(b, i.section(sid).offset() + i.section(sid).length() - 1), Expect.CORRUPT, null));
            l.add(new Damage("bit flipped in the first byte of " + name, (b, i) -> flip(b, i.section(sid).offset()), Expect.CORRUPT, name));
            l.add(new Damage("bit flipped in the middle of " + name, (b, i) -> flip(b, i.section(sid).offset() + i.section(sid).length() / 2), Expect.CORRUPT, name));
            l.add(new Damage("bit flipped in the last byte of " + name, (b, i) -> flip(b, i.section(sid).offset() + i.section(sid).length() - 1), Expect.CORRUPT, name));
        }
        for (int len : new int[] {0, 7, 8, 100, SegmentFormat.HEADER_BLOCK, SegmentFormat.HEADER_BLOCK + SegmentFormat.LOCATOR_LEN - 1}) {
            l.add(new Damage("cut to " + len + " bytes", (b, i) -> cut(b, len), Expect.CORRUPT, null));
        }
        l.add(new Damage("cut at the start of the footer", (b, i) -> cut(b, footerOff(b)), Expect.CORRUPT, null));
        l.add(new Damage("cut in the middle of the footer", (b, i) -> cut(b, footerOff(b) + 100), Expect.CORRUPT, null));
        l.add(new Damage("cut in the middle of the locator", (b, i) -> cut(b, b.length - 16), Expect.CORRUPT, null));
        l.add(new Damage("cut at the last byte", (b, i) -> cut(b, b.length - 1), Expect.CORRUPT, null));
        l.add(new Damage("a byte appended", (b, i) -> cut(b, b.length + 1), Expect.CORRUPT, null));
        for (int pos : new int[] {10, 11, 12, 16, 24, 32, 40, 44, 48, 51}) {
            l.add(new Damage("bit flipped in header byte " + pos, (b, i) -> flip(b, pos), Expect.CORRUPT, null));
        }
        // open reports the damage of the manifest's newest candidate through readLatest, whose exception has no section of its own: the cause has it
        l.add(new Damage("bit flipped in the magic", (b, i) -> flip(b, 3), Expect.CORRUPT, null));
        l.add(new Damage("bit flipped in the major version", (b, i) -> flip(b, 8), Expect.CORRUPT, null));
        l.add(new Damage("bit flipped in the second byte of the major version", (b, i) -> flip(b, 9), Expect.CORRUPT, null));
        l.add(new Damage("magic zeroed", (b, i) -> {
            byte[] c = b.clone();
            Arrays.fill(c, 0, 8, (byte) 0);
            return c;
        }, Expect.CORRUPT, null));
        l.add(new Damage("all bytes zeroed", (b, i) -> new byte[b.length], Expect.CORRUPT, null));
        for (int rel : new int[] {0, 5, 40, 64 + 12, 64 + 40 * 11 + 39}) {
            l.add(new Damage("bit flipped in footer byte " + rel, (b, i) -> flip(b, footerOff(b) + rel), Expect.CORRUPT, null));
        }
        for (int rel : new int[] {0, 7, 8, 16, 20, 24, 31}) {
            l.add(new Damage("bit flipped in locator byte " + rel, (b, i) -> flip(b, b.length - SegmentFormat.LOCATOR_LEN + rel), Expect.CORRUPT, null));
        }
        return l.stream();
    }

    private static int footerOff(byte[] file) {
        return (int) SegmentForge.le(file).getLong(file.length - SegmentFormat.LOCATOR_LEN + 8);
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("segmentDamage")
    void aDamagedSegmentFileIsRefusedByOpenAndNothingIsDeleted(Damage damage) throws IOException {
        for (long segId : new long[] {1, 2}) {
            copy(single, dir);
            SegmentInfo info = SegmentReader.inspect(seg(segId));
            Files.write(seg(segId), damage.apply().apply(Files.readAllBytes(seg(segId)), info));
            Map<String, byte[]> before = snapshot(dir);
            String what = damage.name() + " in segment " + segId;
            IOException e = openRefused(dir, what);
            assertEquals(Expect.CORRUPT, damage.expect());
            assertInstanceOf(CorruptException.class, e, what);
            // a flip inside a section is found when the section is read: the exception names it
            if (damage.section() != null) assertEquals(damage.section(), ((CorruptException) e).section(), what);
            assertUnchanged(before, dir, what);
            clear();
        }
    }

    // ---- the fallback to the records ----

    private void assertRebuiltFromTheRecords(TieredIndex r, String what) {
        assertEquals(2 * ROWS, r.size(), what);
        int hits = 0;
        for (int i = 0; i < 2 * ROWS; i += 20) {
            Hit[] h = r.search(data[i], 1, null);
            if (h.length == 1 && h[0].id() == idOf(i)) hits++;
        }
        assertTrue(hits >= 38, what + ": the rebuilt index finds its own records, " + hits + " of 40");
    }

    @Test
    void whenOpenCannotTrustTheDirectoryTheRecordsAreServedAndTheDirectoryIsLeftAlone() throws IOException {
        List<Damage> picks = new ArrayList<>();
        segmentDamage().forEach(d -> {
            if (d.name().equals("bit flipped in the middle of VEC") || d.name().equals("cut in the middle of L0_EDGES") || d.name().equals("bit flipped in header byte 24")
                    || d.name().equals("magic zeroed") || d.name().equals("bit flipped in the major version") || d.name().equals("bit flipped in locator byte 20")
                    || d.name().equals("cut at the last byte of SQ8_PARAMS")) picks.add(d);
        });
        assertEquals(7, picks.size());
        for (Damage d : picks) {
            copy(single, dir);
            SegmentInfo info = SegmentReader.inspect(seg(1));
            Files.write(seg(1), d.apply().apply(Files.readAllBytes(seg(1)), info));
            Map<String, byte[]> before = snapshot(dir);
            TieredIndex r = TieredIndex.open(dir, DIM, Metric.L2, cfg(), () -> records1);
            try {
                Throwable cause = r.openFallbackCause();
                assertTrue(cause instanceof CorruptException || cause instanceof UnsupportedException, d.name() + ": the cause is " + cause);
                assertRebuiltFromTheRecords(r, d.name());
            } finally {
                r.close();
            }
            assertUnchanged(before, dir, d.name());
            clear();
        }
    }

    // ---- the manifest and the tombstone file ----

    /** flips bit {@code bit} of byte {@code pos} of a file in place; the old byte is returned */
    private static byte flipInPlace(FileChannel ch, long pos, int bit) throws IOException {
        ByteBuffer b = ByteBuffer.allocate(1);
        ch.read(b, pos);
        byte old = b.get(0);
        ch.write(ByteBuffer.wrap(new byte[] {(byte) (old ^ (1 << bit))}), pos);
        return old;
    }

    @Test
    void everyBitOfTheManifestIsGuarded() throws IOException {
        copy(single, dir);
        Path mf = dir.resolve(Manifest.manifestName(1));
        final int len = (int) Files.size(mf);
        try (FileChannel ch = FileChannel.open(mf, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
            for (int pos = 0; pos < len; pos++) {
                for (int bit = 0; bit < 8; bit++) {
                    byte old = flipInPlace(ch, pos, bit);
                    String what = "bit " + bit + " of manifest byte " + pos;
                    IOException e = latestRefused(dir, what);
                    // every byte, the major version included, is under the magic check, the length check or the CRC of the body (#123)
                    assertInstanceOf(CorruptException.class, e, what);
                    ch.write(ByteBuffer.wrap(new byte[] {old}), pos);
                }
            }
        }
        assertEquals(1, Manifest.readLatest(dir).generation(), "the loop restored every byte");
    }

    @Test
    void everyCutOfTheManifestIsRefused() throws IOException {
        copy(single, dir);
        Path mf = dir.resolve(Manifest.manifestName(1));
        final int len = (int) Files.size(mf);
        try (FileChannel ch = FileChannel.open(mf, StandardOpenOption.WRITE)) {
            for (int cut = len - 1; cut >= 0; cut--) {
                ch.truncate(cut);
                assertInstanceOf(CorruptException.class, latestRefused(dir, "manifest cut at " + cut), "manifest cut at " + cut);
            }
        }
        assertInstanceOf(CorruptException.class, openRefused(dir, "an empty manifest"));
    }

    @Test
    void everyBitAndEveryCutOfTheTombstoneFileIsRefused() throws IOException {
        copy(single, dir);
        Path tf = dir.resolve(Manifest.tombName(1));
        final int len = (int) Files.size(tf);
        byte[] raw = Files.readAllBytes(tf);
        try (FileChannel ch = FileChannel.open(tf, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
            for (int pos = 0; pos < len; pos++) {
                for (int bit = 0; bit < 8; bit++) {
                    byte old = flipInPlace(ch, pos, bit);
                    String what = "bit " + bit + " of tombstone byte " + pos;
                    IOException e = latestRefused(dir, what);
                    // the manifest holds the length and the CRC of the file it names, so this is found before the file is parsed
                    CorruptException cause = assertInstanceOf(CorruptException.class, e.getCause(), what);
                    assertEquals("TOMBSTONES", cause.section(), what);
                    ch.write(ByteBuffer.wrap(new byte[] {old}), pos);
                }
            }
        }
        Files.write(tf, raw);
        try (FileChannel ch = FileChannel.open(tf, StandardOpenOption.WRITE)) {
            for (int cut = len - 1; cut >= 0; cut--) {
                ch.truncate(cut);
                CorruptException cause = assertInstanceOf(CorruptException.class, latestRefused(dir, "tombstone cut at " + cut).getCause(), "cut at " + cut);
                assertEquals("TOMBSTONES", cause.section(), "tombstone cut at " + cut);
            }
        }
        Files.write(tf, raw);
        assertEquals(1, Manifest.readLatest(dir).generation());
    }

    /** rewrites the tombstone file by {@code change} (the CRC inside it is recomputed) and the manifest to name the new bytes, so that only the change can be found */
    private void forgeTombstones(Consumer<byte[]> change) throws IOException {
        Manifest m = Manifest.readLatest(dir);
        Path tf = dir.resolve(Manifest.tombName(m.tombGeneration()));
        byte[] t = Files.readAllBytes(tf);
        change.accept(t);
        SegmentForge.le(t).putInt(t.length - 4, SegmentFormat.crc32c(t, 0, t.length - 4));
        Files.write(tf, t);
        Manifest m2 = new Manifest(m.generation(), m.epoch(), m.hwm(), m.versionCounter(), m.nextSegId(), m.fingerprint(), m.entries(), m.tombGeneration(),
                t.length, SegmentFormat.crc32c(t, 0, t.length));
        Files.write(dir.resolve(Manifest.manifestName(m.generation())), m2.encode(Manifest.FORMAT_MAJOR, Manifest.FORMAT_MINOR));
    }

    @Test
    void aTombstoneFileThatIsNotUnderstoodIsRefusedWithValidChecksums() throws IOException {
        // a larger major
        copy(single, dir);
        forgeTombstones(t -> SegmentForge.le(t).putShort(8, (short) 2));
        assertInstanceOf(UnsupportedException.class, openRefused(dir, "tombstone major 2"));
        // a larger minor opens
        clear();
        copy(single, dir);
        forgeTombstones(t -> SegmentForge.le(t).putShort(10, (short) 9));
        try (TieredIndex u = TieredIndex.open(dir, cfg())) {
            assertServes(u, want1);
        }
        // a wrong magic
        clear();
        copy(single, dir);
        forgeTombstones(t -> t[0] ^= 1);
        CorruptException e = assertInstanceOf(CorruptException.class, latestRefused(dir, "tombstone magic").getCause());
        assertEquals("TOMBSTONES", e.section());
        // a marked row is hidden
        clear();
        copy(single, dir);
        forgeTombstones(t -> t[24 + 16] |= 1); // the first word of the bitmap of the first segment
        final List<Manifest.Entry> listed = Manifest.readLatest(dir).entries();
        final long rows = listed.get(0).n(); // of the first segment, whose bitmap comes first
        final long all = listed.stream().mapToLong(Manifest.Entry::n).sum();
        try (TieredIndex u = TieredIndex.open(dir, cfg())) {
            assertEquals(all - 1, u.size());
        }
        // a mark for a row that the segment does not have
        clear();
        copy(single, dir);
        assertTrue(rows % 64 != 0, "the last word of the bitmap has spare bits: " + rows + " rows");
        forgeTombstones(t -> t[24 + 16 + 8 * (int) (rows / 64) + 7] |= (byte) 0x80); // the highest bit of the last word of the first segment
        CorruptException beyond = assertInstanceOf(CorruptException.class, latestRefused(dir, "a tombstone bit beyond the rows").getCause());
        assertEquals("TOMBSTONES", beyond.section());
    }

    // ---- versions, in the directory ----

    /**
     * rewrites a segment file with {@code change} applied (the CRCs of the file are recomputed) and the newest manifest to name the new file, so that
     * the manifest is intact and agrees with the segment
     */
    private void rewriteSegment(long segId, Consumer<SegmentForge> change) throws IOException {
        Manifest m = Manifest.readLatest(dir);
        SegmentForge forge = SegmentForge.of(Files.readAllBytes(seg(segId)));
        change.accept(forge);
        byte[] bytes = forge.build();
        Files.write(seg(segId), bytes);
        int hl = forge.headBuf().getInt(SegmentFormat.H_LEN);
        int headerCrc = forge.headBuf().getInt(hl - 4);
        List<Manifest.Entry> entries = new ArrayList<>();
        for (Manifest.Entry e : m.entries()) {
            entries.add(e.segId() != segId ? e : new Manifest.Entry(e.segId(), e.n(), bytes.length, headerCrc, e.verMin(), e.verMax()));
        }
        Manifest m2 = new Manifest(m.generation(), m.epoch(), m.hwm(), m.versionCounter(), m.nextSegId(), m.fingerprint(), entries, m.tombGeneration(),
                m.tombLength(), m.tombCrc());
        Files.write(dir.resolve(Manifest.manifestName(m.generation())), m2.encode(Manifest.FORMAT_MAJOR, Manifest.FORMAT_MINOR));
    }

    @Test
    void aLargerMinorSegmentOpensAndServesWhatTheWritersTierServed() throws IOException {
        copy(single, dir);
        rewriteSegment(1, f -> f.headBuf().putShort(SegmentFormat.H_MINOR, (short) 3));
        rewriteSegment(2, f -> {
            f.headBuf().putShort(SegmentFormat.H_MINOR, (short) 7);
            f.appendSection(900, 0, 1, new byte[5000]); // a section of the newer minor that this reader does not know
        });
        try (TieredIndex u = TieredIndex.open(dir, cfg())) {
            assertNull(u.openFallbackCause());
            assertEquals(2 * ROWS, u.size());
            assertServes(u, want1);
        }
        // the same with the header and the footer metadata grown by the newer minor
        clear();
        copy(single, dir);
        rewriteSegment(1, f -> {
            f.headBuf().putShort(SegmentFormat.H_MINOR, (short) 2);
            f.headBuf().putInt(SegmentFormat.H_LEN, 60);
            f.headBuf().putInt(52, 0x5EED5EED);
            f.meta = Arrays.copyOf(f.meta, 96);
        });
        try (TieredIndex u = TieredIndex.open(dir, cfg())) {
            assertServes(u, want1);
        }
    }

    @Test
    void aSegmentWithAnUnknownRequiredSectionIsRefusedNotSkipped() throws IOException {
        copy(twoGen, dir);
        // the newest segment of generation 2; generation 1 is intact and would open
        rewriteSegment(3, f -> {
            f.headBuf().putShort(SegmentFormat.H_MINOR, (short) 1);
            f.appendSection(901, SegmentFormat.FLAG_REQUIRED, 1, new byte[64]);
        });
        Map<String, byte[]> before = snapshot(dir);
        assertInstanceOf(UnsupportedException.class, openRefused(dir, "unknown required section"));
        assertUnchanged(before, dir, "unknown required section");
        TieredIndex r = TieredIndex.open(dir, DIM, Metric.L2, cfg(), () -> records1);
        try {
            assertInstanceOf(UnsupportedException.class, r.openFallbackCause());
            assertRebuiltFromTheRecords(r, "unknown required section");
        } finally {
            r.close();
        }
    }

    @Test
    void aSegmentOfALargerMajorInTheNewestGenerationIsRefusedNotReplacedByAnOlderGeneration() throws IOException {
        copy(twoGen, dir);
        rewriteSegment(3, f -> f.headBuf().putShort(SegmentFormat.H_MAJOR, (short) 2));
        Map<String, byte[]> before = snapshot(dir);
        // generation 1 is intact, and opening it would serve the tier as it was before the third segment: silently stale
        assertInstanceOf(UnsupportedException.class, latestRefused(dir, "major 2"));
        assertInstanceOf(UnsupportedException.class, openRefused(dir, "major 2"));
        assertUnchanged(before, dir, "major 2");
        TieredIndex r = TieredIndex.open(dir, DIM, Metric.L2, cfg(), () -> records1);
        try {
            assertInstanceOf(UnsupportedException.class, r.openFallbackCause());
            // #130: the rebuilt tier is settled, but it may not replace files of a newer format
            assertThrows(UnsupportedException.class, () -> r.checkpoint(dir));
            assertUnchanged(before, dir, "a checkpoint over major 2");
        } finally {
            r.close();
        }
    }

    @Test
    void aDamagedMagicInTheNewestSegmentMakesTheOlderGenerationTheNewestThatValidates() throws IOException {
        // #83: a damaged magic is damage, like a damaged header CRC, so the newest generation whose segments pass is the live one. It used to be
        // reported as UnsupportedException, which open refuses and does not skip, so one flipped bit in the first eight bytes of the newest
        // segment made the whole open fail. A larger major version (the test above) is the case that stays refused.
        for (int variant = 0; variant < 11; variant++) {
            copy(twoGen, dir);
            byte[] b = Files.readAllBytes(seg(3));
            String what;
            if (variant < 8) {
                b[variant] ^= (byte) (1 << variant);
                what = "bit " + variant + " of magic byte " + variant;
            } else if (variant == 8) {
                Arrays.fill(b, 0, 8, (byte) 0);
                what = "zeroed magic";
            } else if (variant == 9) {
                Arrays.fill(b, (byte) 0); // the length reached the disk and the data did not
                what = "a file of zeros";
            } else {
                b = new byte[0];
                what = "an empty file";
            }
            Files.write(seg(3), b);
            try (TieredIndex u = TieredIndex.open(dir, cfg())) {
                assertNull(u.openFallbackCause(), what);
                assertEquals(2 * ROWS, u.size(), what);
                assertServes(u, want1);
            }
            assertEquals(1, Manifest.readLatest(dir).generation(), what);
            assertTrue(!Files.exists(seg(3)) && !Files.exists(dir.resolve(Manifest.manifestName(2))), what + ": the files of the unfinished generation were removed");
            clear();
        }
    }

    @Test
    void aDamagedMagicInASegmentThatEveryGenerationNamesIsRefusedAsCorruptAndNothingIsDeleted() throws IOException {
        copy(twoGen, dir);
        byte[] b = Files.readAllBytes(seg(1)); // segment 1 is in generation 1 and in generation 2: no generation is left to fall back to
        b[0] ^= 1;
        Files.write(seg(1), b);
        Map<String, byte[]> before = snapshot(dir);
        CorruptException e = assertInstanceOf(CorruptException.class, openRefused(dir, "magic of segment 1"));
        assertTrue(e.getMessage().contains("2 candidates"), e.getMessage());
        // the message says which file and which field, and the cause is the reader's exception for the header
        assertTrue(e.getMessage().contains(Manifest.segName(1)) && e.getMessage().contains("magic"), e.getMessage());
        assertEquals("HEADER", assertInstanceOf(CorruptException.class, e.getCause()).section());
        assertUnchanged(before, dir, "magic of segment 1");
        TieredIndex r = TieredIndex.open(dir, DIM, Metric.L2, cfg(), () -> records1);
        try {
            assertInstanceOf(CorruptException.class, r.openFallbackCause());
            assertRebuiltFromTheRecords(r, "magic of segment 1");
        } finally {
            r.close();
        }
    }

    @Test
    void everyBitOfTheMajorVersionOfTheNewestSegmentIsDamageUnlessTheHeaderCrcMatches() throws IOException {
        // #116: the two cases on every bit of both major version bytes, in a directory with an older generation to fall back to
        for (int pos = SegmentFormat.H_MAJOR; pos < SegmentFormat.H_MINOR; pos++) {
            for (int bit = 0; bit < 8; bit++) {
                final int at = pos;
                final byte mask = (byte) (1 << bit);
                String what = "bit " + bit + " of header byte " + pos + " of the newest segment";

                // damage: the header CRC no longer matches, so the older generation is served and the unfinished one is removed
                copy(twoGen, dir);
                byte[] b = Files.readAllBytes(seg(3));
                b[at] ^= mask;
                Files.write(seg(3), b);
                try (TieredIndex u = TieredIndex.open(dir, cfg())) {
                    assertNull(u.openFallbackCause(), what);
                    assertEquals(2 * ROWS, u.size(), what);
                    assertServes(u, want1);
                }
                assertEquals(1, Manifest.readLatest(dir).generation(), what);
                assertTrue(!Files.exists(seg(3)) && !Files.exists(dir.resolve(Manifest.manifestName(2))), what);
                clear();

                // a newer format: the same change with a matching header CRC is refused, never skipped
                copy(twoGen, dir);
                rewriteSegment(3, f -> f.head[at] ^= mask);
                Map<String, byte[]> before = snapshot(dir);
                assertInstanceOf(UnsupportedException.class, latestRefused(dir, what + ", matching CRC"));
                assertInstanceOf(UnsupportedException.class, openRefused(dir, what + ", matching CRC"));
                assertUnchanged(before, dir, what + ", matching CRC");
                clear();
            }
        }
    }

    /** the manifest bytes with the CRC at the end recomputed */
    private static byte[] withManifestCrc(byte[] m) {
        SegmentForge.le(m).putInt(m.length - 4, SegmentFormat.crc32c(m, 0, m.length - 4));
        return m;
    }

    @Test
    void everyBitOfTheMajorVersionOfTheNewestManifestIsDamageUnlessItsCrcMatches() throws IOException {
        // #123: the two cases on every bit of both major version bytes of a manifest
        for (int pos = 8; pos < 10; pos++) {
            for (int bit = 0; bit < 8; bit++) {
                String what = "bit " + bit + " of byte " + pos + " of the newest manifest";

                // damage: the older generation is served, the unfinished one is removed, and the work goes on
                copy(twoGen, dir);
                Path m2 = dir.resolve(Manifest.manifestName(2));
                byte[] b = Files.readAllBytes(m2);
                b[pos] ^= (byte) (1 << bit);
                Files.write(m2, b);
                try (TieredIndex u = TieredIndex.open(dir, cfg())) {
                    assertNull(u.openFallbackCause(), what);
                    assertServes(u, want1);
                    assertTrue(!Files.exists(m2) && !Files.exists(seg(3)), what);
                    u.checkpoint(dir);
                }
                assertEquals(2, Manifest.readLatest(dir).generation(), what);
                clear();

                // a newer format: the same change with a matching CRC is refused, never skipped
                copy(twoGen, dir);
                Files.write(m2, withManifestCrc(b));
                Map<String, byte[]> before = snapshot(dir);
                assertInstanceOf(UnsupportedException.class, latestRefused(dir, what + ", matching CRC"));
                assertInstanceOf(UnsupportedException.class, openRefused(dir, what + ", matching CRC"));
                assertUnchanged(before, dir, what + ", matching CRC");
                clear();
            }
        }
    }

    @Test
    void aDamagedMajorVersionOfTheOnlyManifestDoesNotBlockTheNextCheckpoint() throws IOException {
        // #123: there is no older generation here, so open refuses the directory; the rebuilt tier must still be able to replace it
        copy(single, dir);
        Path m1 = dir.resolve(Manifest.manifestName(1));
        byte[] b = Files.readAllBytes(m1);
        b[8] ^= 1;
        Files.write(m1, b);
        assertInstanceOf(CorruptException.class, openRefused(dir, "the only manifest"));
        try (TieredIndex r = TieredIndex.open(dir, DIM, Metric.L2, cfg(), () -> records1)) {
            assertInstanceOf(CorruptException.class, r.openFallbackCause());
            r.sealNow();
            assertTrue(r.awaitIdle(60_000));
            r.checkpoint(dir);
        }
        try (TieredIndex u = TieredIndex.open(dir, cfg())) {
            assertEquals(2 * ROWS, u.size());
        }
    }

    @Test
    void aNewerMajorWithAShorterManifestBodyIsRefusedAsUnsupported() throws IOException {
        // only the frame is promised for a newer major: here a body of 16 bytes, shorter than this version's fixed fields
        copy(twoGen, dir);
        byte[] m = new byte[20];
        System.arraycopy(Files.readAllBytes(dir.resolve(Manifest.manifestName(2))), 0, m, 0, 8);
        SegmentForge.le(m).putShort(8, (short) 2).putShort(10, (short) 0).putInt(12, 16);
        Files.write(dir.resolve(Manifest.manifestName(2)), withManifestCrc(m));
        assertInstanceOf(UnsupportedException.class, latestRefused(dir, "short major 2"));
        // the same bytes as major 1 are damage: this version's fields do not fit
        SegmentForge.le(m).putShort(8, (short) 1);
        Files.write(dir.resolve(Manifest.manifestName(2)), withManifestCrc(m));
        assertEquals(1, Manifest.readLatest(dir).generation());
    }

    // ---- damage to the newest of two generations ----

    @Test
    void aFlippedSectionBitInTheNewestGenerationIsRefusedNotHiddenByTheOlderOne() throws IOException {
        for (int id : new int[] {1, 4, 12}) {
            copy(twoGen, dir);
            SegmentInfo.Section s = SegmentReader.inspect(seg(3)).section(id);
            byte[] b = Files.readAllBytes(seg(3));
            b[(int) (s.offset() + s.length() / 2)] ^= 0x04;
            Files.write(seg(3), b);
            Map<String, byte[]> before = snapshot(dir);
            // the header, footer and locator are intact, so generation 2 validates; its sections are read in full at open and one fails
            CorruptException e = assertInstanceOf(CorruptException.class, openRefused(dir, "section " + id));
            assertEquals(SegmentFormat.sectionName(id), e.section());
            assertUnchanged(before, dir, "section " + id);
            clear();
        }
    }

    @Test
    void damageToTheHeaderFooterOrLocatorOfTheNewestSegmentMakesTheOlderGenerationTheNewestThatValidates() throws IOException {
        // the documented rule: the newest manifest that validates is live; one whose segment fails the header, footer or locator check is skipped.
        // generation 1 is served as it was, and the files of generation 2 are removed by the open.
        int[] spots = {20, 40, -1, -1};
        for (int k = 0; k < spots.length; k++) {
            copy(twoGen, dir);
            byte[] b = Files.readAllBytes(seg(3));
            int pos = k < 2 ? spots[k] : k == 2 ? footerOff(b) + 10 : b.length - 10;
            b[pos] ^= 0x01;
            Files.write(seg(3), b);
            try (TieredIndex u = TieredIndex.open(dir, cfg())) {
                assertNull(u.openFallbackCause());
                assertEquals(2 * ROWS, u.size(), "byte " + pos);
                assertServes(u, want1);
            }
            Manifest live = Manifest.readLatest(dir);
            assertEquals(1, live.generation(), "byte " + pos);
            assertTrue(!Files.exists(seg(3)) && !Files.exists(dir.resolve(Manifest.manifestName(2))), "the files of the unfinished generation were removed");
            clear();
        }
    }
}
