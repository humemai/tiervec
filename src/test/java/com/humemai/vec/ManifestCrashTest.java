package com.humemai.vec;

import static com.humemai.vec.SegmentTestSupport.graph;
import static com.humemai.vec.SegmentTestSupport.params;
import static com.humemai.vec.SegmentTestSupport.vectors;
import static com.humemai.vec.SegmentTestSupport.versOf;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.humemai.vec.SegmentFormat.CorruptException;
import com.humemai.vec.SegmentFormat.UnsupportedException;

import java.io.IOException;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * The manifest and its publish order. A process kill after a step leaves exactly the files up to that step (the fault hook throws there
 * and the publisher cleans up nothing), so these tests cover process kills. They do not cover power loss: a simulated file system that
 * drops data that was not flushed is not part of this version.
 */
class ManifestCrashTest {
    @TempDir
    Path dir;

    private final Manifest.Fingerprint fp = new Manifest.Fingerprint(8, Metric.L2, 16, 64, false);
    private float[][] qs;
    private SealedGraph g1, g2;

    @BeforeEach
    void build() {
        g1 = graph(vectors(300, 8, 1), Metric.L2, params(false));
        g2 = graph(vectors(300, 8, 2), Metric.L2, params(false));
        qs = vectors(20, 8, 77);
    }

    private Manifest.Entry write(long id, SealedGraph g) throws IOException {
        SegmentInfo si = SegmentWriter.write(g, versOf(g.n), dir.resolve(Manifest.segName(id)));
        return new Manifest.Entry(id, si.n(), si.fileLength(), si.headerCrc(), si.verMin(), si.verMax());
    }

    private Set<String> names() throws IOException {
        try (Stream<Path> s = Files.list(dir)) {
            return s.map(p -> p.getFileName().toString()).collect(Collectors.toCollection(TreeSet::new));
        }
    }

    private static Set<String> expected(Manifest m) {
        Set<String> s = new TreeSet<>();
        s.add(Manifest.manifestName(m.generation()));
        s.add(Manifest.tombName(m.tombGeneration()));
        for (Manifest.Entry e : m.entries()) s.add(Manifest.segName(e.segId()));
        return s;
    }

    private void assertServes(Manifest m) throws IOException {
        for (Manifest.Entry e : m.entries()) {
            SealedGraph orig = e.segId() == 1 ? g1 : g2;
            SealedGraph loaded = SegmentReader.read(dir.resolve(Manifest.segName(e.segId())), true);
            for (float[] q : qs) assertArrayEquals(orig.search(q, 10, null), loaded.search(q, 10, null));
        }
    }

    /** generation 1 names segment 1 and is published and cleaned; segment 2 is written and generation 2 is ready to publish */
    private Manifest generationOne() throws IOException {
        Manifest.Entry e1 = write(1, g1);
        Manifest m1 = Manifest.publish(dir, Manifest.of(1, fp, List.of(e1)), FaultHook.NONE);
        assertEquals(0, Manifest.gc(dir, m1));
        return m1;
    }

    private Manifest generationTwoInput(Manifest m1) throws IOException {
        Manifest.Entry e2 = write(2, g2);
        return Manifest.of(2, fp, List.of(m1.entries().get(0), e2));
    }

    // ---- U5: a crash after every step of the publish ----

    @ParameterizedTest(name = "crash after {0}")
    @EnumSource(FaultHook.Step.class)
    void crashAfterEachStepLeavesOneGenerationAndNoOrphansSurviveGc(FaultHook.Step crashAt) throws IOException {
        Manifest m1 = generationOne();
        Manifest in2 = generationTwoInput(m1);
        IOException crash = assertThrows(IOException.class, () -> Manifest.publish(dir, in2, step -> {
            if (step == crashAt) throw new IOException("simulated crash after " + step);
        }));
        assertTrue(crash.getMessage().contains(crashAt.name()));

        Manifest got = Manifest.readLatest(dir);
        assertNotNull(got);
        assertTrue(got.generation() == 1 || got.generation() == 2, "generation " + got.generation());
        boolean visible = crashAt.ordinal() >= FaultHook.Step.MANIFEST_RENAMED.ordinal();
        assertEquals(visible ? 2 : 1, got.generation());
        assertEquals(visible ? 2 : 1, got.entries().size());
        assertServes(got);

        // the next open removes every file the live generation does not name, and nothing else
        Files.write(dir.resolve("keep.txt"), new byte[] {1, 2, 3});
        Manifest.gc(dir, got);
        Set<String> want = expected(got);
        want.add("keep.txt");
        assertEquals(want, names());
        assertServes(got);
    }

    @ParameterizedTest(name = "retry after a crash after {0}")
    @EnumSource(value = FaultHook.Step.class, names = {"SEGMENTS_WRITTEN", "DIR_SYNCED_AFTER_SEGMENTS", "MANIFEST_TMP_WRITTEN", "MANIFEST_TMP_SYNCED"})
    void publishCanBeRetriedAfterACrashBeforeTheRename(FaultHook.Step crashAt) throws IOException {
        Manifest m1 = generationOne();
        Manifest in2 = generationTwoInput(m1);
        assertThrows(IOException.class, () -> Manifest.publish(dir, in2, step -> {
            if (step == crashAt) throw new IOException("crash");
        }));
        assertEquals(1, Manifest.readLatest(dir).generation());
        Manifest m2 = Manifest.publish(dir, in2, FaultHook.NONE); // leftovers of the crashed attempt are overwritten
        assertEquals(2, Manifest.readLatest(dir).generation());
        assertEquals(m2, Manifest.readLatest(dir));
        assertServes(m2);
    }

    @Test
    void crashBeforeTheCleanupLeavesTheOldGenerationToBeDeleted() throws IOException {
        Manifest m1 = generationOne();
        Manifest m2 = Manifest.publish(dir, generationTwoInput(m1), FaultHook.NONE);
        // the process dies here: the in-memory switch and the cleanup never ran
        assertEquals(2, Manifest.readLatest(dir).generation());
        assertTrue(names().contains(Manifest.manifestName(1)));
        assertEquals(2, Manifest.gc(dir, m2), "the old manifest and the old tombstone file");
        assertEquals(expected(m2), names());
        assertServes(m2);
    }

    @Test
    void aSegmentCutShortWithNoManifestIsAnOrphan() throws IOException {
        Manifest m1 = generationOne();
        // a crash during the segment write: part of a segment file, no manifest names it
        byte[] seg = Files.readAllBytes(dir.resolve(Manifest.segName(1)));
        Files.write(dir.resolve(Manifest.segName(3)), Arrays.copyOf(seg, 5000));
        assertEquals(m1, Manifest.readLatest(dir));
        assertEquals(1, Manifest.gc(dir, m1));
        assertEquals(expected(m1), names());
    }

    @Test
    void publishRefusesAGenerationThatExists() throws IOException {
        Manifest m1 = generationOne();
        assertThrows(FileAlreadyExistsException.class, () -> Manifest.publish(dir, Manifest.of(1, fp, m1.entries()), FaultHook.NONE));
        assertEquals(m1, Manifest.readLatest(dir));
    }

    // ---- U6: the newest manifest is damaged ----

    /** both generations published, nothing cleaned (the state between steps 7 and 8) */
    private Manifest twoGenerations() throws IOException {
        Manifest m1 = generationOne();
        Manifest m2 = Manifest.publish(dir, generationTwoInput(m1), FaultHook.NONE);
        assertEquals(m2, Manifest.readLatest(dir));
        return m1;
    }

    @ParameterizedTest(name = "damage {0}")
    @ValueSource(ints = {0, 1, 2, 3, 4})
    void aDamagedNewestManifestFallsBackToThePreviousGeneration(int kind) throws IOException {
        Manifest m1 = twoGenerations();
        Path m2 = dir.resolve(Manifest.manifestName(2));
        byte[] raw = Files.readAllBytes(m2);
        switch (kind) {
            case 0: // one flipped bit
                raw[raw.length / 2] ^= 0x04;
                Files.write(m2, raw);
                break;
            case 1: // cut in half
                Files.write(m2, Arrays.copyOf(raw, raw.length / 2));
                break;
            case 2: // intact manifest, a segment it names is gone
                Files.delete(dir.resolve(Manifest.segName(2)));
                break;
            case 3: // intact manifest, a segment it names was cut short
                Files.write(dir.resolve(Manifest.segName(2)), Arrays.copyOf(Files.readAllBytes(dir.resolve(Manifest.segName(2))), 9000));
                break;
            default: // intact manifest, its tombstone file is gone
                Files.delete(dir.resolve(Manifest.tombName(2)));
                break;
        }
        Manifest got = Manifest.readLatest(dir);
        assertEquals(m1, got);
        assertServes(got);
        Manifest.gc(dir, got);
        assertEquals(expected(m1), names());
    }

    @Test
    void whenNoManifestValidatesReadLatestSaysSo() throws IOException {
        assertNull(Manifest.readLatest(dir), "an empty directory has no manifest");
        twoGenerations();
        for (long gen : new long[] {1, 2}) {
            Path p = dir.resolve(Manifest.manifestName(gen));
            byte[] raw = Files.readAllBytes(p);
            raw[10] ^= 0x01;
            Files.write(p, raw);
        }
        CorruptException e = assertThrows(CorruptException.class, () -> Manifest.readLatest(dir));
        assertTrue(e.getMessage().contains("2 candidates"), e.getMessage());
    }

    @Test
    void aManifestInANewerFormatIsRefusedNotSkipped() throws IOException {
        Manifest m1 = generationOne();
        Manifest m2 = Manifest.publish(dir, generationTwoInput(m1), FaultHook.NONE);
        // a larger minor opens
        Files.write(dir.resolve(Manifest.manifestName(2)), m2.encode(Manifest.FORMAT_MAJOR, Manifest.FORMAT_MINOR + 4));
        assertEquals(m2, Manifest.readLatest(dir));
        // a larger major is refused even though generation 1 is intact: serving it would silently be stale
        Manifest m3 = new Manifest(3, Manifest.NONE, Manifest.NONE, 0L, 3L, fp, m2.entries(), 2L, m2.tombLength(), m2.tombCrc());
        Files.write(dir.resolve(Manifest.manifestName(3)), m3.encode(Manifest.FORMAT_MAJOR + 1, 0));
        assertThrows(UnsupportedException.class, () -> Manifest.readLatest(dir));
    }

    @Test
    void manifestFieldsRoundTrip() throws IOException {
        Manifest m1 = generationOne();
        Manifest m2 = Manifest.publish(dir, generationTwoInput(m1), FaultHook.NONE);
        Manifest back = Manifest.readLatest(dir);
        assertEquals(m2, back);
        assertEquals(fp, back.fingerprint());
        assertEquals(Manifest.NONE, back.epoch());
        assertEquals(Manifest.NONE, back.hwm());
        assertEquals(3, back.nextSegId());
        assertEquals(2, back.tombGeneration());
    }
}
