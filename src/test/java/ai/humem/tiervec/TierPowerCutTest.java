package ai.humem.tiervec;

import static ai.humem.tiervec.SegmentTestSupport.vectors;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.function.Consumer;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * The directories that a power cut can leave during a checkpoint, opened with {@code TieredIndex.open}. Each case starts from the bytes that a
 * process kill after one step of the publish leaves (the fault hook only looks: it keeps every file as it is at that step) and removes what the
 * order of the publish does not yet make durable at that step: a name that is not flushed (the new segment before the directory flush, the
 * temporary manifest, the rename), and the bytes of a file that is not flushed (the temporary manifest). The cases that follow break the order
 * on purpose: the rename or the name reached the disk and the bytes of the manifest, the new segment or the tombstone file did not.
 *
 * <p>Every case must open as exactly the generation that is named in the case (the one before the publish, or the new one), serve the answers
 * of that generation's in-memory index, leave exactly the files of that generation after the open, and accept a following checkpoint.
 * Nothing else may happen: no other exception, no mixture of the two generations.
 *
 * <p>What this does not prove: that the library issues the flushes in this order. The cases take the order of {@code SEGMENT-FORMAT.md} as given
 * and test what the readers do with the states it allows, and with some that it does not. A file system that records writes, flushes and renames
 * and drops what was not flushed at a crash would test the order itself; this version has none, and the claim stays "process kill, not power loss".
 */
class TierPowerCutTest {
    static final int DIM = TierReopenTest.DIM;
    static final int ROWS = 400;

    /** the directory that the second checkpoint publishes into (generation 1 is there first) */
    @TempDir
    static Path live;

    @TempDir
    Path dir;

    static TieredIndex tier;
    static float[][] data, qs;
    /** what generation 1 (segments 1 and 2) and generation 2 (segments 1, 2 and 3) serve, from the in-memory index */
    static Hit[][] want1, want2;
    /** every file of the directory at each step of the publish of generation 2 */
    static final Map<FaultHook.Step, Map<String, byte[]>> images = new EnumMap<>(FaultHook.Step.class);

    @BeforeAll
    static void build() throws IOException {
        data = vectors(3 * ROWS, DIM, 291);
        qs = vectors(40, DIM, 292);
        tier = new TieredIndex(DIM, Metric.L2, TierReopenTest.cfg());
        batch(0);
        batch(1);
        tier.checkpoint(live); // generation 1
        want1 = results(tier);
        batch(2);
        want2 = results(tier);
        tier.checkpoint(live, step -> images.put(step, snapshot(live))); // generation 2 adds segment 3
        assertEquals(EnumSet.allOf(FaultHook.Step.class), images.keySet());
        assertEquals(2, Manifest.readLatest(live).generation());
        assertEquals(3, Manifest.readLatest(live).entries().size());
        assertTrue(images.get(FaultHook.Step.SEGMENTS_WRITTEN).containsKey(Manifest.segName(3)));
        assertTrue(images.get(FaultHook.Step.MANIFEST_TMP_WRITTEN).containsKey(Manifest.manifestName(2) + ".tmp"));
        assertTrue(images.get(FaultHook.Step.MANIFEST_RENAMED).containsKey(Manifest.manifestName(2)));
    }

    @AfterAll
    static void close() {
        tier.close();
    }

    private static void batch(int b) {
        for (int i = b * ROWS; i < (b + 1) * ROWS; i++) tier.add(SegmentTestSupport.idOf(i), data[i]);
        tier.sealNow();
        assertTrue(tier.awaitIdle(60_000));
    }

    private static Hit[][] results(TieredIndex t) {
        Hit[][] out = new Hit[qs.length][];
        for (int i = 0; i < qs.length; i++) out[i] = t.search(qs[i], 10, null);
        return out;
    }

    private static void assertServes(TieredIndex u, Hit[][] want, String what) {
        for (int i = 0; i < qs.length; i++) assertArrayEquals(want[i], u.search(qs[i], 10, null), what + ", query " + i);
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

    private static Set<String> expected(Manifest m) {
        Set<String> s = new TreeSet<>();
        s.add(Manifest.manifestName(m.generation()));
        s.add(Manifest.tombName(m.tombGeneration()));
        for (Manifest.Entry e : m.entries()) s.add(Manifest.segName(e.segId()));
        return s;
    }

    private static Set<String> names(Path d) throws IOException {
        try (Stream<Path> s = Files.list(d)) {
            return s.map(p -> p.getFileName().toString()).filter(n -> !n.equals(DirectoryLock.FILE)).collect(Collectors.toCollection(TreeSet::new));
        }
    }

    // ---- the states ----

    /** one state: the step it starts from, what the cut changes in the files of that step, and the generation that must be live afterwards */
    record Cut(String name, FaultHook.Step step, Consumer<Map<String, byte[]>> change, long generation) {
        @Override
        public String toString() {
            return name;
        }
    }

    private static final UnaryOperator<byte[]> EMPTY = b -> new byte[0];
    private static final UnaryOperator<byte[]> HALF = b -> Arrays.copyOf(b, b.length / 2);
    private static final UnaryOperator<byte[]> ZEROS = b -> new byte[b.length]; // the length reached the disk and the bytes did not

    /** the name is not there after the cut */
    private static Consumer<Map<String, byte[]>> lose(String name) {
        return f -> assertNotNull(f.remove(name), name + " is in the directory at this step");
    }

    /** the file is there after the cut with other bytes */
    private static Consumer<Map<String, byte[]>> change(String name, UnaryOperator<byte[]> bytes) {
        return f -> {
            byte[] b = f.get(name);
            assertNotNull(b, name + " is in the directory at this step");
            f.put(name, bytes.apply(b));
        };
    }

    /** the rename is not there after the cut: the bytes are under the old name */
    private static Consumer<Map<String, byte[]>> undoRename(String from, String to) {
        return f -> {
            byte[] b = f.remove(from);
            assertNotNull(b, from + " is in the directory at this step");
            assertNull(f.put(to, b), to + " is not in the directory at this step");
        };
    }

    static Stream<Cut> cuts() {
        final String m2 = Manifest.manifestName(2), tmp = m2 + ".tmp", seg3 = Manifest.segName(3), tomb2 = Manifest.tombName(2);
        final FaultHook.Step renamed = FaultHook.Step.MANIFEST_RENAMED;
        List<Cut> l = new ArrayList<>();
        // a process kill after each step: the files are as they are
        for (FaultHook.Step s : FaultHook.Step.values()) {
            l.add(new Cut("process kill after " + s, s, f -> { }, s.ordinal() >= renamed.ordinal() ? 2 : 1));
        }
        // the names are not flushed before the directory flush of step 3: the new segment may be gone
        l.add(new Cut("power cut after SEGMENTS_WRITTEN, the new segment's name is lost", FaultHook.Step.SEGMENTS_WRITTEN, lose(seg3), 1));
        // the bytes of the temporary manifest are not flushed before step 5, and its name is not flushed at all before the rename
        l.add(new Cut("power cut after MANIFEST_TMP_WRITTEN, the temporary manifest is lost", FaultHook.Step.MANIFEST_TMP_WRITTEN, lose(tmp), 1));
        l.add(new Cut("power cut after MANIFEST_TMP_WRITTEN, the temporary manifest is empty", FaultHook.Step.MANIFEST_TMP_WRITTEN, change(tmp, EMPTY), 1));
        l.add(new Cut("power cut after MANIFEST_TMP_WRITTEN, half of the temporary manifest is left", FaultHook.Step.MANIFEST_TMP_WRITTEN, change(tmp, HALF), 1));
        l.add(new Cut("power cut after MANIFEST_TMP_WRITTEN, the temporary manifest is zeros", FaultHook.Step.MANIFEST_TMP_WRITTEN, change(tmp, ZEROS), 1));
        l.add(new Cut("power cut after MANIFEST_TMP_SYNCED, the name of the temporary manifest is lost", FaultHook.Step.MANIFEST_TMP_SYNCED, lose(tmp), 1));
        // the rename is not flushed before step 7: the old name or the new one, so here the old one
        l.add(new Cut("power cut after MANIFEST_RENAMED, the rename is lost", renamed, undoRename(m2, tmp), 1));
        // the order is broken: the rename or the name reached the disk and the bytes did not. The readers must find it and use generation 1
        for (FaultHook.Step s : new FaultHook.Step[] {renamed, FaultHook.Step.DIR_SYNCED_AFTER_RENAME}) {
            final String at = " after " + s + ", ";
            l.add(new Cut("broken order" + at + "the manifest is empty", s, change(m2, EMPTY), 1));
            l.add(new Cut("broken order" + at + "half of the manifest is left", s, change(m2, HALF), 1));
            l.add(new Cut("broken order" + at + "the manifest is zeros", s, change(m2, ZEROS), 1));
            l.add(new Cut("broken order" + at + "the new segment is empty", s, change(seg3, EMPTY), 1));
            l.add(new Cut("broken order" + at + "half of the new segment is left", s, change(seg3, HALF), 1));
            l.add(new Cut("broken order" + at + "the tombstone file is lost", s, lose(tomb2), 1));
            l.add(new Cut("broken order" + at + "half of the tombstone file is left", s, change(tomb2, HALF), 1));
        }
        return l.stream();
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("cuts")
    void aCutLeavesExactlyOneOfTheTwoGenerationsAndItServesTheSameAnswers(Cut cut) throws IOException {
        Map<String, byte[]> files = new TreeMap<>(images.get(cut.step()));
        cut.change().accept(files);
        for (Map.Entry<String, byte[]> e : files.entrySet()) Files.write(dir.resolve(e.getKey()), e.getValue());

        final long gen = cut.generation();
        final Hit[][] want = gen == 2 ? want2 : want1;
        Manifest latest = Manifest.readLatest(dir);
        assertNotNull(latest, cut.name());
        assertEquals(gen, latest.generation(), cut.name());
        try (TieredIndex u = TieredIndex.open(dir, TierReopenTest.cfg())) {
            assertNull(u.openFallbackCause(), cut.name());
            assertEquals((int) (gen + 1) * ROWS, u.size(), cut.name());
            assertServes(u, want, cut.name());
            assertEquals(expected(latest), names(dir), cut.name() + ": open removed every file of the other generation and every temporary file");
            // the work goes on from here: the next checkpoint is the next generation, and it opens
            u.checkpoint(dir);
        }
        assertEquals(gen + 1, Manifest.readLatest(dir).generation(), cut.name());
        try (TieredIndex v = TieredIndex.open(dir, TierReopenTest.cfg())) {
            assertServes(v, want, cut.name() + ", after the next checkpoint");
        }
    }
}
