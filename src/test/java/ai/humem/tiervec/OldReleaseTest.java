package ai.humem.tiervec;

import static ai.humem.tiervec.SegmentTestSupport.vectors;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.function.LongPredicate;
import java.util.function.Supplier;
import java.util.stream.Stream;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/**
 * The two directions between this code and the release 0.1.1, which does not know the persisted tail (#158), with the real jar of that
 * release (the build fetches it) in a class loader of its own: a directory that the release wrote opens here with both vector storage
 * options, and the release refuses a directory that this code wrote and changes nothing there.
 */
class OldReleaseTest {
    static final int DIM = TierReopenTest.DIM;
    static final float[][] DATA = vectors(700, DIM, 1591);
    static final float[][] QUERIES = vectors(30, DIM, 1592);

    static URLClassLoader release;

    @TempDir
    Path dir;

    @BeforeAll
    static void load() throws IOException {
        Path jar = Path.of(System.getProperty("tiervec.oldRelease", "target/old-release/tiervec-old-release.jar"));
        assertTrue(Files.isRegularFile(jar), "the jar of the old release is not at " + jar.toAbsolutePath() + ": run the tests through the build");
        release = new URLClassLoader(new URL[] {jar.toUri().toURL()}, ClassLoader.getPlatformClassLoader());
    }

    @AfterAll
    static void unload() throws IOException {
        release.close();
    }

    // ---- the release, through reflection ----

    private static Class<?> old(String name) throws ClassNotFoundException {
        Class<?> c = release.loadClass("ai.humem.tiervec." + name);
        assertTrue(c.getClassLoader() == release, name + " comes from the jar of the release");
        return c;
    }

    private static Object call(Object target, Class<?> type, String method, Class<?>[] types, Object... args) throws Throwable {
        try {
            return type.getMethod(method, types).invoke(target, args);
        } catch (InvocationTargetException e) {
            throw e.getCause();
        }
    }

    private static Object call(Object tier, String method, Class<?>[] types, Object... args) throws Throwable {
        return call(tier, tier.getClass(), method, types, args);
    }

    private static Object newTier() throws Throwable {
        Class<?> metric = old("Metric");
        Object l2 = call(null, metric, "valueOf", new Class<?>[] {String.class}, "L2");
        return old("TieredIndex").getConstructor(int.class, metric).newInstance(DIM, l2);
    }

    private static void add(Object tier, int from, int to) throws Throwable {
        for (int i = from; i < to; i++) call(tier, "add", new Class<?>[] {long.class, float[].class}, (long) i, DATA[i]);
    }

    private static void sealAndCheckpoint(Object tier, Path d) throws Throwable {
        call(tier, "sealNow", new Class<?>[0]);
        call(tier, "checkpoint", new Class<?>[] {Path.class}, d);
    }

    private static Object open(Path d) throws Throwable {
        return call(null, old("TieredIndex"), "open", new Class<?>[] {Path.class, old("TieredIndex$Config")}, d, null);
    }

    private static int size(Object tier) throws Throwable {
        return (int) call(tier, "size", new Class<?>[0]);
    }

    private static void close(Object tier) throws Throwable {
        call(tier, "close", new Class<?>[0]);
    }

    /** the answers of the release, in the form of {@link TierHaltTest#digest} */
    private static List<String> digest(Object tier) throws Throwable {
        List<String> lines = new ArrayList<>();
        for (float[] q : QUERIES) {
            Object[] hits = (Object[]) call(tier, "search", new Class<?>[] {float[].class, int.class, LongPredicate.class}, q, 10, null);
            StringBuilder sb = new StringBuilder();
            for (Object h : hits) {
                if (sb.length() > 0) sb.append(' ');
                sb.append(call(h, "id", new Class<?>[0])).append(':').append(Float.floatToIntBits((float) call(h, "score", new Class<?>[0])));
            }
            lines.add(sb.toString());
        }
        return lines;
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

    // ---- the tests ----

    @ParameterizedTest
    @EnumSource(TieredIndex.VectorStorage.class)
    void aDirectoryThatTheReleaseWroteOpensAndTakesATail(TieredIndex.VectorStorage storage) throws Throwable {
        // Windows: the second storage option gets its own directory from JUnit, so a mapped file of one run is not in the way of another
        Object writer = newTier();
        add(writer, 0, 300);
        sealAndCheckpoint(writer, dir);
        add(writer, 300, 600);
        sealAndCheckpoint(writer, dir);
        List<String> want = digest(writer);
        close(writer);
        Manifest old = Manifest.readLatest(dir);
        assertEquals(2, old.generation());
        assertEquals(0, old.tailRows());
        assertEquals(1, Files.readAllBytes(dir.resolve(Manifest.manifestName(2)))[8], "the release writes major 1");

        TieredIndex.Config config = TieredIndex.Config.defaults().withVectorStorage(storage);
        List<String> served;
        try (TieredIndex t = TieredIndex.open(dir, config)) {
            assertNull(t.openFallbackCause());
            assertEquals(600, t.size());
            assertEquals(0, t.pendingTail());
            TierHaltTest.assertSameAnswers(want, TierHaltTest.digest(t, QUERIES), "the release's directory");
            for (int i = 600; i < 620; i++) t.add(i, DATA[i]);
            t.delete(7);
            served = TierHaltTest.digest(t, QUERIES);
            t.checkpoint(dir);
        }
        try (TieredIndex u = TieredIndex.open(dir, config)) {
            assertEquals(619, u.size());
            assertEquals(20, u.pendingTail());
            assertEquals(served, TierHaltTest.digest(u, QUERIES));
        }
    }

    @Test
    void theReleaseRefusesADirectoryWithATailAndChangesNothing() throws Throwable {
        try (TieredIndex t = new TieredIndex(DIM, Metric.L2)) {
            for (int i = 0; i < 600; i++) t.add(i, DATA[i]);
            t.sealNow();
            for (int i = 600; i < 650; i++) t.add(i, DATA[i]);
            t.delete(3);
            t.checkpoint(dir);
            assertTrue(Manifest.readLatest(dir).tailRows() > 0);
        }
        Map<String, byte[]> before = snapshot(dir);
        Class<?> unsupported = old("SegmentFormat$UnsupportedException");

        // open refuses
        Throwable refused = assertThrows(Throwable.class, () -> open(dir));
        assertEquals(unsupported, refused.getClass(), String.valueOf(refused));
        assertUnchanged(before, dir, "after the refused open");

        // the open that holds records builds from them: with none it serves nothing, not the segments without their tail
        Supplier<List<Object>> noRecords = List::of;
        Class<?> metric = old("Metric");
        Object l2 = call(null, metric, "valueOf", new Class<?>[] {String.class}, "L2");
        Object cold = call(null, old("TieredIndex"), "open", new Class<?>[] {Path.class, int.class, metric, old("TieredIndex$Config"), Supplier.class},
                dir, DIM, l2, null, noRecords);
        assertEquals(0, size(cold));
        assertEquals(unsupported, call(cold, "openFallbackCause", new Class<?>[0]).getClass());

        // and no checkpoint of the release replaces the directory
        add(cold, 0, 30);
        Throwable kept = assertThrows(Throwable.class, () -> sealAndCheckpoint(cold, dir));
        assertEquals(unsupported, kept.getClass(), String.valueOf(kept));
        close(cold);
        assertUnchanged(before, dir, "after the refused checkpoint");
    }
}
