package ai.humem.tiervec;

import static ai.humem.tiervec.SegmentTestSupport.idOf;
import static ai.humem.tiervec.SegmentTestSupport.vectors;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.DisabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.junit.jupiter.api.io.TempDir;

/**
 * A crash after every change that a publish makes to the file system, on a simulated file system ({@link CrashFs}) that keeps what was
 * flushed and may keep or lose everything else. Three checkpoints are recorded: the first generation, a second that adds a segment, and a
 * third after deletes and a merge, which replaces the segments and so deletes old files. Then, for every prefix of the recorded changes, every
 * combination of the name changes that no directory flush made durable is tried with every content that unflushed file bytes may have
 * (flushed only, all written, cut halfway, zeros, torn blocks). Each of these directories must open as exactly the generation before the
 * publish or exactly the one being published (none, before the first one is there), answer the queries exactly as the in-memory index of
 * that generation did, leave no file of another generation behind, and take the next checkpoint.
 *
 * <p>To show that the test can fail, it also removes each flush of the recorded publishes in turn and expects a broken directory every time.
 *
 * <p>The model is described in {@link CrashFs}. It tests the order of the flushes that the library issues; it says nothing about a file
 * system or a drive that does not keep what it was asked to flush.
 */
@DisabledOnOs(value = OS.WINDOWS, disabledReason = "the library cannot flush a directory on Windows (docs/SEGMENT-FORMAT.md), so there is no order to hold it to")
class SimulatedFsCrashTest {
    static final int DIM = TierReopenTest.DIM;
    static final int ROWS = 120;
    static final String INDEX = "index";
    /** the most unflushed name changes of one crash point; all their combinations are tried */
    static final int MAX_PENDING = 12;
    private static final Pattern MANIFEST = Pattern.compile("manifest-(\\d{20})\\.tvm");

    /** what one generation serves */
    record Generation(int size, Hit[][] hits) {}

    /** the recorded changes of a run, the index directory below the recorded one, and what each generation in it serves */
    record Recording(String index, List<CrashFs.Op> log, Map<Long, Generation> generations) {}

    @TempDir
    static Path scratch;

    static float[][] data, qs;
    static Recording published;

    @BeforeAll
    static void record() throws IOException {
        data = vectors(2 * ROWS, DIM, 171);
        qs = vectors(25, DIM, 172);
        CrashFs fs = new CrashFs(Files.createTempDirectory(scratch, "record"));
        Path index = fs.root().resolve(INDEX);
        // the host makes the directory and does not flush its name: checkpoint does that before the first generation (#126)
        Files.createDirectory(index);
        published = publishThreeGenerations(fs, index);
    }

    static Recording publishThreeGenerations(CrashFs fs, Path index) throws IOException {
        Map<Long, Generation> generations = new HashMap<>();
        try (TieredIndex t = new TieredIndex(DIM, Metric.L2, TierReopenTest.cfg())) {
            for (int b = 0; b < 2; b++) {
                for (int i = b * ROWS; i < (b + 1) * ROWS; i++) t.add(idOf(i), data[i]);
                t.sealNow();
                assertTrue(t.awaitIdle(60_000));
                generations.put(b + 1L, generation(t));
                t.checkpoint(index);
            }
            for (int i = 0; i < 2 * ROWS; i += 5) t.delete(idOf(i));
            t.mergeNow();
            assertTrue(t.awaitIdle(60_000));
            generations.put(3L, generation(t));
            t.checkpoint(index);
        }
        Manifest last = Manifest.readLatest(index);
        assertEquals(3, last.generation());
        assertEquals(1, last.entries().size(), "the merge left one segment, so the third publish deleted the two older ones");
        return new Recording(fs.root().relativize(index).toString(), List.copyOf(fs.log), generations);
    }

    private static Generation generation(TieredIndex t) {
        Hit[][] hits = new Hit[qs.length][];
        for (int i = 0; i < qs.length; i++) hits[i] = t.search(qs[i], 10, null);
        return new Generation(t.size(), hits);
    }

    // ---- what may be served ----

    /**
     * The generations that a crash after the first {@code k} changes may leave, 0 for none: the newest one whose rename a directory flush has
     * made durable, and any later one that has been renamed into place.
     */
    static Set<Long> allowed(List<CrashFs.Op> log, int k) {
        long durable = 0;
        List<long[]> renamed = new ArrayList<>(); // directory node, generation
        for (CrashFs.Op op : log.subList(0, k)) {
            if (op.kind() == CrashFs.Kind.RENAME) {
                Matcher m = MANIFEST.matcher(op.to());
                if (m.matches()) renamed.add(new long[] {op.dir(), Long.parseLong(m.group(1))});
            } else if (op.kind() == CrashFs.Kind.SYNC_DIR) {
                for (long[] r : renamed) {
                    if (r[0] == op.node()) durable = Math.max(durable, r[1]);
                }
            }
        }
        Set<Long> out = new TreeSet<>();
        out.add(durable);
        for (long[] r : renamed) {
            if (r[1] > durable) out.add(r[1]);
        }
        return out;
    }

    // ---- every crash ----

    /** counts what the crashes showed, so that a test can tell that it looked at what it meant to */
    static final class Tally {
        final Set<String> seen = new HashSet<>();
        final Set<Long> served = new TreeSet<>();
        final List<String> broken = new ArrayList<>();
        int opened;
    }

    /**
     * Crashes after each of the changes {@code from} to {@code to} in every way the model allows. What survives is taken from {@code model}, what
     * is allowed to be served from {@code truth}; the two differ only when a test takes a flush out of the model.
     */
    static void crashEverywhere(Recording rec, List<CrashFs.Op> truth, List<CrashFs.Op> model, int from, int to, Tally tally) throws IOException {
        for (int k = from; k <= to; k++) {
            CrashFs.Disk disk = CrashFs.replay(model, k);
            final int pending = disk.pending();
            assertTrue(pending <= MAX_PENDING, pending + " unflushed name changes after change " + k + " are too many to try every combination");
            Set<Long> allowed = allowed(truth, k);
            for (int mask = 0; mask < 1 << pending; mask++) {
                boolean[] kept = new boolean[pending];
                for (int i = 0; i < pending; i++) kept[i] = (mask & 1 << i) != 0;
                for (CrashFs.Content content : CrashFs.Content.values()) {
                    CrashFs.Image image = disk.image(kept, content);
                    if (!tally.seen.add(allowed + digest(image))) continue; // this directory was tried with this expectation
                    String wrong = openAndCompare(rec, image, allowed, tally);
                    if (wrong != null) {
                        List<CrashFs.Op> ops = disk.pendingOps();
                        List<CrashFs.Op> lost = new ArrayList<>();
                        for (int i = 0; i < pending; i++) {
                            if (!kept[i]) lost.add(ops.get(i));
                        }
                        tally.broken.add("crash after change " + k + (k > 0 ? " (" + model.get(k - 1) + ")" : "") + ", unflushed bytes " + content + ", lost " + lost + ": " + wrong);
                    }
                }
            }
        }
    }

    private static String digest(CrashFs.Image image) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            for (Map.Entry<String, CrashFs.Item> e : image.entries.entrySet()) {
                md.update((e.getKey() + "\n" + (e.getValue().bytes() == null ? -1 : e.getValue().bytes().length) + "\n").getBytes(java.nio.charset.StandardCharsets.UTF_8));
                if (e.getValue().bytes() != null) md.update(e.getValue().bytes());
            }
            return HexFormat.of().formatHex(md.digest());
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
    }

    /** null when the directory behaves as the contract says, otherwise what is wrong with it */
    private static String openAndCompare(Recording rec, CrashFs.Image image, Set<Long> allowed, Tally tally) throws IOException {
        Path dir = Files.createTempDirectory(scratch, "crash");
        try {
            image.writeTo(dir, (path, node) -> { });
            Path index = dir.resolve(rec.index());
            try (TieredIndex u = TieredIndex.open(index, TierReopenTest.cfg())) {
                tally.opened++;
                Manifest m = Manifest.readLatest(index);
                if (!allowed.contains(m.generation())) return "open served generation " + m.generation() + ", allowed " + allowed;
                tally.served.add(m.generation());
                Generation want = rec.generations().get(m.generation());
                if (u.size() != want.size()) return "generation " + m.generation() + " has " + u.size() + " rows, expected " + want.size();
                for (int i = 0; i < qs.length; i++) {
                    if (!Arrays.equals(want.hits()[i], u.search(qs[i], 10, null))) return "generation " + m.generation() + " answers query " + i + " differently";
                }
                Set<String> files = names(index);
                if (!files.equals(expected(m))) return "after open the directory holds " + files + ", expected " + expected(m);
                u.checkpoint(index);
                long next = Manifest.readLatest(index).generation();
                return next == m.generation() + 1 ? null : "the next checkpoint made generation " + next;
            } catch (NoSuchFileException e) {
                tally.served.add(0L);
                return allowed.contains(0L) ? null : "no generation left, allowed " + allowed + ": " + e.getMessage();
            } catch (IOException | RuntimeException e) {
                return "open failed, allowed " + allowed + ": " + e;
            }
        } finally {
            try (Stream<Path> walk = Files.walk(dir)) {
                for (Path p : walk.sorted(Comparator.reverseOrder()).collect(Collectors.toList())) Files.delete(p);
            }
        }
    }

    private static Set<String> expected(Manifest m) {
        Set<String> s = new TreeSet<>();
        s.add(Manifest.manifestName(m.generation()));
        s.add(Manifest.tombName(m.tombGeneration()));
        if (m.tailRows() > 0) s.add(Manifest.tailName(m.generation()));
        for (Manifest.Entry e : m.entries()) s.add(Manifest.segName(e.segId()));
        return s;
    }

    private static Set<String> names(Path d) throws IOException {
        try (Stream<Path> s = Files.list(d)) {
            return s.map(p -> p.getFileName().toString()).filter(n -> !n.equals(DirectoryLock.FILE)).collect(Collectors.toCollection(TreeSet::new));
        }
    }

    static String report(Tally tally) {
        return tally.broken.size() + " broken directories, the first ones:\n" + tally.broken.stream().limit(8).collect(Collectors.joining("\n"));
    }

    // ---- the tests ----

    @Test
    void aCrashAfterAnyChangeOfAPublishLeavesTheGenerationBeforeOrTheOneBeingPublished() throws IOException {
        Tally tally = new Tally();
        crashEverywhere(published, published.log(), published.log(), 0, published.log().size(), tally);
        assertTrue(tally.broken.isEmpty(), report(tally));
        assertEquals(Set.of(0L, 1L, 2L, 3L), tally.served, "every state of the run was reached");
        assertTrue(tally.opened > 50, "directories opened: " + tally.opened);
    }

    @Test
    void aKillDuringAPublishThenAPowerCutDuringTheNextOpenLeavesOneOfTheTwoGenerations() throws IOException {
        // #125: the process dies after any change of the run, so nothing that it wrote is lost yet, and the next process opens the directory. The
        // open may find a generation whose rename no flush has made durable, and it deletes the files of the one before. A power cut during
        // or after that open must still find one of the two
        Tally tally = new Tally();
        int opensThatChangedSomething = 0;
        for (int k = 0; k <= published.log().size(); k++) {
            CrashFs fs = CrashFs.resume(published.log().subList(0, k), Files.createTempDirectory(scratch, "resume"));
            try (TieredIndex u = TieredIndex.open(fs.root().resolve(INDEX), TierReopenTest.cfg())) {
                assertTrue(allowed(published.log(), k).contains(Manifest.readLatest(fs.root().resolve(INDEX)).generation()), "after the kill at change " + k);
            } catch (NoSuchFileException e) {
                assertTrue(allowed(published.log(), k).contains(0L), "after the kill at change " + k + ": " + e);
            }
            if (fs.log.size() == k) continue; // the open changed nothing: the crashes are those of the publish itself
            opensThatChangedSomething++;
            crashEverywhere(published, fs.log, fs.log, k, fs.log.size(), tally);
        }
        assertTrue(tally.broken.isEmpty(), report(tally));
        assertTrue(opensThatChangedSomething > 10, "opens that had something to clean up: " + opensThatChangedSomething);
    }

    @Test
    void aCrashDuringPublishesIntoADirectoryThatCheckpointCreatesKeepsTheGeneration() throws IOException {
        // #126: the host does not make the directory, checkpoint does, here with a parent that is missing too
        CrashFs fs = new CrashFs(Files.createTempDirectory(scratch, "created"));
        Recording rec = publishThreeGenerations(fs, fs.root().resolve("indexes").resolve(INDEX));
        Tally tally = new Tally();
        crashEverywhere(rec, rec.log(), rec.log(), 0, rec.log().size(), tally);
        assertTrue(tally.broken.isEmpty(), report(tally));
        assertEquals(Set.of(0L, 1L, 2L, 3L), tally.served);
    }

    @Test
    void aRetryAfterAKillWhileTheDirectoryWasMadeKeepsTheGeneration() throws IOException {
        // #126: the first process is killed between making a directory and flushing its name. The next one finds the directory there, and what
        // it publishes must not depend on a name that nobody flushed
        CrashFs first = new CrashFs(Files.createTempDirectory(scratch, "killed"));
        Recording killed = publishThreeGenerations(first, first.root().resolve("indexes").resolve(INDEX));
        int firstFile = 0;
        while (killed.log().get(firstFile).kind() != CrashFs.Kind.CREATE) firstFile++;
        Tally tally = new Tally();
        for (int k = 0; k <= firstFile; k++) {
            CrashFs fs = CrashFs.resume(killed.log().subList(0, k), Files.createTempDirectory(scratch, "retry"));
            Path index = fs.root().resolve("indexes").resolve(INDEX);
            Generation one;
            try (TieredIndex t = new TieredIndex(DIM, Metric.L2, TierReopenTest.cfg())) {
                for (int i = 0; i < ROWS; i++) t.add(idOf(i), data[i]);
                t.sealNow();
                assertTrue(t.awaitIdle(60_000));
                one = generation(t);
                t.checkpoint(index);
            }
            Recording rec = new Recording("indexes/" + INDEX, List.copyOf(fs.log), Map.of(1L, one));
            tally.seen.clear(); // the expected answers are those of this retry's own tier
            crashEverywhere(rec, rec.log(), rec.log(), k, rec.log().size(), tally);
        }
        assertTrue(firstFile >= 2, "changes before the first file: " + firstFile);
        assertTrue(tally.broken.isEmpty(), report(tally));
    }

    /**
     * Three checkpoints of a tier with rows in its tail (#158): a segment and a tail, then only the tail and the tombstones change (deletes and
     * a replace of sealed ids, more tail rows), then the tail is sealed and a new one begins.
     */
    static Recording publishThreeGenerationsWithTails(CrashFs fs, Path index) throws IOException {
        Map<Long, Generation> generations = new HashMap<>();
        try (TieredIndex t = new TieredIndex(DIM, Metric.L2, TierReopenTest.cfg())) {
            for (int i = 0; i < ROWS; i++) t.add(idOf(i), data[i]);
            t.sealNow();
            assertTrue(t.awaitIdle(60_000));
            for (int i = ROWS; i < ROWS + 20; i++) t.add(idOf(i), data[i]);
            generations.put(1L, generation(t));
            t.checkpoint(index);
            for (int i = 0; i < ROWS; i += 11) t.delete(idOf(i));
            t.add(idOf(1), data[ROWS + 60]);
            for (int i = ROWS + 20; i < ROWS + 30; i++) t.add(idOf(i), data[i]);
            generations.put(2L, generation(t));
            t.checkpoint(index);
            t.sealNow();
            assertTrue(t.awaitIdle(60_000));
            for (int i = ROWS + 30; i < ROWS + 40; i++) t.add(idOf(i), data[i]);
            generations.put(3L, generation(t));
            t.checkpoint(index);
        }
        assertTrue(Manifest.readLatest(index).tailRows() > 0);
        return new Recording(fs.root().relativize(index).toString(), List.copyOf(fs.log), generations);
    }

    @Test
    void aCrashDuringAPublishWithATailLeavesTheGenerationBeforeOrTheOneBeingPublished() throws IOException {
        CrashFs fs = new CrashFs(Files.createTempDirectory(scratch, "tails"));
        Path index = fs.root().resolve(INDEX);
        Files.createDirectory(index);
        Recording rec = publishThreeGenerationsWithTails(fs, index);
        Tally tally = new Tally();
        crashEverywhere(rec, rec.log(), rec.log(), 0, rec.log().size(), tally);
        assertTrue(tally.broken.isEmpty(), report(tally));
        assertEquals(Set.of(0L, 1L, 2L, 3L), tally.served, "every state of the run was reached");

        // the flush of every tail file is needed, as the flushes of the other files are; only a directory flush before a later rename is not
        List<String> needed = new ArrayList<>(), notNeeded = new ArrayList<>();
        removeEachFlush(rec, needed, notNeeded);
        for (long g = 1; g <= 3; g++) assertTrue(needed.contains("flush of " + Manifest.tailName(g)), "needed: " + needed);
        assertTrue(notNeeded.stream().allMatch(n -> n.startsWith("directory flush before the rename of generation ")), "not needed: " + notNeeded);
    }

    @Test
    void takingAFlushOutOfThePublishOrderLeavesABrokenDirectory() throws IOException {
        List<String> needed = new ArrayList<>(), notNeeded = new ArrayList<>();
        removeEachFlush(published, needed, notNeeded);
        // Every flush of a file, of the new directory's name (checkpoint's), and of the directory after each rename (step 7) is needed, and so is
        // the directory flush before the first rename (step 3): without it a first manifest can name files that are not there. In a later publish
        // step 3 alone can go: an N+1 that lost a name is skipped, and generation N is whole until the flush of step 7 has returned.
        assertEquals(List.of("directory flush before the rename of generation 2", "directory flush before the rename of generation 3"), notNeeded);
        assertEquals(14, needed.size(), "needed: " + needed);
    }

    /** takes each flush of a recording out in turn and sorts it into the ones whose absence leaves a broken directory and the others */
    private static void removeEachFlush(Recording rec, List<String> needed, List<String> notNeeded) throws IOException {
        List<CrashFs.Op> log = rec.log();
        for (int i = 0; i < log.size(); i++) {
            CrashFs.Op op = log.get(i);
            if (op.kind() != CrashFs.Kind.SYNC_FILE && op.kind() != CrashFs.Kind.SYNC_DIR) continue;
            // a flush of a directory without an unflushed name change has nothing to do, and taking it out changes nothing
            if (op.kind() == CrashFs.Kind.SYNC_DIR && CrashFs.replay(log, i).pendingOps().stream().noneMatch(p -> p.dir() == op.node())) continue;
            List<CrashFs.Op> without = new ArrayList<>(log);
            without.set(i, CrashFs.NOTHING);
            // the next flush of the same directory makes up for a directory flush, so there is nothing new to see after it
            int last = log.size();
            for (int j = i + 1; op.kind() == CrashFs.Kind.SYNC_DIR && j < last; j++) {
                if (log.get(j).kind() == CrashFs.Kind.SYNC_DIR && log.get(j).node() == op.node()) last = j;
            }
            Tally tally = new Tally();
            for (int k = i + 1; k <= last && tally.broken.isEmpty(); k++) crashEverywhere(rec, log, without, k, k, tally);
            (tally.broken.isEmpty() ? notNeeded : needed).add(flushName(log, i));
        }
    }

    /** the flush at {@code i} of the recorded log, in the words of the publish order */
    private static String flushName(List<CrashFs.Op> log, int i) {
        CrashFs.Op op = log.get(i);
        if (op.kind() == CrashFs.Kind.SYNC_FILE) {
            return "flush of " + log.subList(0, i).stream().filter(p -> p.kind() == CrashFs.Kind.CREATE && p.node() == op.node()).findFirst().orElseThrow().name();
        }
        if (op.node() == CrashFs.ROOT) return "flush of the new directory's name";
        long renames = log.subList(0, i).stream().filter(p -> p.kind() == CrashFs.Kind.RENAME).count();
        boolean before = i + 1 < log.size() && log.get(i + 1).kind() == CrashFs.Kind.CREATE && log.get(i + 1).name().endsWith(".tvm.tmp");
        return before ? "directory flush before the rename of generation " + (renames + 1) : "directory flush after the rename of generation " + renames;
    }
}
