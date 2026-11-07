package com.humemai.vec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.URI;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import javax.tools.JavaCompiler;
import javax.tools.ToolProvider;
import org.junit.jupiter.api.Assumptions;

/**
 * What the record-id reuse tests share: the assertions on a probe result, and the means to run the same probe against a modified copy of {@link TieredIndex}.
 *
 * <p>A mutant is made like this. The test reads {@code src/main/java/com/humemai/vec/TieredIndex.java} from the project directory, replaces one anchor text that
 * occurs exactly once by one replacement line, compiles the copy with the JDK compiler API into a temporary directory (it needs a JDK, not a JRE), and loads it
 * through a class loader that looks into the mutant directory first for every class of the package {@code com.humemai.vec}. The probe class is loaded by the same
 * loader, so it links against the mutant. The production class is never edited and gets no test hook.
 */
final class RidReuseRig {
    private RidReuseRig() {}

    static final String TIER_SOURCE = "src/main/java/com/humemai/vec/TieredIndex.java";
    static final String[] EXACT = {"lostDeletes", "staleHits", "wrongScore", "oldRank1", "dups", "unsorted", "phantoms", "sizeMismatch", "badDeleteReturn"};
    static final String[] HARD = {"lostDeletes", "staleHits", "wrongScore", "oldRank1", "dups", "unsorted", "phantoms", "sizeMismatch", "badDeleteReturn", "floorLeft",
        "tierFailures", "coverageFail", "timeouts", "readerErrors", "rateFail"};

    // ------------------------------------------------------------------------------------------------------------------
    // reading and checking a result

    static long n(Map<String, Object> r, String key) {
        Object o = r.get(key);
        assertNotNull(o, "the probe result has no entry " + key);
        return ((Number) o).longValue();
    }

    @SuppressWarnings("unchecked")
    static String describe(Map<String, Object> r) {
        StringBuilder sb = new StringBuilder();
        for (String k : HARD) if (n(r, k) != 0) sb.append(' ').append(k).append('=').append(n(r, k));
        sb.append("\n").append(r.get("line"));
        for (String s : (List<String>) r.get("notes")) sb.append("\n  ").append(s);
        return sb.toString();
    }

    /**
     * A run on the real index: every pass counter is exactly 0, every requested case ran, every operation group of the merge cases was confirmed inside a running
     * merge, and the run checked something (a probe that silently does nothing must not pass).
     */
    static void assertClean(String what, Map<String, Object> r, List<String> cases) {
        for (String cs : cases) n(r, "case_" + cs);
        long bad = 0;
        for (String k : HARD) bad += n(r, k);
        assertTrue(bad == 0 && Boolean.TRUE.equals(r.get("pass")), () -> what + ": the probe reports violations:" + describe(r));
        assertTrue(n(r, "idsChecked") >= 1000, () -> what + ": only " + n(r, "idsChecked") + " ids were checked");
        assertTrue(n(r, "hitsChecked") >= 10_000, () -> what + ": only " + n(r, "hitsChecked") + " hits were checked");
        int groups = RidReuseProbe.expectedWindowGroups(cases);
        assertEquals(groups, n(r, "windowGroups"), () -> what + ": operation groups run inside a merge window" + describe(r));
        assertEquals(groups, n(r, "windowGroupsConfirmed"), () -> what + ": operation groups confirmed inside the running merge" + describe(r));
    }

    /**
     * A mutant run that counts as a catch only when it was a real run: no worker error, every merge window was reached (a void case is not a catch), and with
     * strict set (a run of the one case that carries the signature) no precondition was missed either. A full run must not be strict: a broken floor can
     * legitimately miss the precondition of the floor-chain case, which asks for floor entries.
     */
    static void assertReachedAndSound(String what, Map<String, Object> r, List<String> cases, boolean strict) {
        assertEquals(0, n(r, "tierFailures"), () -> what + ": the run died with an exception instead of reporting a violation" + describe(r));
        if (strict) assertEquals(0, n(r, "coverageFail"), () -> what + ": a window or precondition was not reached (void run, not a catch)" + describe(r));
        for (String cs : cases) n(r, "case_" + cs);
        assertTrue(n(r, "idsChecked") >= 1000, () -> what + ": only " + n(r, "idsChecked") + " ids were checked");
        int groups = RidReuseProbe.expectedWindowGroups(cases);
        assertEquals(groups, n(r, "windowGroups"), () -> what + ": operation groups run inside a merge window" + describe(r));
        assertEquals(groups, n(r, "windowGroupsConfirmed"), () -> what + ": operation groups confirmed inside the running merge" + describe(r));
    }

    // ------------------------------------------------------------------------------------------------------------------
    // mutants

    /** one single-line change of TieredIndex.java; anchor null means no change at all (the control) */
    record Mutation(String name, String anchor, String replacement, String what) {}

    static final Mutation X0 = new Mutation("X0", null, null, "no change: the control, must pass like the real class");
    static final Mutation M1 = new Mutation("M1", "if (p.phys() || p.inFrozen()) floor.put(id, ver);", "if (p.phys() || p.inFrozen()) floor.remove(id);",
            "add removes the floor entry of the id instead of raising it");
    static final Mutation M2 = new Mutation("M2", "if (p.phys() || p.inFrozen()) floor.put(id, ++counter);", "if (p.phys() || p.inFrozen()) ++counter;",
            "delete takes a version but writes no floor entry");
    static final Mutation M3 = new Mutation("M3", "if (!needed) floor.remove(e.getKey());", "floor.remove(e.getKey());",
            "cleanupFloor drops every floor entry after a merge, needed or not");

    /** with no compiler (a JRE) the mutant tests cannot run: skip with a message, except on CI, where a JDK is a given and a skip would hide a broken setup */
    static void requireCompiler() {
        if (ToolProvider.getSystemJavaCompiler() != null) return;
        String msg = "no system Java compiler: the mutant tests compile a copy of TieredIndex.java and need a JDK, not a JRE";
        if (System.getenv("CI") != null) fail(msg + " (CI is set, so this is not skipped)");
        Assumptions.assumeTrue(false, msg);
    }

    static Path projectDir() {
        return Path.of(System.getProperty("basedir", ".")).toAbsolutePath().normalize();
    }

    static URL locationOf(Class<?> c) {
        return c.getProtectionDomain().getCodeSource().getLocation();
    }

    static int countOf(String text, String anchor) {
        int count = 0;
        for (int i = text.indexOf(anchor); i >= 0; i = text.indexOf(anchor, i + 1)) count++;
        return count;
    }

    /** write the mutated copy and compile it into root/NAME/classes; returns that directory */
    static Path compile(Mutation mu, Path root) throws Exception {
        Path src = projectDir().resolve(TIER_SOURCE);
        assertTrue(Files.isRegularFile(src), "the project source " + src + " was not found (basedir " + projectDir() + ")");
        String text = Files.readString(src, StandardCharsets.UTF_8);
        String mutated = text;
        if (mu.anchor() != null) {
            assertEquals(1, countOf(text, mu.anchor()), mu.name() + ": the anchor must occur exactly once in " + TIER_SOURCE + ": " + mu.anchor());
            int at = text.indexOf(mu.anchor());
            mutated = text.substring(0, at) + mu.replacement() + text.substring(at + mu.anchor().length());
        }
        String[] a = text.split("\n", -1);
        String[] b = mutated.split("\n", -1);
        assertEquals(a.length, b.length, mu.name() + ": the mutation must not add or remove a line");
        int changed = 0;
        for (int i = 0; i < a.length; i++) if (!a[i].equals(b[i])) changed++;
        assertEquals(mu.anchor() == null ? 0 : 1, changed, mu.name() + ": the number of changed lines");
        Path dir = root.resolve(mu.name());
        Path classes = dir.resolve("classes");
        Files.createDirectories(dir.resolve("src"));
        Files.createDirectories(classes);
        Path out = dir.resolve("src").resolve("TieredIndex.java");
        Files.writeString(out, mutated, StandardCharsets.UTF_8);
        JavaCompiler jc = ToolProvider.getSystemJavaCompiler();
        assertNotNull(jc, "no system Java compiler");
        ByteArrayOutputStream log = new ByteArrayOutputStream();
        int rc = jc.run(null, log, log, "-d", classes.toString(), "--add-modules", "jdk.incubator.vector", "-cp", Path.of(locationOf(TieredIndex.class).toURI()).toString(),
                "-proc:none", "-nowarn", "-encoding", "UTF-8", out.toString());
        assertEquals(0, rc, mu.name() + ": the mutated copy did not compile:\n" + log.toString(StandardCharsets.UTF_8));
        assertTrue(Files.isRegularFile(classes.resolve("com/humemai/vec/TieredIndex.class")), mu.name() + ": TieredIndex.class was not written");
        return classes;
    }

    /** looks into its own URLs first for every class of com.humemai.vec, and asks the parent for everything else (the JDK, the Vector API module, JUnit) */
    static final class ChildFirst extends URLClassLoader {
        ChildFirst(URL[] urls, ClassLoader parent) {
            super(urls, parent);
        }

        @Override
        protected Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException {
            if (!name.startsWith("com.humemai.vec.")) return super.loadClass(name, resolve);
            synchronized (getClassLoadingLock(name)) {
                Class<?> c = findLoadedClass(name);
                if (c == null) {
                    try {
                        c = findClass(name);
                    } catch (ClassNotFoundException e) {
                        c = super.loadClass(name, false);
                    }
                }
                if (resolve) resolveClass(c);
                return c;
            }
        }
    }

    /** a loader in which TieredIndex comes from the mutant directory, and everything else of the package from the main and test classes */
    static ClassLoader loaderFor(Path mutantClasses) throws Exception {
        List<URL> urls = new ArrayList<>();
        urls.add(mutantClasses.toUri().toURL());
        urls.add(locationOf(TieredIndex.class));
        urls.add(locationOf(RidReuseProbe.class));
        return new ChildFirst(urls.toArray(new URL[0]), RidReuseRig.class.getClassLoader());
    }

    /** run the probe class of the given loader; proves that the probe and the index class came from that loader, and from the mutant directory when given */
    @SuppressWarnings("unchecked")
    static Map<String, Object> runOn(ClassLoader loader, Path expectedTierDir, String cfg, long seed, List<String> cases) throws Exception {
        Class<?> probe = Class.forName("com.humemai.vec.RidReuseProbe", true, loader);
        assertSame(loader, probe.getClassLoader(), "the probe class must be defined by the mutant loader, not by the application loader");
        Method run = probe.getDeclaredMethod("runOneMap", String.class, long.class, List.class);
        run.setAccessible(true);
        Map<String, Object> r;
        try {
            r = (Map<String, Object>) run.invoke(null, cfg, seed, cases);
        } catch (InvocationTargetException e) {
            if (e.getCause() instanceof Exception ex) throw ex;
            if (e.getCause() instanceof Error er) throw er;
            throw e;
        }
        String tier = String.valueOf(r.get("tierClass"));
        if (expectedTierDir != null) {
            Path loaded = Path.of(URI.create(tier)).toRealPath();
            assertTrue(loaded.startsWith(expectedTierDir.toRealPath()), "TieredIndex was loaded from " + tier + ", not from the mutant directory " + expectedTierDir);
        }
        return r;
    }

    /** compile a mutant once and keep its directory and loader for the tests of one class */
    static final class Mutants implements AutoCloseable {
        final Path root;
        final Map<String, Path> dirs = new java.util.HashMap<>();
        final Map<String, ClassLoader> loaders = new java.util.HashMap<>();

        Mutants() throws IOException {
            root = Files.createTempDirectory("ridreuse-mutants-");
        }

        synchronized Path dir(Mutation mu) throws Exception {
            Path d = dirs.get(mu.name());
            if (d == null) {
                d = compile(mu, root);
                dirs.put(mu.name(), d);
            }
            return d;
        }

        synchronized ClassLoader loader(Mutation mu) throws Exception {
            ClassLoader l = loaders.get(mu.name());
            if (l == null) {
                l = loaderFor(dir(mu));
                loaders.put(mu.name(), l);
            }
            return l;
        }

        Map<String, Object> run(Mutation mu, String cfg, long seed, List<String> cases) throws Exception {
            return runOn(loader(mu), dir(mu), cfg, seed, cases);
        }

        @Override
        public void close() throws IOException {
            for (ClassLoader l : loaders.values()) if (l instanceof URLClassLoader u) u.close();
            loaders.clear();
            // only the temporary directory this object created
            try (Stream<Path> s = Files.walk(root)) {
                s.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
            }
        }
    }

    // ------------------------------------------------------------------------------------------------------------------
    // what a caught mutant must show

    static void assertCaughtM1(String what, Map<String, Object> r, List<String> cases, boolean strict) {
        assertReachedAndSound(what, r, cases, strict);
        assertTrue(n(r, "staleHits") > 0, () -> what + ": M1 (" + M1.what() + ") must give staleHits above 0, the probe did not see it" + describe(r));
        assertTrue(n(r, "case_b_sealed") > 0, () -> what + ": M1 must give a violation in case b_sealed" + describe(r));
    }

    static void assertCaughtM2(String what, Map<String, Object> r, List<String> cases, boolean strict) {
        assertReachedAndSound(what, r, cases, strict);
        assertTrue(n(r, "lostDeletes") > 0, () -> what + ": M2 (" + M2.what() + ") must give lostDeletes above 0, the probe did not see it" + describe(r));
        assertTrue(n(r, "case_b_sealed") > 0, () -> what + ": M2 must give a violation in case b_sealed" + describe(r));
    }

    static void assertCaughtM3(String what, Map<String, Object> r, List<String> cases, boolean strict) {
        assertReachedAndSound(what, r, cases, strict);
        assertTrue(n(r, "case_c2_delete_in_merge") > 0, () -> what + ": M3 (" + M3.what() + ") must give a violation in case c2_delete_in_merge" + describe(r));
        assertTrue(n(r, "exact") > 0, () -> what + ": M3 must show a wrong answer, not only a coverage or worker counter" + describe(r));
    }
}
