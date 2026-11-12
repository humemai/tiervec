package com.humemai.vec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.lang.reflect.Modifier;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;

/**
 * The public surface of the package against docs/API.md (issue #64). Every public top-level class of the package is either in the
 * "Supported classes" table or marked {@link Internal} and in the "Internal classes" table, so a new public class cannot slip in
 * without a decision. Nested classes that the tables name must exist. The source files are read from the project directory, like the
 * mutant tests do.
 */
class ApiSurfaceTest {
    private static final Path SOURCES = Path.of("src/main/java/com/humemai/vec");
    private static final Path API = Path.of("docs/API.md");
    private static final Pattern PUBLIC_TOP = Pattern.compile(
            "(?m)^public\\s+(?:final\\s+|abstract\\s+|sealed\\s+)*(?:class|interface|enum|record|@interface)\\s+(\\w+)");
    private static final Pattern FIRST_CELL = Pattern.compile("(?m)^\\|\\s*(`[^|]*`)\\s*\\|");
    private static final Pattern NAME = Pattern.compile("`([A-Za-z0-9_.]+)`");

    private static Class<?> load(String name) throws ClassNotFoundException {
        return Class.forName("com.humemai.vec." + name, false, ApiSurfaceTest.class.getClassLoader());
    }

    private static Set<String> publicTopLevel() throws IOException {
        Set<String> names = new TreeSet<>();
        List<Path> files;
        try (Stream<Path> s = Files.list(SOURCES)) {
            files = s.filter(p -> p.toString().endsWith(".java")).toList();
        }
        for (Path f : files) {
            Matcher m = PUBLIC_TOP.matcher(Files.readString(f));
            if (m.find()) names.add(m.group(1));
        }
        return names;
    }

    /** every name in the first column of the table under the heading */
    private static Set<String> tableNames(String heading) throws IOException {
        String text = Files.readString(API).replace("\r\n", "\n"); // a Windows checkout may have CRLF
        int start = text.indexOf("\n## " + heading + "\n");
        assertTrue(start >= 0, "docs/API.md has no section " + heading);
        int end = text.indexOf("\n## ", start + 1);
        String section = end < 0 ? text.substring(start) : text.substring(start, end);
        Set<String> names = new TreeSet<>();
        Matcher row = FIRST_CELL.matcher(section);
        while (row.find()) {
            Matcher n = NAME.matcher(row.group(1));
            while (n.find()) names.add(n.group(1));
        }
        return names;
    }

    private static Set<String> topLevel(Set<String> names) {
        Set<String> tops = new TreeSet<>();
        for (String n : names) if (n.indexOf('.') < 0) tops.add(n);
        return tops;
    }

    @Test
    void everyPublicClassIsSupportedOrInternal() throws Exception {
        Set<String> publicClasses = publicTopLevel();
        assertTrue(publicClasses.size() >= 15, "the scan found only " + publicClasses.size() + " public classes: " + publicClasses);

        Set<String> supportedNames = tableNames("Supported classes");
        Set<String> internalNames = tableNames("Internal classes");
        Set<String> supported = topLevel(supportedNames);
        Set<String> internal = topLevel(internalNames);
        assertTrue(supported.size() >= 8, "the supported table was not read: " + supportedNames);
        assertTrue(internal.size() >= 5, "the internal table was not read: " + internalNames);

        Set<String> marked = new TreeSet<>();
        for (String name : publicClasses) {
            Class<?> c = load(name);
            assertTrue(Modifier.isPublic(c.getModifiers()), name + " is not public");
            if (c.isAnnotationPresent(Internal.class)) marked.add(name);
        }
        marked.remove("Internal");

        // the annotation and the Internal table say the same
        assertEquals(internal, marked, "the classes marked @Internal and the Internal classes table differ");

        // the two tables do not overlap, and together they hold every public class except the annotation itself
        Set<String> overlap = new TreeSet<>(supported);
        overlap.retainAll(internal);
        assertTrue(overlap.isEmpty(), "in both tables: " + overlap);
        Set<String> accounted = new TreeSet<>(supported);
        accounted.addAll(internal);
        Set<String> expected = new TreeSet<>(publicClasses);
        expected.remove("Internal");
        assertEquals(expected, accounted, "the public classes and the two tables of docs/API.md differ");
    }

    @Test
    void everyNestedClassTheTablesNameExists() throws Exception {
        Set<String> nested = new TreeSet<>();
        for (String n : tableNames("Supported classes")) if (n.indexOf('.') >= 0) nested.add(n);
        assertTrue(nested.size() >= 8, "the nested names were not read: " + nested);
        for (String n : nested) {
            Class<?> c = load(n.replace('.', '$'));
            assertTrue(Modifier.isPublic(c.getModifiers()), n + " is not public");
        }
    }
}
