package com.humemai.vec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * A JVM that is started without the jdk.incubator.vector module can build and search a tier. The test starts such a JVM on the
 * compiled classes (the surefire JVM has the module, so the check needs a child) and runs {@link NoIncubatorProbe} in it. The probe fails
 * on its own if the module is present, so this cannot pass by running with the module.
 */
class NoIncubatorTest {
    @TempDir
    Path tmp;

    static Path locationOf(Class<?> c) throws Exception {
        return Path.of(c.getProtectionDomain().getCodeSource().getLocation().toURI());
    }

    @Test
    void aJvmWithoutTheIncubatorModuleBuildsAndSearchesATier() throws Exception {
        String java = Path.of(System.getProperty("java.home"), "bin", "java").toString();
        String cp = locationOf(TieredIndex.class) + File.pathSeparator + locationOf(NoIncubatorProbe.class);
        Path log = tmp.resolve("probe.log");
        ProcessBuilder pb = new ProcessBuilder(java, "-Xmx512m", "-cp", cp, NoIncubatorProbe.class.getName());
        // a launcher variable that adds the module would defeat the check
        pb.environment().remove("JAVA_TOOL_OPTIONS");
        pb.environment().remove("_JAVA_OPTIONS");
        pb.environment().remove("JDK_JAVA_OPTIONS");
        pb.redirectErrorStream(true);
        pb.redirectOutput(log.toFile());
        Process p = pb.start();
        boolean ended = p.waitFor(300, TimeUnit.SECONDS);
        if (!ended) {
            p.destroyForcibly();
            p.waitFor();
        }
        String out = Files.readString(log);
        assertTrue(ended, "the probe JVM did not end within 300 s:\n" + out);
        assertEquals(0, p.exitValue(), "the probe JVM failed:\n" + out);
        assertTrue(out.contains("NO-INCUBATOR OK"), "the probe JVM did not report success:\n" + out);
        assertTrue(out.contains("scalarReason=Vector API unavailable"), "the kernels did not fall back because of the missing module:\n" + out);
        System.out.println(out);
    }
}
