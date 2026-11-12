package ai.humem.tiervec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Shared by the JUnit wrappers around the test programs. The programs keep their own check counters; a wrapper reads the counters before and after. */
final class Checks {
    private Checks() {}

    /** Run standalone, a program ends the JVM when a check failed. Inside surefire the wrapper reads the counters instead. */
    static void embedded() {
        T.exitOnFailure = false;
        GraphT.exitOnFailure = false;
        TierT.exitOnFailure = false;
        IvfT.exitOnFailure = false;
    }

    /** Zero failed checks, and at least minChecks checks ran (a suite that silently skips its body must not pass). */
    static void clean(String suite, int checks, int failed, int minChecks) {
        assertEquals(0, failed, suite + ": " + failed + " failed checks, the FAIL lines are in the test output");
        assertTrue(checks >= minChecks, suite + " ran " + checks + " checks, expected at least " + minChecks);
    }
}
