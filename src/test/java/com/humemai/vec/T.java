package com.humemai.vec;

/** Tiny assert helper: counts passes and failures, prints one line per failure, exit code 1 when anything failed. */
final class T {
    private T() {}

    static int pass, fail;

    /** false while a JUnit wrapper runs the suite inside the test JVM: done() must then not end the JVM, the wrapper asserts the counters */
    static boolean exitOnFailure = true;

    static void check(boolean ok, String name) {
        if (ok) pass++;
        else {
            fail++;
            System.out.println("FAIL " + name);
        }
    }

    static void section(String name, int passBefore, int failBefore) {
        System.out.println((fail == failBefore ? "PASS " : "FAIL ") + name + " (" + (pass - passBefore) + " checks, " + (fail - failBefore) + " failed)");
    }

    static void done(String suite) {
        System.out.println(suite + ": " + pass + " passed, " + fail + " failed");
        if (fail > 0 && exitOnFailure) System.exit(1);
    }
}
