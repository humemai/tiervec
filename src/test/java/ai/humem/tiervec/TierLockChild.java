package ai.humem.tiervec;

import java.nio.file.Path;

/**
 * The second process of {@link TierLockTest}. It is not a test (surefire does not pick the name up).
 *
 * <p>Arguments: a mode and an index directory. {@code open} opens the directory, closes the tier and exits 0, or prints the message and exits
 * {@link TierLockTest#LOCKED_CODE} when it is locked; it first checks that the overload with a rebuild source refuses in the same way and does
 * not rebuild. {@code checkpoint} writes a small tier into the directory, with the same exit codes. {@code hold} opens the directory, prints
 * {@link TierLockTest#READY} and keeps the tier until its input ends.
 */
public final class TierLockChild {
    private TierLockChild() {}

    public static void main(String[] args) throws Exception {
        final Path dir = Path.of(args[1]);
        try {
            switch (args[0]) {
                case "open" -> {
                    try {
                        TieredIndex.open(dir, TierReopenTest.DIM, Metric.L2, TierReopenTest.cfg(), () -> {
                            throw new AssertionError("a locked directory must not lead to a rebuild");
                        }).close();
                    } catch (DirectoryLockedException e) {
                        TieredIndex.open(dir, TierReopenTest.cfg()).close(); // throws the same
                    }
                }
                case "checkpoint" -> {
                    try (TieredIndex t = TierLockTest.small()) {
                        t.checkpoint(dir);
                    }
                }
                case "hold" -> {
                    try (TieredIndex t = TieredIndex.open(dir, TierReopenTest.cfg())) {
                        System.out.println(TierLockTest.READY + " " + t.size());
                        System.out.flush();
                        while (System.in.read() >= 0) {
                            // until the parent closes the stream
                        }
                    }
                }
                default -> throw new IllegalArgumentException(args[0]);
            }
        } catch (DirectoryLockedException e) {
            System.out.println(e.getMessage());
            System.exit(TierLockTest.LOCKED_CODE);
        }
    }
}
