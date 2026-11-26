package ai.humem.tiervec;

import java.util.concurrent.Executor;

/**
 * {@link HostEnv#standalone()}. A tier that has it runs its own worker thread and build threads, as a tier made without an environment
 * does; the executor is here for a caller that uses this environment as the base of its own.
 */
final class StandaloneEnv implements HostEnv {
    static final StandaloneEnv INSTANCE = new StandaloneEnv();
    private static final Permit GRANTED = () -> {};

    private StandaloneEnv() {}

    @Override
    public Executor executor() {
        return task -> {
            Thread t = new Thread(task, "tiered-task");
            t.setDaemon(true);
            t.start();
        };
    }

    @Override
    public Permit acquireBuild(BuildKind kind, long estimatedPeakBytes) {
        return GRANTED;
    }
}
