package ai.humem.tiervec;

import java.util.concurrent.Executor;

/**
 * What a host that runs many indexes in one JVM gives each of them: the threads, the permission to build, the memory budget, and a place
 * for the statistics. A tier that is made with a host environment creates no thread, pool, or timer of its own: seals, merges, and the
 * parallel part of a build run as tasks of {@link #executor()}, and a seal, a merge, or a rebuild starts only with a permit of
 * {@link #acquireBuild}. The tier without one ({@link #standalone()}) keeps its own worker thread and build threads and is never refused.
 *
 * <p>Who triggers the work. After {@code add}, {@code delete}, and {@code apply} the tier looks whether a seal or a merge is due and submits
 * a task. What becomes due without a write (a seal by age, a retry after a refusal) needs a call: the tier asks {@link #schedule}, and a host
 * that has no timer there calls {@link TieredIndex#maintain()} from its own.
 *
 * <p>Every method is called by the library on the thread that needs the answer, sometimes while it holds a lock of the index: a method must
 * not block and must not call into the index. An executor may run a task on the calling thread. The parallel part of a build never waits for
 * a task that has not started, so an executor with one thread is enough. A writer that waits for room in the tail ({@code add} at the hard
 * tail limit, {@code awaitCapacity}) needs the executor to run the seal, so a host does not call these from the only thread of the executor.
 */
public interface HostEnv {

    /** what a permit is asked for */
    enum BuildKind { SEAL, MERGE, REBUILD }

    /** the permission to run one build; the library closes it when the build has ended, whether it worked or not */
    interface Permit extends AutoCloseable {
        @Override
        void close();
    }

    /** @return the executor that runs the tasks of the tier */
    Executor executor();

    /**
     * Asks to run one build. The estimate is arithmetic over the number of rows, the dimension, and the graph parameters; it has not been
     * compared with measurements.
     *
     * @param kind what is to be built
     * @param estimatedPeakBytes the memory that the build is expected to need at its peak, on top of what the index holds
     * @return a permit, or null to refuse: a background build is then asked for again later, and a call that builds on its own thread
     *         ({@code sealNow}, {@code mergeNow}, {@code rebuildFrom}) throws {@link BuildRefusedException}
     */
    Permit acquireBuild(BuildKind kind, long estimatedPeakBytes);

    /**
     * The resident bytes that this index should stay below, read at each decision. A merge is deferred while {@code bytesHeld()} plus its
     * estimate is above it, and a seal starts at a smaller tail so that the tail takes at most about an eighth of it. It cannot shrink what
     * the vectors themselves need.
     *
     * @return the budget in bytes; {@link Long#MAX_VALUE} (the default) means none
     */
    default long memoryBudgetBytes() {
        return Long.MAX_VALUE;
    }

    /** @return a sink that the tier binds when it is made ({@link TieredIndex#bindMetrics}), or null (the default) for none */
    default MetricSink metrics() {
        return null;
    }

    /**
     * Runs a task of the tier after a delay, for a retry or a seal by age. The task is short: it submits work to the executor.
     *
     * @param task the task
     * @param delayNanos how long to wait, in nanoseconds
     * @return true when the task was scheduled; false (the default) when the host has no timer for it
     */
    default boolean schedule(Runnable task, long delayNanos) {
        return false;
    }

    /** @return the environment of a tier that runs on its own: its own threads, no refusal, no budget */
    static HostEnv standalone() {
        return StandaloneEnv.INSTANCE;
    }
}
