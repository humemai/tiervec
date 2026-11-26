package ai.humem.tiervec;

import static ai.humem.tiervec.SegmentTestSupport.vectors;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;

/** The host environment of issue #13: no thread of the library's own, permits before every build, refusals, the memory budget, and close. */
@Timeout(value = 5, unit = TimeUnit.MINUTES)
class HostEnvTest {
    private static final int DIM = 16;

    @TempDir
    Path dir;

    private final List<TieredIndex> opened = new ArrayList<>();

    @AfterEach
    void closeAll() {
        for (TieredIndex t : opened) t.close();
    }

    /** a host: the executor of the test, a count of the permits, a switch that refuses them, a budget, and a timer that only records */
    private static final class Host implements HostEnv {
        final Executor executor;
        volatile boolean refuse;
        volatile long budget = Long.MAX_VALUE;
        final List<BuildKind> asked = new ArrayList<>();
        final List<Long> estimates = new ArrayList<>();
        final AtomicInteger held = new AtomicInteger();
        final List<Runnable> timers = new ArrayList<>();
        final List<Long> delays = new ArrayList<>();

        Host(Executor executor) {
            this.executor = executor;
        }

        @Override
        public Executor executor() {
            return executor;
        }

        @Override
        public synchronized Permit acquireBuild(BuildKind kind, long estimatedPeakBytes) {
            asked.add(kind);
            estimates.add(estimatedPeakBytes);
            if (refuse) return null;
            held.incrementAndGet();
            return held::decrementAndGet;
        }

        @Override
        public long memoryBudgetBytes() {
            return budget;
        }

        @Override
        public synchronized boolean schedule(Runnable task, long delayNanos) {
            timers.add(task);
            delays.add(delayNanos);
            return true;
        }
    }

    /** an executor that runs nothing until the test says so */
    private static final class Queue implements Executor {
        private final ArrayDeque<Runnable> tasks = new ArrayDeque<>();

        @Override
        public synchronized void execute(Runnable r) {
            tasks.add(r);
        }

        synchronized int size() {
            return tasks.size();
        }

        void runAll() {
            for (;;) {
                Runnable r;
                synchronized (this) {
                    r = tasks.poll();
                }
                if (r == null) return;
                r.run();
            }
        }
    }

    private static TieredIndex.Config cfg(int tailLimit, int maxSegments, int buildThreads) {
        return new TieredIndex.Config(tailLimit, 0L, 50, maxSegments, 4, 0.3, 0, 4, Quant.NONE, GraphIndex.Params.defaults().withEfConstruction(64),
                buildThreads);
    }

    private TieredIndex tier(TieredIndex.Config c, HostEnv env) {
        TieredIndex t = new TieredIndex(DIM, Metric.L2, c, env);
        opened.add(t);
        return t;
    }

    private static Set<Thread> threads() {
        ThreadGroup mine = Thread.currentThread().getThreadGroup(); // the JVM's own helper threads live in other groups
        Set<Thread> s = new HashSet<>();
        for (Thread t : Thread.getAllStackTraces().keySet()) {
            if (t.getThreadGroup() == mine) s.add(t);
        }
        return s;
    }

    private static void fill(TieredIndex t, float[][] data, int from, int to) {
        for (int i = from; i < to; i++) t.add(i, data[i]);
    }

    private static void assertFindsItself(TieredIndex t, float[][] data, int n) {
        int found = 0;
        for (int i = 0; i < n; i += 10) {
            Hit[] h = t.search(data[i], 1, null);
            if (h.length == 1 && h[0].id() == i) found++;
        }
        assertTrue(found >= (n / 10) * 9 / 10, found + " of " + n / 10);
    }

    @Test
    void aHostedTierCreatesNoThreadAndAsksForAPermitBeforeEveryBuild() throws Exception {
        float[][] data = vectors(3000, DIM, 11);
        Host host = new Host(Runnable::run); // a same-thread executor: any other thread would be one that the library made
        Set<Thread> before = threads();
        TieredIndex t = tier(cfg(200, 2, 4), host);
        Set<Thread> during = new HashSet<>();
        t.setSealHooks(() -> during.addAll(threads()), null);
        fill(t, data, 0, 3000);
        for (int i = 0; i < 300; i++) t.delete(i * 7);
        assertTrue(t.awaitIdle(60_000));
        t.mergeNow();
        t.rebuildFrom(List.of(new TieredIndex.IdVector(1, data[1])));
        assertTrue(t.stats().seals() > 0 && t.stats().merges() > 0, "seals " + t.stats().seals() + ", merges " + t.stats().merges());
        during.addAll(threads());
        during.removeAll(before);
        assertTrue(during.isEmpty(), "new threads: " + during);
        assertFalse(t.workerAlive());
        assertEquals(TieredIndex.Health.OK, t.health());
        assertEquals(t.stats().seals(), host.asked.stream().filter(k -> k == HostEnv.BuildKind.SEAL).count());
        assertEquals(t.stats().merges(), host.asked.stream().filter(k -> k == HostEnv.BuildKind.MERGE).count());
        assertTrue(host.estimates.stream().allMatch(e -> e > 0), "estimates " + host.estimates);
        assertEquals(0, host.held.get(), "every permit was closed");
    }

    @Test
    void aRebuildAsksForAPermitAndARefusalLeavesTheIndexAsItWas() {
        float[][] data = vectors(400, DIM, 12);
        Host host = new Host(Runnable::run);
        TieredIndex t = tier(cfg(1_000_000, 8, 1), host);
        fill(t, data, 0, 100);
        List<TieredIndex.IdVector> records = new ArrayList<>();
        for (int i = 0; i < 400; i++) records.add(new TieredIndex.IdVector(i, data[i]));
        host.refuse = true;
        assertThrows(BuildRefusedException.class, () -> t.rebuildFrom(records));
        assertThrows(BuildRefusedException.class, () -> t.rebuildFrom(() -> new TieredIndex.Snapshot(399, records), 1L));
        assertThrows(BuildRefusedException.class, t::sealNow);
        assertEquals(100, t.size());
        assertEquals(0, t.segmentCount());
        host.refuse = false;
        t.rebuildFrom(records);
        assertEquals(400, t.size());
        assertEquals(List.of(HostEnv.BuildKind.REBUILD, HostEnv.BuildKind.REBUILD, HostEnv.BuildKind.SEAL, HostEnv.BuildKind.REBUILD), host.asked);
        t.add(1000, data[0]);
        t.sealNow();
        host.refuse = true;
        assertThrows(BuildRefusedException.class, t::mergeNow);
        assertEquals(2, t.segmentCount());
    }

    @Test
    void aRefusedBackgroundBuildIsDeferredAndTriedAgainAfterABackOff() throws Exception {
        float[][] data = vectors(600, DIM, 13);
        Host host = new Host(Runnable::run);
        host.refuse = true;
        TieredIndex t = tier(cfg(100, 8, 1), host);
        long t0 = System.nanoTime();
        fill(t, data, 0, 300);
        long elapsed = System.nanoTime() - t0;
        assertEquals(0, t.segmentCount(), "nothing is built without a permit");
        assertEquals(0, t.stats().seals());
        assertTrue(t.metrics().get("tiervec.deferredBuilds") >= 1);
        // one ask per back-off period (50 ms, doubling), not one per write
        assertTrue(host.asked.size() <= 2 + elapsed / 50_000_000L, host.asked.size() + " asks in " + elapsed / 1_000_000 + " ms");
        assertEquals(TieredIndex.Health.OK, t.health(), "a refusal is not a failure");
        assertFindsItself(t, data, 300);
        host.refuse = false;
        assertTrue(t.awaitIdle(60_000), "the next try gets its permit");
        assertTrue(t.segmentCount() >= 1);
        assertFindsItself(t, data, 300);
    }

    @Test
    void aSealByAgeAsksTheTimerOfTheHostOncePerTail() throws Exception {
        float[][] data = vectors(100, DIM, 19);
        Host host = new Host(Runnable::run);
        TieredIndex.Config c = new TieredIndex.Config(1_000_000, 2_000L, 50, 8, 4, 0.3, 0, 4, Quant.NONE, GraphIndex.Params.defaults(), 1);
        TieredIndex t = tier(c, host);
        fill(t, data, 0, 100);
        assertEquals(1, host.timers.size(), "one request per tail, not one per write");
        long delay = host.delays.get(0);
        assertTrue(delay > 0 && delay <= 2_001_000_000L, "delay " + delay);
        assertEquals(0, t.segmentCount());
        Thread.sleep(delay / 1_000_000L + 50);
        host.timers.get(0).run();
        assertEquals(1, t.segmentCount(), "the timer task seals the old tail");
    }

    @Test
    void aSmallMemoryBudgetShrinksTheTailDefersMergesAndKeepsTheBytesBelowIt() throws Exception {
        final int n = 3000;
        final long budget = 2L << 20; // the 3000 vectors fit in it; a merge of all of them, by the estimate, does not
        float[][] data = vectors(n, DIM, 14);
        Host host = new Host(Runnable::run);
        TieredIndex t = tier(cfg(2000, 1, 1), host);
        assertEquals(2000, t.effectiveTailLimit());
        host.budget = budget;
        int limit = t.effectiveTailLimit();
        assertTrue(limit < 2000 && limit >= 64, "tail limit " + limit);
        assertTrue(t.estimatedBuildBytes(n) > budget, "estimate " + t.estimatedBuildBytes(n));
        for (int i = 0; i < n; i++) {
            t.add(i, data[i]);
            assertTrue(t.bytesHeld() < budget, "after " + (i + 1) + " adds " + t.bytesHeld() + " bytes are held, the budget is " + budget);
        }
        t.sealNow();
        t.maintain();
        assertEquals(n, t.size());
        assertTrue(t.stats().seals() >= n / limit, "the tail seals at the smaller limit: " + t.stats().seals() + " seals at " + limit);
        assertTrue(t.segmentCount() >= 2, "a merge that does not fit waits: " + t.segmentCount() + " segment(s)");
        assertTrue(t.metrics().get("tiervec.deferredBuilds") >= 1);
        assertTrue(t.bytesHeld() < budget);
        assertFindsItself(t, data, n);
        host.budget = Long.MAX_VALUE;
        assertTrue(t.awaitIdle(60_000), "the merge runs when the budget allows it");
        assertEquals(1, t.segmentCount());
    }

    @Test
    void aMergeThatDoesNotFitTheBudgetWaitsAndAForcedOneRuns() throws Exception {
        float[][] data = vectors(900, DIM, 15);
        Queue queue = new Queue();
        Host host = new Host(queue);
        TieredIndex t = tier(cfg(1_000_000, 1, 1), host);
        for (int s = 0; s < 3; s++) {
            fill(t, data, s * 300, (s + 1) * 300);
            t.sealNow();
        }
        host.budget = t.bytesHeld() + t.estimatedBuildBytes(900) / 2;
        t.maintain();
        queue.runAll();
        assertEquals(3, t.segmentCount());
        assertEquals(0, t.stats().merges());
        assertEquals(1L, t.metrics().get("tiervec.deferredBuilds"));
        assertTrue(t.bytesHeld() < host.budget);
        assertFalse(host.asked.contains(HostEnv.BuildKind.MERGE), "a merge over the budget does not ask for a permit");
        host.budget = Long.MAX_VALUE;
        assertTrue(awaitIdleRunning(t, queue), "the merge runs after the back-off");
        assertEquals(1, t.segmentCount());
        fill(t, data, 0, 10);
        t.sealNow();
        host.budget = 1;
        t.mergeNow(); // a call of the host itself: the permit decides, not the budget
        assertEquals(1, t.segmentCount());
    }

    private static boolean awaitIdleRunning(TieredIndex t, Queue queue) throws InterruptedException {
        long end = System.nanoTime() + 60_000_000_000L;
        while (System.nanoTime() < end) {
            t.maintain();
            queue.runAll();
            if (t.segmentCount() == 1) return true;
            Thread.sleep(5);
        }
        return false;
    }

    @Test
    void anExecutorWithOneThreadFinishesAParallelBuild() throws Exception {
        float[][] data = vectors(4000, DIM, 16);
        ExecutorService one = Executors.newSingleThreadExecutor();
        try {
            Host host = new Host(one);
            TieredIndex t = tier(cfg(2000, 1, 4), host);
            fill(t, data, 0, 4000);
            assertTrue(t.awaitIdle(120_000));
            assertTrue(t.stats().seals() >= 1);
            assertFindsItself(t, data, 4000);
            t.close();
            one.shutdown();
            assertTrue(one.awaitTermination(60, TimeUnit.SECONDS), "no task of a closed tier stays");
        } finally {
            one.shutdownNow();
        }
    }

    @Test
    void closeCancelsAQueuedTaskAndARejectingExecutorShowsAsFailed() {
        float[][] data = vectors(300, DIM, 17);
        Queue queue = new Queue();
        Host host = new Host(queue);
        TieredIndex t = tier(cfg(100, 8, 1), host);
        fill(t, data, 0, 200);
        assertEquals(1, queue.size(), "one task is with the executor, however many writes asked");
        t.close();
        t.close();
        queue.runAll();
        assertEquals(0, t.stats().seals(), "a task that starts after close does nothing");
        assertTrue(host.asked.isEmpty());
        TieredIndex r = tier(cfg(100, 8, 1), new Host(task -> {
            throw new RejectedExecutionException("shut down");
        }));
        fill(r, data, 0, 150);
        assertEquals(TieredIndex.Health.FAILED, r.health());
        assertNotNull(r.lastError());
        assertEquals(150, r.size());
    }

    @Test
    void aReopenedTierRunsOnTheHostToo() throws Exception {
        float[][] data = vectors(200, DIM, 18);
        Host host = new Host(Runnable::run);
        TieredIndex t = tier(cfg(1_000_000, 8, 1), host);
        fill(t, data, 0, 200);
        t.sealNow();
        t.checkpoint(dir);
        t.close();
        Set<Thread> before = threads();
        TieredIndex u = TieredIndex.open(dir, cfg(1_000_000, 8, 1), host);
        opened.add(u);
        Set<Thread> after = threads();
        after.removeAll(before);
        assertTrue(after.isEmpty(), "new threads: " + after);
        assertEquals(200, u.size());
        assertFalse(u.workerAlive());
    }
}
