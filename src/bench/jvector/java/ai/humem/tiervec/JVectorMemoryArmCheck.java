package ai.humem.tiervec;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.atomic.AtomicReference;

/**
 * The checks of the JVector arm of the agent-memory benchmark that a replay on one thread cannot make (issue #160): a write waits while a
 * removal holds its lock and the wait is counted, and writes and searches of several threads beside removals leave exactly the live rows. The CI
 * job "bench smoke" compiles it with the arm and runs it. Exit code 1 when a check failed.
 */
public final class JVectorMemoryArmCheck {
    private static final int DIM = 16, THREADS = 4, IDS = 200, WRITES = 1500;

    private static int checks, failures;

    private static synchronized void check(boolean ok, String what) {
        checks++;
        if (!ok && failures++ < 20) System.out.println("SELFCHECK-FAILURE " + what);
    }

    private static float[] unit(SplittableRandom r) {
        float[] v = new float[DIM];
        double n = 0;
        for (int i = 0; i < DIM; i++) {
            v[i] = (float) (r.nextDouble() - 0.5);
            n += v[i] * v[i];
        }
        for (int i = 0; i < DIM; i++) v[i] /= (float) Math.sqrt(n);
        return v;
    }

    public static void main(String[] args) throws Exception {
        waitIsCounted();
        writersAndSearchersBesideRemovals();
        System.out.println("SELFCHECK-SUMMARY jvector checks=" + checks + " failures=" + failures);
        System.exit(failures == 0 && checks > 0 ? 0 : 1);
    }

    private static void waitIsCounted() throws InterruptedException {
        SplittableRandom r = new SplittableRandom(1);
        try (JVectorMemoryArm arm = new JVectorMemoryArm(DIM, Map.of())) {
            for (int i = 0; i < 100; i++) arm.put(i, unit(r));
            check(arm.writerWaits() == 0, "no write waits while no removal runs");
            Thread writer = new Thread(() -> arm.put(1000, unit(new SplittableRandom(2))));
            arm.writeLockOfRemoval().writeLock().lock();
            try {
                writer.start();
                long deadline = System.nanoTime() + 30_000_000_000L;
                while (!arm.writeLockOfRemoval().hasQueuedThread(writer) && System.nanoTime() < deadline) Thread.onSpinWait();
                check(arm.writeLockOfRemoval().hasQueuedThread(writer), "a write waits while a removal holds the lock");
            } finally {
                arm.writeLockOfRemoval().writeLock().unlock();
            }
            writer.join(30_000);
            check(!writer.isAlive() && arm.writerWaits() == 1 && arm.writerWaitNanos() > 0, "the wait is counted: " + arm.writerWaits());
        }
    }

    private static void writersAndSearchersBesideRemovals() throws InterruptedException {
        AtomicReference<Throwable> error = new AtomicReference<>();
        List<Map<Long, float[]>> live = new ArrayList<>();
        try (JVectorMemoryArm arm = new JVectorMemoryArm(DIM, Map.of("removeAt", "0.05"))) {
            List<Thread> threads = new ArrayList<>();
            for (int t = 0; t < THREADS; t++) {
                int thread = t;
                Map<Long, float[]> mine = new HashMap<>();
                live.add(mine);
                threads.add(new Thread(() -> {
                    SplittableRandom r = new SplittableRandom(100 + thread);
                    try {
                        for (int i = 0; i < WRITES; i++) {
                            long id = (long) thread * IDS + r.nextInt(IDS); // the ids of one thread, so no two writes of an id run together
                            if (mine.containsKey(id) && r.nextInt(3) == 0) {
                                arm.delete(id);
                                mine.remove(id);
                            } else {
                                float[] v = unit(r);
                                arm.put(id, v);
                                mine.put(id, v);
                            }
                            arm.maintain();
                            if (i % 10 == 0) {
                                float[] q = unit(r);
                                arm.search(q, 10, null);
                                long[] one = {id};
                                Hit[] hits = arm.search(q, 1, Violations.only(one));
                                check(mine.containsKey(id) ? hits.length == 1 && Math.abs(hits[0].score() - LiveRows.cosine(q, mine.get(id))) < 1e-4
                                        : hits.length == 0, "a thread reads its own write of the id " + id);
                            }
                        }
                    } catch (Throwable e) {
                        error.compareAndSet(null, e);
                    }
                }));
            }
            threads.forEach(Thread::start);
            for (Thread t : threads) t.join(120_000);
            check(threads.stream().noneMatch(Thread::isAlive), "every thread ended");
            if (error.get() != null) error.get().printStackTrace();
            check(error.get() == null, "no thread failed: " + error.get());

            Map<Long, float[]> all = new HashMap<>();
            live.forEach(all::putAll);
            SplittableRandom r = new SplittableRandom(7);
            for (Map.Entry<Long, float[]> e : all.entrySet()) {
                float[] q = unit(r);
                Hit[] hits = arm.search(q, 1, Violations.only(new long[] {e.getKey()}));
                check(hits.length == 1 && Math.abs(hits[0].score() - LiveRows.cosine(q, e.getValue())) < 1e-4, "the live id " + e.getKey() + " is held");
            }
            for (int i = 0; i < 200; i++) {
                float[] q = unit(r);
                Hit[] hits = arm.search(q, 10, null);
                check(hits.length == Math.min(10, all.size()), "ten hits: " + hits.length);
                for (Hit h : hits) {
                    check(all.containsKey(h.id()) && Math.abs(h.score() - LiveRows.cosine(q, all.get(h.id()))) < 1e-4, "a live id with its score: " + h);
                }
            }
            check(arm.counters().get("removals") > 0, "removals ran: " + arm.counters());
            System.out.println("jvector concurrent check " + arm.counters() + " live=" + all.size());
        }
    }
}
