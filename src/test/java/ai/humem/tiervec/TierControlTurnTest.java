package ai.humem.tiervec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/**
 * A caller of {@code sealNow} that waits while the worker builds gets the next turn, also when the worker has more to do (issue #144). The
 * worker takes the lock of the builds again right after it releases it, so a lock that lets a newcomer pass a waiting thread can keep the
 * caller out for as long as writes keep coming. The test holds the worker inside a seal until the caller waits, and then looks at who seals
 * next.
 */
@Timeout(value = 3, unit = TimeUnit.MINUTES, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
class TierControlTurnTest {
    private static final int DIM = 8;
    private static final int TAIL = 64;
    private static final int ROUNDS = 20;
    private static final String WORKER = "tiered-worker";
    private static final String CALLER = "turn-caller";

    private long nextId;

    private void fillATail(TieredIndex t) {
        for (int i = 0; i < TAIL; i++, nextId++) t.add(nextId, TierT.vec(144, nextId, DIM));
    }

    @Test
    void aWaitingCallerSealsNextWhileTheWorkerHasMoreToDo() throws Exception {
        TieredIndex.Config c = new TieredIndex.Config(TAIL, 0L, 10, 64, 4, 0.3, 0, 4, Quant.NONE, GraphIndex.Params.defaults().withEfConstruction(32), 1);
        TieredIndex t = new TieredIndex(DIM, Metric.L2, c);
        LinkedBlockingQueue<String> sealers = new LinkedBlockingQueue<>(); // the thread of every seal, in order
        Semaphore workerInside = new Semaphore(0);
        Semaphore workerMayLeave = new Semaphore(0);
        AtomicReference<Throwable> failed = new AtomicReference<>();
        t.setSealHooks(() -> { // runs inside the lock of the builds
            String who = Thread.currentThread().getName();
            sealers.add(who);
            if (who.equals(WORKER)) {
                workerInside.release();
                workerMayLeave.acquireUninterruptibly();
            }
        }, null);
        Thread caller = null;
        try {
            for (int round = 1; round <= ROUNDS; round++) {
                fillATail(t);
                assertTrue(workerInside.tryAcquire(30, TimeUnit.SECONDS), "round " + round + ": the worker did not seal");
                fillATail(t); // the next seal is due when this one is done
                caller = new Thread(() -> {
                    try {
                        t.sealNow();
                    } catch (Throwable e) {
                        failed.set(e);
                    }
                }, CALLER);
                caller.start();
                long end = System.nanoTime() + 30_000_000_000L;
                while (caller.getState() != Thread.State.WAITING && System.nanoTime() < end) Thread.sleep(1);
                assertEquals(Thread.State.WAITING, caller.getState(), "round " + round + ": the caller does not wait for the lock");
                sealers.clear();
                workerMayLeave.release();
                assertEquals(CALLER, sealers.poll(30, TimeUnit.SECONDS), "round " + round + ": who seals after the worker");
                caller.join(30_000);
                assertEquals(Thread.State.TERMINATED, caller.getState(), "round " + round + ": sealNow did not return");
            }
            assertNull(failed.get());
        } finally {
            workerMayLeave.release(1_000_000);
            if (caller != null) caller.join(30_000);
            t.close();
        }
    }
}
