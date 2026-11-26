package ai.humem.tiervec;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.LockSupport;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/**
 * A caller of {@code sealNow} gets its turn while the worker always has a next task (issue #144). The worker takes the lock of the seals and
 * merges again right after it releases it, so a lock that lets a newcomer pass a waiting thread can keep the caller out for as long as
 * writes keep coming.
 */
@Timeout(value = 3, unit = TimeUnit.MINUTES, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
class TierControlTurnTest {
    private static final int DIM = 8;
    private static final int TAIL = 64;
    private static final int CALLS = 5;
    /** a bound on the memory of the test; at 16 rows per millisecond the writer needs more than half a minute to reach it */
    private static final long MAX_ROWS = 600_000;

    @Test
    void sealNowGetsItsTurnWhileWritesKeepTheWorkerBusy() throws Exception {
        TieredIndex.Config c = new TieredIndex.Config(TAIL, 0L, 10, 8, 4, 0.3, 0, 4, Quant.NONE, GraphIndex.Params.defaults().withEfConstruction(32), 1);
        TieredIndex t = new TieredIndex(DIM, Metric.L2, c);
        AtomicBoolean stop = new AtomicBoolean();
        AtomicReference<Throwable> failed = new AtomicReference<>();
        // every seal takes at least 20 ms, in which the writer fills the next tail: the worker never runs out of work
        t.setSealHooks(() -> LockSupport.parkNanos(20_000_000L), null);
        Thread writer = new Thread(() -> {
            try {
                for (long id = 0; id < MAX_ROWS && !stop.get(); id++) {
                    t.add(id, TierT.vec(144, id, DIM));
                    if ((id & 15) == 0) Thread.sleep(1);
                }
            } catch (Throwable e) {
                failed.set(e);
            }
        }, "turn-writer");
        Thread caller = new Thread(() -> {
            try {
                for (int i = 0; i < CALLS; i++) t.sealNow();
            } catch (Throwable e) {
                failed.set(e);
            }
        }, "turn-caller");
        try {
            writer.start();
            long ready = System.nanoTime() + 30_000_000_000L;
            while (t.stats().seals() < 3 && System.nanoTime() < ready) Thread.sleep(5);
            assertTrue(t.stats().seals() >= 3, "the worker did not start to seal");
            caller.start();
            caller.join(20_000);
            boolean stuck = caller.isAlive();
            stop.set(true);
            writer.join(30_000);
            caller.join(60_000); // with no more writes the worker runs out of work and the caller gets in
            assertFalse(stuck, CALLS + " calls of sealNow did not return in 20 s while writes went on");
            assertNull(failed.get());
        } finally {
            stop.set(true);
            t.close();
        }
    }
}
