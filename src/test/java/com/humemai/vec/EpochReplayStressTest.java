package com.humemai.vec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Random;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.LockSupport;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Randomized runs with a fake host (issue #7). The crash loop: the host commits transactions and applies them, the tier is sealed, merged and
 * checkpointed (sometimes with a crash in the middle of the publish), the host forgets its unflushed transactions and mints their ids again, and
 * the tier is reopened with the host's record source; after every reopen the tier must hold exactly the host's live records. The rebuild race:
 * a host thread applies numbered batches without a pause while the main thread rebuilds the tier from snapshots, and no write may be lost.
 * The long variants are in EpochReplayLongTest.
 */
class EpochReplayStressTest {
    static final int DIM = 16;
    static final Metric METRIC = Metric.L2;

    @TempDir
    Path dir;

    @ParameterizedTest
    @ValueSource(longs = {1, 2, 3})
    void crashAndReopenLoop(long seed) throws Exception {
        crashLoop(seed, 20, dir);
    }

    @ParameterizedTest
    @ValueSource(longs = {1, 2, 3})
    void aRebuildNeverLosesAWriteThatArrivesDuringIt(long seed) throws Exception {
        rebuildRace(seed, 5, 200_000L);
    }

    @Test
    void theFakeHostForgetsOnlyWhatIsNotDurable() {
        FakeHost h = new FakeHost(DIM, 9, 5);
        h.commit(10);
        h.flush();
        h.commit(5, 1);
        h.crash();
        assertEquals(10, h.live().size());
        assertEquals(10L, h.recoveredMax());
        assertEquals(11L, h.commit(1).adds().get(0).id(), "the lost ids are minted again");
    }

    static void crashLoop(long seed, int iterations, Path dir) throws Exception {
        Random rnd = new Random(seed);
        final long epoch = 1000 + seed;
        FakeHost h = new FakeHost(DIM, seed, epoch);
        h.commit(120);
        h.flush();
        TieredIndex t = TieredIndex.open(dir, DIM, METRIC, FakeHost.cfg(), h);
        try {
            for (int it = 0; it < iterations; it++) {
                int txns = 1 + rnd.nextInt(4);
                for (int k = 0; k < txns; k++) {
                    FakeHost.Txn x = h.commitRandom(rnd);
                    t.apply(FakeHost.batch(x));
                }
                if (rnd.nextInt(10) < 3) h.flush();
                boolean crash = rnd.nextBoolean();
                if (rnd.nextInt(10) < 7) {
                    h.flushDeletes();
                    t.sealNow();
                    t.mergeNow();
                    assertTrue(t.awaitIdle(60_000), "the tier did not settle");
                    FaultHook hook = FaultHook.NONE;
                    if (rnd.nextBoolean()) {
                        final FaultHook.Step stop = FaultHook.Step.values()[rnd.nextInt(FaultHook.Step.values().length)];
                        hook = step -> {
                            if (step == stop) throw new IOException("crash after " + step);
                        };
                        crash = true;
                    }
                    try {
                        t.checkpoint(dir, hook);
                    } catch (IOException e) {
                        if (!String.valueOf(e.getMessage()).startsWith("crash after")) throw e;
                    }
                }
                if (crash) {
                    h.crash();
                    t.close();
                    t = TieredIndex.open(dir, DIM, METRIC, FakeHost.cfg(), h);
                }
                FakeHost.assertMatches(t, h, METRIC, "seed " + seed + " iteration " + it + (crash ? " after a crash" : ""));
                assertEquals(epoch, t.epoch());
            }
        } finally {
            t.close();
        }
    }

    private static TieredIndex.Snapshot slowSnapshot(FakeHost h, int pauseMs) {
        TieredIndex.Snapshot s;
        synchronized (h) {
            s = h.snapshot();
        }
        // the host takes its time between the moment of the snapshot and the moment the library reads it
        if (pauseMs > 0) {
            try {
                Thread.sleep(pauseMs);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        return s;
    }

    static void rebuildRace(long seed, int rounds, long writerPauseNanos) throws Exception {
        final long epoch = 77 + seed;
        final FakeHost h = new FakeHost(DIM, seed, epoch);
        h.commit(150);
        final TieredIndex t = new TieredIndex(DIM, METRIC, FakeHost.cfg());
        final AtomicBoolean stop = new AtomicBoolean();
        final AtomicReference<Throwable> failure = new AtomicReference<>();
        Thread writer = new Thread(() -> {
            Random rnd = new Random(seed * 31 + 7);
            try {
                while (!stop.get()) {
                    synchronized (h) {
                        FakeHost.Txn x = h.commitRandom(rnd);
                        t.apply(FakeHost.batch(x));
                    }
                    LockSupport.parkNanos(writerPauseNanos);
                }
            } catch (Throwable e) {
                failure.set(e);
            }
        }, "fake-host-writer");
        try {
            t.rebuildFrom(() -> slowSnapshot(h, 0), epoch);
            writer.start();
            for (int r = 0; r < rounds; r++) {
                final int pause = 2 + (int) ((seed + r) % 4);
                t.rebuildFrom(() -> slowSnapshot(h, pause), epoch);
            }
            stop.set(true);
            writer.join(60_000);
            assertNull(failure.get(), "the host thread failed: " + failure.get());
            FakeHost.assertMatches(t, h, METRIC, "seed " + seed + " after " + rounds + " rebuilds under writes");
            assertEquals(epoch, t.epoch());
        } finally {
            stop.set(true);
            writer.join(60_000);
            t.close();
        }
    }
}
