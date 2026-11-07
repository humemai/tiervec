package com.humemai.vec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.LongPredicate;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * A search that starts before a merge swaps its new segment in must not return an id that was deleted before the search started (issue #5, concurrent variant),
 * without any timing. A search reads the view once and decides at scan time, per row, whether a copy in a segment is hidden by the version floor. The merge
 * drops the hidden copies from the merged segment and then removes the floor entries that no segment needs any more. A search that holds the old view and is
 * still scanning an old segment then finds no floor entry and sees the hidden copy as valid.
 *
 * <p>The test holds the search inside its first call of the allow predicate, runs {@code mergeNow()} on the test thread, and lets the search go on. Only the
 * public API is used. The check is on the real class: the floor entry outlives the search that still reads the old view, and it is gone a short time after
 * that search ends (a leaked entry would keep a hidden copy hidden for ever and fail the last assertion).
 */
@Tag("slow")
class RidReuseSwapRaceTest {
    static final int DIM = 32, ROWS = 1000, TARGET = 1500;

    @Test
    void searchThatStartedBeforeAMergeSwapDoesNotReturnAnIdDeletedBeforeItStarted() throws Exception {
        TieredIndex.Config d = TieredIndex.Config.defaults();
        // only sealNow and mergeNow seal and merge; the merge is not throttled
        TieredIndex.Config cfg = new TieredIndex.Config(1 << 28, 0L, 1000, 1000, 4, 1.01, 0, 4, d.tailQuant(), d.graph(), 2);
        TieredIndex ix = new TieredIndex(DIM, Metric.COSINE, cfg);
        try {
            for (int b = 0; b < 3; b++) {
                for (int id = b * ROWS; id < (b + 1) * ROWS; id++) ix.add(id, RidReuseProbe.vec(1, id, 0, DIM));
                ix.sealNow();
            }
            assertEquals(3, ix.segmentCount(), "three sealed segments");
            assertEquals(0, ix.pendingTail(), "an empty tail");
            float[] q = RidReuseProbe.vec(1, TARGET, 0, DIM);
            assertTrue(ix.delete(TARGET), "the id is live before the delete");
            assertEquals(1, ix.stats().floorEntries(), "the delete wrote one floor entry");

            CountDownLatch inside = new CountDownLatch(1);
            CountDownLatch resume = new CountDownLatch(1);
            AtomicBoolean first = new AtomicBoolean();
            LongPredicate allow = id -> {
                if (first.compareAndSet(false, true)) {
                    inside.countDown();
                    try {
                        resume.await(60, TimeUnit.SECONDS);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                }
                return true;
            };
            AtomicReference<Hit[]> result = new AtomicReference<>();
            AtomicReference<Throwable> error = new AtomicReference<>();
            Thread searcher = new Thread(() -> {
                try {
                    result.set(ix.search(q, 10, allow, 200));
                } catch (Throwable t) {
                    error.set(t);
                }
            }, "ridreuse-swap-search");
            searcher.setDaemon(true);
            searcher.start();
            assertTrue(inside.await(60, TimeUnit.SECONDS), "the search never called the allow predicate");

            ix.mergeNow(); // the swap, and the floor cleanup, while the search holds the old view
            assertEquals(1, ix.segmentCount(), "the merge swapped in one segment");
            resume.countDown();
            searcher.join(60_000);
            assertFalse(searcher.isAlive(), "the search did not finish");
            assertNull(error.get(), "the search failed");
            Hit[] hits = result.get();
            assertNotNull(hits, "the search returned nothing");
            for (Hit h : hits) {
                assertTrue(h.id() != TARGET, "id " + TARGET + " was deleted before the search started, yet the search returned it with score " + h.score()
                        + " after a merge swapped in a new segment and dropped the floor entry (floor entries now: " + ix.stats().floorEntries() + ")");
            }
            long end = System.nanoTime() + 10_000_000_000L; // the worker drops the entry once no search holds the old view
            while (ix.stats().floorEntries() != 0 && System.nanoTime() < end) Thread.sleep(10);
            assertEquals(0, ix.stats().floorEntries(), "the floor entry was not dropped after the last search of the old view ended");
        } finally {
            ix.close();
        }
    }
}
