package ai.humem.tiervec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.LongPredicate;
import org.junit.jupiter.api.Test;

/**
 * A search that started before {@code rebuildFrom} swaps the rebuilt state in must not return an id that was deleted before the search started, nor an older
 * vector of an id that was added again (issue #28, the recovery path of the race in issue #26), without any timing. A search reads the view once and decides at
 * scan time, per row, whether a copy in a segment is hidden by the version floor. {@code rebuildFrom} replaces the view, and it must keep the floor entries that
 * a search still running on the old view needs, the same way a merge swap does.
 *
 * <p>The test holds two searches inside their first call of the allow predicate, one with the vector of an id that was deleted and one with the older vector of
 * an id that was added again with a new vector, runs {@code rebuildFrom} on the test thread with a source that holds the current records, and lets both searches
 * go on. Only the public API is used. The floor entries are not part of the assertions while the searches run: the answers are. After the searches end the floor
 * map must be empty and the tier idle (a floor entry that nobody drops would keep a hidden copy hidden for ever).
 */
class RebuildSwapRaceTest {
    static final int DIM = 32, ROWS = 1000, DELETED = 1500, READDED = 500;

    /** a search that stops inside its first call of the allow predicate, and goes on when told to */
    private static final class HeldSearch {
        final CountDownLatch inside = new CountDownLatch(1);
        final CountDownLatch resume = new CountDownLatch(1);
        final AtomicBoolean first = new AtomicBoolean();
        final AtomicReference<Hit[]> result = new AtomicReference<>();
        final AtomicReference<Throwable> error = new AtomicReference<>();
        final Thread thread;

        HeldSearch(TieredIndex ix, float[] q, String name) {
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
            thread = new Thread(() -> {
                try {
                    result.set(ix.search(q, 10, allow, 200));
                } catch (Throwable t) {
                    error.set(t);
                }
            }, name);
            thread.setDaemon(true);
            thread.start();
        }

        Hit[] finish() throws InterruptedException {
            resume.countDown();
            thread.join(60_000);
            assertFalse(thread.isAlive(), "the search did not finish");
            assertNull(error.get(), "the search failed");
            assertNotNull(result.get(), "the search returned nothing");
            return result.get();
        }
    }

    @Test
    void searchThatStartedBeforeARebuildSwapDoesNotReturnHiddenCopies() throws Exception {
        TieredIndex.Config d = TieredIndex.Config.defaults();
        // only sealNow and mergeNow seal and merge; the merge is not throttled
        TieredIndex.Config cfg = new TieredIndex.Config(1 << 28, 0L, 1000, 1000, 4, 1.01, 0, 4, d.tailQuant(), d.graph(), 2);
        TieredIndex ix = new TieredIndex(DIM, Metric.COSINE, cfg);
        try {
            for (int b = 0; b < 3; b++) {
                for (int id = b * ROWS; id < (b + 1) * ROWS; id++) ix.add(id, RidReuseProbe.vec(1, id, 0, DIM));
                // the id is added again with a new vector, and the new copy is sealed into the newest segment
                if (b == 2) ix.add(READDED, RidReuseProbe.vec(1, READDED, 1, DIM));
                ix.sealNow();
            }
            assertEquals(3, ix.segmentCount(), "three sealed segments");
            assertEquals(0, ix.pendingTail(), "an empty tail");
            assertTrue(ix.delete(DELETED), "the id is live before the delete");
            assertEquals(2, ix.stats().floorEntries(), "the add again and the delete wrote one floor entry each");

            // the live records, which is what the host would hand to rebuildFrom: no deleted id, the new vector of the id that was added again
            List<TieredIndex.IdVector> src = new ArrayList<>();
            for (int id = 0; id < 3 * ROWS; id++) {
                if (id == DELETED) continue;
                src.add(new TieredIndex.IdVector(id, RidReuseProbe.vec(1, id, id == READDED ? 1 : 0, DIM)));
            }

            // each query is the vector of a copy that the floor hides, so that an unhidden copy ranks first in its segment
            HeldSearch forDeleted = new HeldSearch(ix, RidReuseProbe.vec(1, DELETED, 0, DIM), "rebuild-swap-search-deleted");
            HeldSearch forReadded = new HeldSearch(ix, RidReuseProbe.vec(1, READDED, 0, DIM), "rebuild-swap-search-readded");
            assertTrue(forDeleted.inside.await(60, TimeUnit.SECONDS), "the search never called the allow predicate");
            assertTrue(forReadded.inside.await(60, TimeUnit.SECONDS), "the search never called the allow predicate");

            ix.rebuildFrom(src); // the swap, while both searches hold the old view
            assertEquals(1, ix.segmentCount(), "the rebuild swapped in one segment");
            assertEquals(3 * ROWS - 1, ix.size(), "the rebuilt tier holds the live records");

            Hit[] deletedHits = forDeleted.finish();
            Hit[] readdedHits = forReadded.finish();
            for (Hit h : deletedHits) {
                assertTrue(h.id() != DELETED, "id " + DELETED + " was deleted before the search started, yet the search returned it with score " + h.score()
                        + " after rebuildFrom swapped in a new view (floor entries now: " + ix.stats().floorEntries() + ")");
            }
            for (Hit h : readdedHits) {
                assertTrue(h.id() != READDED || h.score() < 0.99f, "id " + READDED + " was added again before the search started, yet the search returned its older vector"
                        + " with score " + h.score() + " after rebuildFrom swapped in a new view (floor entries now: " + ix.stats().floorEntries() + ")");
            }

            assertTrue(ix.awaitIdle(10_000), "the tier did not settle after the last search of the old view ended");
            assertEquals(0, ix.stats().floorEntries(), "a floor entry was left after the last search of the old view ended");
        } finally {
            ix.close();
        }
    }
}
