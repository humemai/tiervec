package com.humemai.vec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.LockSupport;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Updater threads replace ids in batches while searcher threads run (issue #10). There are GROUPS groups of MEMBERS ids. A batch moves one group from one
 * generation to the next: it adds the MEMBERS ids of the next generation and deletes the MEMBERS ids of the current one. Each group belongs to one updater, so
 * its generations come in order. A search that sees a group must see exactly MEMBERS ids of one generation: more or a mixture means it saw the adds without the
 * deletes of one batch, fewer means it saw the deletes without the adds. A search over all groups must see every group that way at once.
 *
 * <p>The searches use an allow predicate that accepts only member ids. Fewer than {@code ef} ids pass it, so a search is exhaustive (the beam of a segment ends
 * when its frontier is empty, or falls back to an exact scan) and the tail is exact: a missing id is a missing id, not a recall miss. It runs on a tail only and
 * on a tier that seals and merges all the time.
 */
@Tag("slow")
class ApplyStressTest {
    static final int DIM = 16, GROUPS = 8, MEMBERS = 4, WRITERS = 2, READERS = 4, FILLERS = 100;
    static final long BASE = 1000, RUN_MS = 8000;

    static long idOf(long gen, int group, int member) {
        return BASE + (gen * GROUPS + group) * MEMBERS + member;
    }

    static boolean isMember(long id) {
        return id >= BASE;
    }

    static int groupOf(long id) {
        return (int) (((id - BASE) / MEMBERS) % GROUPS);
    }

    static long genOf(long id) {
        return (id - BASE) / ((long) MEMBERS * GROUPS);
    }

    static float[] vec(long id) {
        return TierT.vec(11, id, DIM);
    }

    /** the batch that moves a group from one generation to the next */
    static TieredIndex.Batch move(int group, long fromGen, long toGen) {
        List<TieredIndex.IdVector> adds = new ArrayList<>();
        List<Long> deletes = new ArrayList<>();
        for (int m = 0; m < MEMBERS; m++) {
            long id = idOf(toGen, group, m);
            adds.add(new TieredIndex.IdVector(id, vec(id)));
            deletes.add(idOf(fromGen, group, m));
        }
        return new TieredIndex.Batch(adds, deletes);
    }

    /** null when the hits show every group asked for (onlyGroup, or all when it is negative) with exactly MEMBERS ids of one generation, else what is wrong */
    static String violation(Hit[] hits, int onlyGroup) {
        int[] count = new int[GROUPS];
        long[] gen = new long[GROUPS];
        Arrays.fill(gen, -1L);
        for (Hit h : hits) {
            long id = h.id();
            if (!isMember(id)) return "a hit that is not a member: " + id;
            int g = groupOf(id);
            if (onlyGroup >= 0 && g != onlyGroup) return "a hit of group " + g + " in a search for group " + onlyGroup;
            long ge = genOf(id);
            if (gen[g] >= 0 && gen[g] != ge) return "group " + g + " shows generations " + gen[g] + " and " + ge + " together (the adds without the deletes of one batch)";
            gen[g] = ge;
            count[g]++;
        }
        for (int g = 0; g < GROUPS; g++) {
            if (onlyGroup >= 0 && g != onlyGroup) continue;
            if (count[g] != MEMBERS) {
                return "group " + g + " shows " + count[g] + " of " + MEMBERS + " ids" + (count[g] == 0 ? " (neither the old nor the new ones)" : "") + ", " + hits.length + " hits in all";
            }
        }
        return null;
    }

    @Test
    void batchesAreAtomicOnATail() throws Exception {
        TieredIndex.Config d = TieredIndex.Config.defaults();
        run(new TieredIndex.Config(1 << 28, 0L, 1000, 1000, 4, 1.01, 0, 4, d.tailQuant(), d.graph(), 2), "tail only");
    }

    @Test
    void batchesAreAtomicWhileTheTierSealsAndMerges() throws Exception {
        TieredIndex.Config d = TieredIndex.Config.defaults();
        run(new TieredIndex.Config(40, 0L, 20, 3, 2, 0.3, 0, 4, d.tailQuant(), d.graph(), 1), "seals and merges");
    }

    private void run(TieredIndex.Config cfg, String label) throws Exception {
        TieredIndex ix = new TieredIndex(DIM, Metric.COSINE, cfg);
        try {
            List<TieredIndex.IdVector> init = new ArrayList<>();
            for (long id = 0; id < FILLERS; id++) init.add(new TieredIndex.IdVector(id, vec(id)));
            for (int g = 0; g < GROUPS; g++) {
                for (int m = 0; m < MEMBERS; m++) init.add(new TieredIndex.IdVector(idOf(0, g, m), vec(idOf(0, g, m))));
            }
            ix.apply(new TieredIndex.Batch(init, List.of()));

            final long[] gen = new long[GROUPS]; // each group belongs to one updater; the test thread reads it after the join
            final AtomicBoolean stop = new AtomicBoolean();
            final AtomicReference<String> bad = new AtomicReference<>();
            final AtomicReference<Throwable> error = new AtomicReference<>();
            final AtomicLong batches = new AtomicLong(), searches = new AtomicLong();
            List<Thread> threads = new ArrayList<>();
            for (int w = 0; w < WRITERS; w++) {
                final int wi = w;
                threads.add(new Thread(() -> {
                    try {
                        int turn = 0;
                        while (!stop.get()) {
                            int g = wi + WRITERS * (turn++ % (GROUPS / WRITERS));
                            ix.apply(move(g, gen[g], gen[g] + 1));
                            gen[g]++;
                            batches.incrementAndGet();
                            LockSupport.parkNanos(500_000L);
                        }
                    } catch (Throwable t) {
                        error.compareAndSet(null, t);
                        stop.set(true);
                    }
                }, "apply-updater-" + w));
            }
            for (int r = 0; r < READERS; r++) {
                final int ri = r;
                threads.add(new Thread(() -> {
                    try {
                        SplittableRandom rnd = new SplittableRandom(100 + ri);
                        float[] q = TierT.vec(5, ri, DIM);
                        boolean all = (ri & 1) == 0;
                        while (!stop.get()) {
                            final int g = all ? -1 : rnd.nextInt(GROUPS);
                            Hit[] hits = all ? ix.search(q, GROUPS * MEMBERS + 8, id -> isMember(id)) : ix.search(q, 2 * MEMBERS + 4, id -> isMember(id) && groupOf(id) == g);
                            searches.incrementAndGet();
                            String v = violation(hits, g);
                            if (v != null) {
                                bad.compareAndSet(null, v);
                                stop.set(true);
                            }
                        }
                    } catch (Throwable t) {
                        error.compareAndSet(null, t);
                        stop.set(true);
                    }
                }, "apply-searcher-" + r));
            }
            for (Thread t : threads) t.start();
            long end = System.nanoTime() + RUN_MS * 1_000_000L;
            while (!stop.get() && System.nanoTime() < end) Thread.sleep(50);
            stop.set(true);
            for (Thread t : threads) {
                t.join(60_000);
                assertFalse(t.isAlive(), "thread " + t.getName() + " did not stop (" + label + ")");
            }

            if (error.get() != null) fail("a thread failed (" + label + ")", error.get());
            assertNull(bad.get(), "a search saw a batch half applied (" + label + "): " + bad.get());
            // Proof that updaters and searchers really overlapped, not a throughput floor: CI VMs differ by 10x in speed and this test is about
            // atomicity of apply(), not speed.
            assertTrue(batches.get() >= 50, "the updaters applied only " + batches.get() + " batches (" + label + ")");
            assertTrue(searches.get() >= 50, "the searchers finished only " + searches.get() + " searches (" + label + ")");

            ix.awaitIdle(60_000);
            Set<Long> expect = new HashSet<>();
            for (int g = 0; g < GROUPS; g++) {
                for (int m = 0; m < MEMBERS; m++) expect.add(idOf(gen[g], g, m));
            }
            Set<Long> seen = new HashSet<>();
            for (Hit h : ix.search(TierT.vec(5, 99, DIM), GROUPS * MEMBERS + 8, id -> isMember(id))) seen.add(h.id());
            assertEquals(expect, seen, "the final generation of every group, nothing else (" + label + ")");
            assertEquals(GROUPS * MEMBERS + FILLERS, ix.size(), "size() after the run (" + label + ")");
            assertEquals(GROUPS * MEMBERS + FILLERS, ix.stats().live(), "stats().live() after the run (" + label + ")");
            assertEquals(0L, ix.stats().failures(), "background failures (" + label + ")");
            assertNull(ix.lastError(), "last background error (" + label + ")");
        } finally {
            ix.close();
        }
    }
}
