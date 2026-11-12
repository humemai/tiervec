package ai.humem.tiervec;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicLongArray;

/**
 * Tests of TieredIndex. Plain Java, no JUnit. Usage: TierTests [quick|full] [name-substring]
 * quick (default): every test, the 8 writers + 8 readers test runs 10 s. full: the same test runs 60 s and the stress loops are longer.
 */
public final class TierTests {
    static final int DIM = 32;
    static boolean full;

    static TieredIndex.Config cfg(int tail, int small, int maxSeg) {
        return new TieredIndex.Config(tail, 0L, small, maxSeg, 4, 0.3, 0, 4, Quant.NONE,
                new GraphIndex.Params(16, 100, 200, 1.0f, false, 7L), 1);
    }

    static void fill(TieredIndex ix, Map<Long, float[]> model, long seed, int from, int to) {
        for (long id = from; id < to; id++) {
            float[] v = TierT.vec(seed, id, DIM);
            ix.add(id, v);
            model.put(id, v);
        }
    }

    static List<TieredIndex> open = new ArrayList<>();

    static TieredIndex mk(Metric m, TieredIndex.Config c) {
        TieredIndex ix = new TieredIndex(DIM, m, c);
        open.add(ix);
        return ix;
    }

    static double meanRecall(TieredIndex ix, Map<Long, float[]> model, Metric m, int nq, long seed, java.util.function.LongPredicate allow) {
        double s = 0;
        for (int i = 0; i < nq; i++) {
            float[] q = TierT.vec(seed, i, DIM);
            s += TierT.recall(ix.search(q, 10, allow), TierT.brute(model, q, 10, allow, m));
        }
        return s / nq;
    }

    // ------------------------------------------------------------------------------------------------------------------

    static void testTailOnly() {
        TieredIndex ix = mk(Metric.COSINE, cfg(2000, 1000, 8));
        Map<Long, float[]> model = new HashMap<>();
        fill(ix, model, 1, 0, 800);
        TierT.check(ix.size() == 800, "tail: size 800");
        int bad = 0;
        for (int i = 0; i < 60; i++) {
            float[] q = TierT.vec(2, i, DIM);
            TieredIndex.SearchResult r = ix.search(q, 10, null, TieredIndex.Freshness.FRESH);
            if (!TierT.sameIdSet(r.hits(), TierT.brute(model, q, 10, null, Metric.COSINE)) || !TierT.sortedDesc(r.hits())) bad++;
            if (!r.exact() || r.segments() != 0 || r.tailScanned() != 800) bad++;
        }
        TierT.check(bad == 0, "tail: exact equals brute force, meta exact, 0 segments, tailScanned 800 (bad=" + bad + ")");
        bad = 0;
        for (int i = 0; i < 40; i++) {
            float[] q = TierT.vec(3, i, DIM);
            Hit[] h = ix.search(q, 10, id -> id % 3 == 0);
            if (!TierT.sameIdSet(h, TierT.brute(model, q, 10, id -> id % 3 == 0, Metric.COSINE))) bad++;
        }
        TierT.check(bad == 0, "tail: allow-list equals brute force (bad=" + bad + ")");
        // delete and replace inside the tail
        boolean delOk = true;
        for (long id = 0; id < 800; id += 8) {
            delOk &= ix.delete(id);
            model.remove(id);
        }
        TierT.check(delOk, "tail: delete returned true for existing ids");
        TierT.check(!ix.delete(0), "tail: second delete returns false");
        TierT.check(!ix.delete(123456), "tail: delete of a never added id returns false");
        for (long id = 1; id < 800; id += 8) {
            float[] v = TierT.vec(9, id, DIM);
            ix.add(id, v);
            model.put(id, v);
        }
        TierT.check(ix.size() == 700 && model.size() == 700, "tail: size 700 after 100 deletes and 100 replaces");
        bad = 0;
        for (int i = 0; i < 60; i++) {
            float[] q = TierT.vec(4, i, DIM);
            Hit[] h = ix.search(q, 10, null);
            if (!TierT.sameIdSet(h, TierT.brute(model, q, 10, null, Metric.COSINE))) bad++;
            for (Hit x : h) if (x.id() % 8 == 0) bad++;
        }
        TierT.check(bad == 0, "tail: after delete and replace equals brute force of the model, no deleted id (bad=" + bad + ")");
        TierT.check(ix.awaitIdle(5000), "tail: idle");
        TierT.check(ix.stats().seals() == 0 && ix.segmentCount() == 0, "tail: no seal and no segment below tailLimit");
        // grow past the tail limit: a seal must happen
        fill(ix, model, 1, 800, 2100);
        TierT.check(ix.awaitIdle(60_000), "tail: idle after growth");
        TierT.check(ix.stats().seals() >= 1 && ix.segmentCount() >= 1, "tail: sealed after reaching tailLimit");
        TierT.check(ix.pendingTail() < 2000, "tail: tail bounded after the seal (" + ix.pendingTail() + ")");
        TierT.check(ix.size() == model.size(), "tail: size equals model after growth");
    }

    static void testNoGraphBelowSmall() {
        TieredIndex ix = mk(Metric.COSINE, cfg(100, 1000, 8));
        Map<Long, float[]> model = new HashMap<>();
        fill(ix, model, 11, 0, 900);
        TierT.check(ix.awaitIdle(5000), "small: idle");
        TierT.check(ix.stats().seals() == 0 && ix.segmentCount() == 0, "small: 900 vectors, tailLimit 100, no graph is built (smallLimit 1000)");
        TierT.check(ix.pendingTail() == 900, "small: all 900 in the tail");
        TieredIndex.SearchResult r = ix.search(TierT.vec(12, 0, DIM), 10, null, TieredIndex.Freshness.FRESH);
        TierT.check(r.exact() && r.segments() == 0, "small: search is exact");
        fill(ix, model, 11, 900, 1100);
        TierT.check(ix.awaitIdle(60_000), "small: idle after crossing smallLimit");
        TierT.check(ix.stats().seals() >= 1, "small: a graph segment appears once live >= smallLimit");
        TierT.check(ix.size() == 1100, "small: size 1100");
    }

    static void testSegmentsRecall() {
        TieredIndex ix = mk(Metric.COSINE, cfg(500, 300, 8));
        Map<Long, float[]> model = new HashMap<>();
        fill(ix, model, 21, 0, 3000);
        TierT.check(ix.awaitIdle(120_000), "seg: idle");
        fill(ix, model, 21, 3000, 3150); // some of these sit in the tail
        double r = meanRecall(ix, model, Metric.COSINE, 100, 22, null);
        System.out.printf("  seg: mean recall@10 %.4f (segments %d, tail %d)%n", r, ix.segmentCount(), ix.pendingTail());
        TierT.check(r >= 0.93, "seg: recall@10 >= 0.93 (" + r + ")");
        double rf = meanRecall(ix, model, Metric.COSINE, 100, 23, id -> id % 5 == 0);
        System.out.printf("  seg: mean filtered recall@10 %.4f%n", rf);
        TierT.check(rf >= 0.90, "seg: filtered recall@10 >= 0.90 (" + rf + ")");
        TieredIndex.SearchResult sr = ix.search(TierT.vec(22, 1, DIM), 10, null, TieredIndex.Freshness.FRESH);
        TierT.check(!sr.exact() && sr.segments() >= 1, "seg: meta says approximate, segments >= 1");
        int miss = 0;
        for (long id = 3000; id < 3150; id++) if (!TierT.containsId(ix.search(TierT.vec(21, id, DIM), 5, null), id)) miss++;
        TierT.check(miss == 0, "seg: the newest 150 vectors are found (miss=" + miss + ")");
        TierT.check(ix.size() == 3150 && ix.stats().failures() == 0, "seg: size and no worker failure");
    }

    /** filterScanBelow on the tier: a small allowed share is answered exactly in every segment, a larger share is unchanged by the option. */
    static void testFilterScanBelow() {
        TieredIndex ix = mk(Metric.COSINE, cfg(500, 300, 8));
        Map<Long, float[]> model = new HashMap<>();
        fill(ix, model, 41, 0, 3000);
        TierT.check(ix.awaitIdle(120_000), "scanbelow: idle");
        fill(ix, model, 41, 3000, 3150);
        TierT.check(ix.segmentCount() >= 1, "scanbelow: at least one segment");
        TierT.check(ix.filterScanBelow() == 0.0, "scanbelow: off by default");
        for (double bad : new double[] { -0.5, 2.0, Double.NaN }) {
            try { ix.filterScanBelow(bad); TierT.check(false, "scanbelow: " + bad + " must throw"); } catch (IllegalArgumentException ok) { }
        }
        java.util.function.LongPredicate big = id -> id % 5 == 0; // share 0.2
        Hit[][] before = new Hit[20][];
        for (int i = 0; i < 20; i++) before[i] = ix.search(TierT.vec(42, i, DIM), 10, big);
        ix.filterScanBelow(0.05);
        TierT.check(ix.filterScanBelow() == 0.05, "scanbelow: getter returns the setting");
        boolean same = true;
        for (int i = 0; i < 20; i++) {
            Hit[] a = ix.search(TierT.vec(42, i, DIM), 10, big);
            if (a.length != before[i].length) same = false;
            else for (int j = 0; j < a.length; j++) if (a[j].id() != before[i][j].id()) same = false;
        }
        TierT.check(same, "scanbelow: share 0.2 with the option on returns the same hits as with it off");
        java.util.function.LongPredicate small = id -> id % 250 == 0; // share 0.004
        double r = meanRecall(ix, model, Metric.COSINE, 40, 43, small);
        TierT.check(r >= 0.995, "scanbelow: share 0.004 equals brute force in every segment, up to float ties (recall " + r + ")");
        TierT.check(ix.search(TierT.vec(44, 1, DIM), 10, id -> false).length == 0, "scanbelow: reject-all returns nothing");
        TierT.check(ix.stats().failures() == 0, "scanbelow: no worker failure");
    }

    static void testMetrics() {
        for (Metric m : new Metric[] {Metric.L2, Metric.DOT}) {
            TieredIndex ix = mk(m, cfg(500, 300, 8));
            Map<Long, float[]> model = new HashMap<>();
            fill(ix, model, 31, 0, 1700);
            TierT.check(ix.awaitIdle(120_000), m + ": idle");
            double r = meanRecall(ix, model, m, 60, 32, null);
            System.out.printf("  %s: mean recall@10 %.4f%n", m, r);
            TierT.check(r >= 0.90, m + ": recall@10 >= 0.90 (" + r + ")");
            Hit[] h = ix.search(TierT.vec(32, 0, DIM), 5, null);
            boolean ok = true;
            for (Hit x : h) ok &= Math.abs(x.score() - TierT.score(m, TierT.vec(32, 0, DIM), model.get(x.id()))) < 1e-3f;
            TierT.check(ok, m + ": returned scores equal the exact score");
        }
    }

    /** the hooks run at the instant of the swap: a vector is found just before it, and just after it */
    static void testSwapInstantHooks() {
        TieredIndex ix = mk(Metric.COSINE, cfg(100, 50, 8));
        Map<Long, float[]> model = new HashMap<>();
        fill(ix, model, 41, 0, 60);
        TierT.check(ix.awaitIdle(5000) && ix.segmentCount() == 0, "hook: 60 vectors stay in the tail");
        AtomicInteger missBefore = new AtomicInteger(), missAfter = new AtomicInteger(), ran = new AtomicInteger();
        AtomicInteger pendBefore = new AtomicInteger(-1), pendAfter = new AtomicInteger(-1);
        ix.setSealHooks(() -> {
            pendBefore.set(ix.pendingTail());
            for (long id = 0; id < 60; id++) if (!TierT.containsId(ix.search(TierT.vec(41, id, DIM), 3, null), id)) missBefore.incrementAndGet();
            ran.incrementAndGet();
        }, () -> {
            pendAfter.set(ix.pendingTail());
            for (long id = 0; id < 60; id++) if (!TierT.containsId(ix.search(TierT.vec(41, id, DIM), 3, null), id)) missAfter.incrementAndGet();
            ran.incrementAndGet();
        });
        ix.sealNow();
        TierT.check(ran.get() == 2, "hook: both hooks ran");
        TierT.check(missBefore.get() == 0, "hook: 0 misses before the swap (frozen tail still serves) miss=" + missBefore.get());
        TierT.check(missAfter.get() == 0, "hook: 0 misses after the swap (segment serves) miss=" + missAfter.get());
        TierT.check(pendBefore.get() == 60 && pendAfter.get() == 0, "hook: tail 60 before the swap, 0 after (" + pendBefore.get() + ", " + pendAfter.get() + ")");
        TierT.check(ix.segmentCount() == 1, "hook: one segment");
        // sealed-only now sees the 60
        int seen = 0;
        for (long id = 0; id < 60; id++)
            if (TierT.containsId(ix.search(TierT.vec(41, id, DIM), 3, null, TieredIndex.Freshness.SEALED_ONLY).hits(), id)) seen++;
        TierT.check(seen == 60, "hook: sealed-only finds all 60 after the seal (" + seen + ")");
    }

    /** one thread adds and immediately searches the id it just added, another thread chases the newest id; seals run all the time */
    static void testSealStress() {
        final int n = full ? 20_000 : 6000;
        final TieredIndex ix = mk(Metric.COSINE, cfg(150, 100, 4));
        final AtomicLong lastDone = new AtomicLong(-1);
        final AtomicBoolean stop = new AtomicBoolean();
        final AtomicInteger missW = new AtomicInteger(), missR = new AtomicInteger(), checksR = new AtomicInteger(), errs = new AtomicInteger();
        Thread rd = new Thread(() -> {
            long seen = -1;
            try {
                while (!stop.get()) {
                    long id = lastDone.get();
                    if (id >= 0 && id != seen) {
                        seen = id;
                        Hit[] h = ix.search(TierT.vec(51, id, DIM), 5, null);
                        checksR.incrementAndGet();
                        if (!TierT.containsId(h, id)) missR.incrementAndGet();
                    } else {
                        Thread.onSpinWait();
                    }
                }
            } catch (Throwable t) {
                errs.incrementAndGet();
                t.printStackTrace(System.out);
            }
        }, "stress-reader");
        rd.start();
        for (long id = 0; id < n; id++) {
            float[] v = TierT.vec(51, id, DIM);
            ix.add(id, v);
            lastDone.set(id);
            if (!TierT.containsId(ix.search(v, 5, null), id)) missW.incrementAndGet();
        }
        stop.set(true);
        try {
            rd.join();
        } catch (InterruptedException e) {
            throw new IllegalStateException(e);
        }
        TierT.check(ix.awaitIdle(120_000), "stress: idle");
        TieredIndex.Stats s = ix.stats();
        System.out.println("  stress: " + s + " readerChecks=" + checksR.get() + " writerMiss=" + missW.get() + " readerMiss=" + missR.get());
        TierT.check(errs.get() == 0 && s.failures() == 0, "stress: no exception");
        TierT.check(missW.get() == 0, "stress: writer finds each vector right after add (miss=" + missW.get() + ")");
        TierT.check(missR.get() == 0, "stress: other thread finds the newest vector right after add returned (miss=" + missR.get() + ")");
        TierT.check(s.seals() >= 5, "stress: seals ran during the loop (" + s.seals() + ")");
        TierT.check(ix.size() == n, "stress: size == n");
    }

    static void testMerge() {
        TieredIndex ix = mk(Metric.COSINE, cfg(300, 200, 3));
        Map<Long, float[]> model = new HashMap<>();
        // batches of exactly tailLimit with a wait between them: one seal per batch, so 10 segments are produced and the cap of 3 must merge
        // (a fast writer that is not paced produces a few large seals instead, because the tail keeps growing while a seal builds)
        for (int b = 0; b < 3000; b += 300) {
            fill(ix, model, 61, b, b + 300);
            TierT.check(ix.awaitIdle(120_000), "merge: idle after batch " + b);
        }
        TieredIndex.Stats s0 = ix.stats();
        TierT.check(s0.merges() >= 1, "merge: the segment cap forced merges (" + s0.merges() + ")");
        TierT.check(ix.segmentCount() <= 3, "merge: segments <= cap 3 (" + ix.segmentCount() + ")");
        double r0 = meanRecall(ix, model, Metric.COSINE, 100, 62, null);
        System.out.printf("  merge: recall after merges %.4f, %s%n", r0, s0);
        TierT.check(r0 >= 0.93, "merge: recall >= 0.93 after merges (" + r0 + ")");
        // delete 40 percent
        boolean allTrue = true;
        List<Long> deleted = new ArrayList<>();
        for (long id = 0; id < 3000; id++) {
            if (id % 5 <= 1) {
                allTrue &= ix.delete(id);
                model.remove(id);
                deleted.add(id);
            }
        }
        TierT.check(allTrue && ix.size() == 1800, "merge: 1200 deletes, size 1800 (" + ix.size() + ")");
        checkNoDeleted(ix, deleted, 61, "merge/pre-compaction");
        double r1 = meanRecall(ix, model, Metric.COSINE, 100, 63, null);
        TierT.check(r1 >= 0.92, "merge: recall >= 0.92 with 40 percent hidden (" + r1 + ")");
        TierT.check(ix.awaitIdle(120_000), "merge: idle after deletes (dead-fraction compaction)");
        ix.mergeNow();
        TierT.check(ix.awaitIdle(120_000), "merge: idle after mergeNow");
        TierT.check(ix.segmentCount() <= 1, "merge: one segment after mergeNow (" + ix.segmentCount() + ")");
        TierT.check(ix.stats().floorEntries() == 0, "merge: floor map empty once every hidden copy is dropped (" + ix.stats().floorEntries() + ")");
        TierT.check(ix.size() == 1800, "merge: size 1800 after merge");
        checkNoDeleted(ix, deleted, 61, "merge/post-merge");
        double r2 = meanRecall(ix, model, Metric.COSINE, 100, 64, null);
        TierT.check(r2 >= 0.93, "merge: recall >= 0.93 after the compaction (" + r2 + ")");
        TierT.check(ix.stats().failures() == 0, "merge: no worker failure");
    }

    static void checkNoDeleted(TieredIndex ix, List<Long> deleted, long seed, String label) {
        int bad = 0;
        int step = Math.max(1, deleted.size() / 150);
        for (int i = 0; i < deleted.size(); i += step) {
            long id = deleted.get(i);
            Hit[] h = ix.search(TierT.vec(seed, id, DIM), 10, null);
            if (TierT.containsId(h, id)) bad++;
            Hit[] h2 = ix.search(TierT.vec(seed, id, DIM), 10, null, TieredIndex.Freshness.SEALED_ONLY).hits();
            if (TierT.containsId(h2, id)) bad++;
        }
        TierT.check(bad == 0, label + ": deleted ids are never returned, fresh or sealed-only (bad=" + bad + ")");
    }

    /** every hit is a live id, carries the score of its CURRENT vector, is distinct and sorted */
    static int badHits(TieredIndex ix, Map<Long, float[]> model, List<float[]> queries, TieredIndex.Freshness f, Set<Long> mustNotSee) {
        int bad = 0;
        for (float[] q : queries) {
            TieredIndex.SearchResult r = ix.search(q, 10, null, f);
            Hit[] h = r.hits();
            if (!TierT.sortedDesc(h) || !TierT.distinct(h)) bad++;
            if (f == TieredIndex.Freshness.SEALED_ONLY && r.tailScanned() != 0) bad++;
            for (Hit x : h) {
                float[] cur = model.get(x.id());
                if (cur == null) { bad++; continue; }
                if (mustNotSee != null && mustNotSee.contains(x.id())) bad++;
                if (Math.abs(x.score() - TierT.score(Metric.COSINE, q, cur)) > 1e-3f) bad++;
            }
        }
        return bad;
    }

    static void testDeleteReplace() {
        TieredIndex ix = mk(Metric.COSINE, cfg(300, 200, 8));
        Map<Long, float[]> model = new HashMap<>();
        fill(ix, model, 71, 0, 2000);
        TierT.check(ix.awaitIdle(120_000), "dr: idle");
        Set<Long> deleted = new HashSet<>();
        Map<Long, float[]> oldVec = new HashMap<>();
        for (long id = 0; id < 2000; id++) {
            if (id % 7 == 0) {
                ix.delete(id);
                model.remove(id);
                deleted.add(id);
            } else if (id % 7 == 1) {
                oldVec.put(id, model.get(id));
                float[] v = TierT.vec(72, id, DIM);
                ix.add(id, v);
                model.put(id, v);
            }
        }
        fill(ix, model, 71, 2000, 2100);
        for (long id = 2000; id < 2020; id++) {
            ix.delete(id);
            model.remove(id);
            deleted.add(id);
        }
        for (long id = 2020; id < 2040; id++) {
            oldVec.put(id, model.get(id));
            float[] v = TierT.vec(72, id, DIM);
            ix.add(id, v);
            model.put(id, v);
        }
        TierT.check(ix.size() == model.size(), "dr: size equals model (" + ix.size() + " vs " + model.size() + ")");
        List<float[]> queries = new ArrayList<>();
        for (int i = 0; i < 120; i++) queries.add(TierT.vec(73, i, DIM));
        for (Map.Entry<Long, float[]> e : oldVec.entrySet()) if (e.getKey() % 5 == 0) queries.add(e.getValue()); // old vectors as queries
        for (long id : deleted) if (id % 5 == 0) queries.add(TierT.vec(71, id, DIM));                             // deleted vectors as queries
        for (String phase : new String[] {"live (worker may be sealing or merging)", "quiet", "after mergeNow"}) {
            if (phase.equals("quiet")) TierT.check(ix.awaitIdle(120_000), "dr: idle");
            if (phase.equals("after mergeNow")) {
                ix.mergeNow();
                TierT.check(ix.awaitIdle(120_000), "dr: idle after mergeNow");
                TierT.check(ix.stats().floorEntries() == 0, "dr: floor empty after mergeNow (" + ix.stats().floorEntries() + ")");
            }
            int bad = badHits(ix, model, queries, TieredIndex.Freshness.FRESH, deleted);
            TierT.check(bad == 0, "dr/" + phase + ": every hit live, newest score, no deleted id, sorted (bad=" + bad + ")");
            int missNew = 0, oldWins = 0;
            for (Map.Entry<Long, float[]> e : model.entrySet()) {
                long id = e.getKey();
                if (!oldVec.containsKey(id)) continue;
                Hit[] h = ix.search(e.getValue(), 5, null);
                if (!TierT.containsId(h, id)) missNew++;
                else for (Hit x : h) if (x.id() == id && x.score() < 0.999f) oldWins++;
            }
            TierT.check(missNew == 0 && oldWins == 0, "dr/" + phase + ": replaced ids are found by their NEW vector with score ~1 (miss=" + missNew + ", low=" + oldWins + ")");
            int bs = badHits(ix, model, queries, TieredIndex.Freshness.SEALED_ONLY, deleted);
            TierT.check(bs == 0, "dr/" + phase + ": sealed-only never returns a deleted, stale or phantom id (bad=" + bs + ")");
        }
        // sealed-only may lag: a vector added after the last seal is missing from it, present in fresh
        ix.sealNow();
        Map<Long, float[]> fresh = new HashMap<>();
        for (long id = 9000; id < 9005; id++) {
            float[] v = TierT.vec(74, id, DIM);
            ix.add(id, v);
            fresh.put(id, v);
            model.put(id, v);
        }
        TierT.check(ix.pendingTail() == 5, "dr: 5 vectors pending in the tail (" + ix.pendingTail() + ")");
        int lag = 0, freshSeen = 0;
        for (Map.Entry<Long, float[]> e : fresh.entrySet()) {
            if (!TierT.containsId(ix.search(e.getValue(), 5, null, TieredIndex.Freshness.SEALED_ONLY).hits(), e.getKey())) lag++;
            if (TierT.containsId(ix.search(e.getValue(), 5, null, TieredIndex.Freshness.FRESH).hits(), e.getKey())) freshSeen++;
        }
        TierT.check(lag == 5 && freshSeen == 5, "dr: sealed-only lags (5 of 5 missing), fresh sees all 5 (lag=" + lag + ", fresh=" + freshSeen + ")");
        int bs = badHits(ix, model, queries, TieredIndex.Freshness.SEALED_ONLY, null);
        // model now has the 5 fresh ids, which sealed-only simply does not return: still no phantom and no stale vector
        TierT.check(bs == 0, "dr: sealed-only after fresh adds still returns only current vectors (bad=" + bs + ")");
        TierT.check(ix.stats().failures() == 0, "dr: no worker failure");
    }

    static void testRebuild() {
        // (a) exact tier: equality
        TieredIndex a = mk(Metric.COSINE, cfg(2000, 1000, 8));
        Map<Long, float[]> model = new HashMap<>();
        fill(a, model, 81, 0, 600);
        for (long id = 0; id < 600; id += 6) {
            a.delete(id);
            model.remove(id);
        }
        for (long id = 1; id < 600; id += 6) {
            float[] v = TierT.vec(82, id, DIM);
            a.add(id, v);
            model.put(id, v);
        }
        TieredIndex b = mk(Metric.COSINE, cfg(2000, 1000, 8));
        List<TieredIndex.IdVector> raw = new ArrayList<>();
        for (Map.Entry<Long, float[]> e : model.entrySet()) raw.add(new TieredIndex.IdVector(e.getKey(), e.getValue()));
        b.rebuildFrom(raw);
        TierT.check(a.size() == b.size() && b.size() == model.size(), "rebuild/exact: sizes equal");
        int diff = 0;
        for (int i = 0; i < 60; i++) {
            float[] q = TierT.vec(83, i, DIM);
            Hit[] ha = a.search(q, 10, null), hb = b.search(q, 10, null);
            if (ha.length != hb.length) { diff++; continue; }
            for (int j = 0; j < ha.length; j++) if (ha[j].id() != hb[j].id() || Math.abs(ha[j].score() - hb[j].score()) > 1e-6f) diff++;
        }
        TierT.check(diff == 0, "rebuild/exact: rebuilt index returns the same ids and scores on 60 queries (diff=" + diff + ")");
        // (b) graph tier
        TieredIndex g = mk(Metric.COSINE, cfg(300, 200, 4));
        Map<Long, float[]> gm = new HashMap<>();
        fill(g, gm, 84, 0, 2500);
        for (long id = 0; id < 2500; id++) {
            if (id % 9 == 0) {
                g.delete(id);
                gm.remove(id);
            } else if (id % 9 == 1) {
                float[] v = TierT.vec(85, id, DIM);
                g.add(id, v);
                gm.put(id, v);
            }
        }
        TierT.check(g.awaitIdle(120_000), "rebuild/graph: idle");
        List<TieredIndex.IdVector> graw = new ArrayList<>();
        for (Map.Entry<Long, float[]> e : gm.entrySet()) graw.add(new TieredIndex.IdVector(e.getKey(), e.getValue()));
        TieredIndex g2 = mk(Metric.COSINE, cfg(300, 200, 4));
        g2.rebuildFrom(graw);
        TierT.check(g2.size() == gm.size() && g.size() == gm.size(), "rebuild/graph: sizes equal");
        double ov = 0;
        for (int i = 0; i < 80; i++) {
            float[] q = TierT.vec(86, i, DIM);
            ov += TierT.recall(g2.search(q, 10, null), g.search(q, 10, null));
        }
        ov /= 80;
        double r2 = meanRecall(g2, gm, Metric.COSINE, 80, 87, null);
        double r1 = meanRecall(g, gm, Metric.COSINE, 80, 87, null);
        System.out.printf("  rebuild/graph: overlap original vs rebuilt %.4f, recall original %.4f rebuilt %.4f, rebuilt segments %d%n", ov, r1, r2, g2.segmentCount());
        TierT.check(ov >= 0.90, "rebuild/graph: results overlap >= 0.90 (" + ov + ")");
        TierT.check(r2 >= 0.93, "rebuild/graph: rebuilt recall >= 0.93 (" + r2 + ")");
        // in place, then keep using it
        g.rebuildFrom(graw);
        TierT.check(g.size() == gm.size() && g.segmentCount() == 1 && g.stats().floorEntries() == 0, "rebuild/graph: in-place rebuild leaves one segment and no floor");
        float[] nv = TierT.vec(88, 100_000, DIM);
        g.add(100_000, nv);
        TierT.check(TierT.containsId(g.search(nv, 3, null), 100_000) && g.size() == gm.size() + 1, "rebuild/graph: add after rebuild is found");
        TierT.check(g.delete(100_000) && g.size() == gm.size(), "rebuild/graph: delete after rebuild");
        double r3 = meanRecall(g, gm, Metric.COSINE, 80, 87, null);
        TierT.check(r3 >= 0.93, "rebuild/graph: in-place rebuilt recall >= 0.93 (" + r3 + ")");
    }

    static void testClose() {
        TieredIndex ix = new TieredIndex(DIM, Metric.COSINE, cfg(100, 50, 4));
        Map<Long, float[]> model = new HashMap<>();
        fill(ix, model, 91, 0, 500);
        ix.close();
        TierT.check(!ix.workerAlive(), "close: worker thread stopped");
        ix.close();
        TierT.check(!ix.workerAlive(), "close: second close is harmless");
    }

    // ------------------------------------------------------------------------------------------------------------------


    /**
     * Agent-memory rate: one agent, one turn = 1 insert then 4 queries (the access pattern of an agent memory), the tail cap far from reached, seals and
     * merges running in the background. Query latency per phase (quiet, during seal, during merge) with the sample counts.
     */
    static void testAgentRate() {
        final int dim = 128, preload = 1500;
        final int turns = full ? 20_000 : 4000;
        final TieredIndex.Config c = new TieredIndex.Config(400, 0L, 200, 3, 4, 0.3, 20_000, 4, Quant.NONE,
                new GraphIndex.Params(16, 100, 100, 1.0f, false, 7L), 2);
        final TieredIndex ix = new TieredIndex(dim, Metric.COSINE, c);
        open.add(ix);
        for (long id = 0; id < preload; id++) ix.add(id, TierT.vec(101, id, dim));
        TierT.check(ix.awaitIdle(120_000), "agent: idle after preload");
        final int cap = turns * 5 * 4;
        long[][] ql = new long[3][cap];
        long[][] al = new long[3][turns * 5];
        int[] qc = new int[3], ac = new int[3];
        SplittableRandom r = new SplittableRandom(5);
        int miss = 0, t = 0;
        long start = System.nanoTime();
        while (t < turns || ((qc[0] < 100 || qc[1] < 100 || qc[2] < 100) && t < turns * 5)) {
            long id = preload + t;
            float[] v = TierT.vec(101, id, dim);
            int p0 = ix.activity();
            long t0 = System.nanoTime();
            ix.add(id, v);
            long da = System.nanoTime() - t0;
            int pa = p0 != 0 ? p0 : ix.activity();
            al[pa][ac[pa]++] = da;
            for (int j = 0; j < 4; j++) {
                float[] q = j == 0 ? v : TierT.vec(102, r.nextInt(1_000_000), dim);
                int a0 = ix.activity();
                long s0 = System.nanoTime();
                Hit[] h = ix.search(q, 10, null);
                long dt = System.nanoTime() - s0;
                int ph = a0 != 0 ? a0 : ix.activity();
                ql[ph][qc[ph]++] = dt;
                if (j == 0 && !TierT.containsId(h, id)) miss++;
            }
            t++;
        }
        double secs = (System.nanoTime() - start) / 1e9;
        TierT.check(ix.awaitIdle(120_000), "agent: idle at the end");
        TieredIndex.Stats s = ix.stats();
        System.out.printf("  agent: %d turns in %.1f s, dim %d, tailLimit 400, maxSegments 3, buildThreads 2%n", t, secs, dim);
        System.out.println("  agent: " + s);
        System.out.printf("  agent: seal %.0f ms per 400 vectors, merge %.0f ms per 1000 vectors%n",
                s.sealMillis() * 400.0 / Math.max(1, s.sealedVectors()), s.mergeMillis() * 1000.0 / Math.max(1, s.mergedVectors()));
        String[] names = {"quiet", "during seal", "during merge"};
        for (int ph = 0; ph < 3; ph++) {
            System.out.printf("  agent: query latency %-13s n=%d p50=%d us p99=%d us max=%d us | insert n=%d p50=%d us p99=%d us max=%d us%n", names[ph], qc[ph],
                    TierT.pct(ql[ph], qc[ph], 0.50) / 1000, TierT.pct(ql[ph], qc[ph], 0.99) / 1000, TierT.pct(ql[ph], qc[ph], 1.0) / 1000,
                    ac[ph], TierT.pct(al[ph], ac[ph], 0.50) / 1000, TierT.pct(al[ph], ac[ph], 0.99) / 1000, TierT.pct(al[ph], ac[ph], 1.0) / 1000);
            TierT.check(qc[ph] >= 30, "agent: phase '" + names[ph] + "' has at least 30 query samples (" + qc[ph] + ")");
        }
        TierT.check(miss == 0, "agent: the vector inserted this turn is found by the first query (miss=" + miss + ")");
        TierT.check(s.backpressureWaits() == 0, "agent: the hard tail cap was never reached (waits=" + s.backpressureWaits() + ")");
        TierT.check(s.failures() == 0 && ix.size() == preload + t, "agent: no failure, size correct");
    }

    /** 8 writers and 8 readers with seals and merges running; per-phase reader latency */
    static void testConcurrent(int seconds) {
        final int dim = 48, writers = 8, readers = 8, cap = 300_000;
        final TieredIndex.Config c = new TieredIndex.Config(1000, 0L, 500, 4, 4, 0.2, 40_000, 4, Quant.NONE,
                new GraphIndex.Params(16, 100, 150, 1.0f, false, 7L), 3);
        final TieredIndex ix = new TieredIndex(dim, Metric.COSINE, c);
        open.add(ix);
        final AtomicBoolean stop = new AtomicBoolean();
        final AtomicInteger errors = new AtomicInteger(), phantoms = new AtomicInteger(), badOrder = new AtomicInteger();
        final AtomicInteger rywMiss = new AtomicInteger(), wrongDelete = new AtomicInteger();
        final AtomicLong writes = new AtomicLong(), reads = new AtomicLong();
        final AtomicLongArray inserted = new AtomicLongArray(writers);
        final List<HashMap<Long, float[]>> models = new ArrayList<>();
        final List<HashSet<Long>> deletedSets = new ArrayList<>();
        for (int w = 0; w < writers; w++) {
            models.add(new HashMap<>());
            deletedSets.add(new HashSet<>());
        }
        final long[][][] lat = new long[readers][3][cap];
        final int[][] cnt = new int[readers][3];
        final List<Thread> threads = new ArrayList<>();
        final AtomicReference1 firstError = new AtomicReference1();
        for (int w = 0; w < writers; w++) {
            final int wi = w;
            threads.add(new Thread(() -> {
                try {
                    SplittableRandom r = new SplittableRandom(1000 + wi);
                    HashMap<Long, float[]> m = models.get(wi);
                    HashSet<Long> dels = deletedSets.get(wi);
                    long base = (long) wi << 32;
                    long seq = 0;
                    ArrayList<Long> alive = new ArrayList<>();
                    while (!stop.get()) {
                        int op = r.nextInt(10);
                        if (op < 8 || alive.isEmpty()) {
                            long id = base + seq;
                            seq++;
                            inserted.set(wi, seq); // before add: a reader can see the id as soon as add returns
                            float[] v = TierT.vec(wi + 10, id, dim);
                            ix.add(id, v);
                            m.put(id, v);
                            alive.add(id);
                            if ((seq & 7) == 0 && !TierT.containsId(ix.search(v, 5, null), id)) rywMiss.incrementAndGet();
                        } else if (op == 8) {
                            long id = alive.get(r.nextInt(alive.size()));
                            float[] v = TierT.vec(wi + 100_000 + (seq & 0xffff), id & 0xffffffffL, dim);
                            ix.add(id, v);
                            m.put(id, v);
                        } else {
                            int j = r.nextInt(alive.size());
                            long id = alive.get(j);
                            alive.set(j, alive.get(alive.size() - 1));
                            alive.remove(alive.size() - 1);
                            if (!ix.delete(id)) wrongDelete.incrementAndGet();
                            m.remove(id);
                            dels.add(id);
                        }
                        writes.incrementAndGet();
                    }
                } catch (Throwable t) {
                    errors.incrementAndGet();
                    firstError.set(t);
                }
            }, "writer-" + w));
        }
        for (int rd = 0; rd < readers; rd++) {
            final int ri = rd;
            threads.add(new Thread(() -> {
                try {
                    SplittableRandom r = new SplittableRandom(5000 + ri);
                    while (!stop.get()) {
                        float[] q = TierT.vec(777 + ri, r.nextLong() & 0xffffffL, dim);
                        int ph = ix.activity();
                        long t0 = System.nanoTime();
                        Hit[] h = ix.search(q, 10, null);
                        long dt = System.nanoTime() - t0;
                        if (cnt[ri][ph] < cap) lat[ri][ph][cnt[ri][ph]++] = dt;
                        reads.incrementAndGet();
                        if (!TierT.sortedDesc(h) || !TierT.distinct(h)) badOrder.incrementAndGet();
                        for (Hit x : h) {
                            long w = x.id() >>> 32;
                            long s = x.id() & 0xffffffffL;
                            if (w >= writers || s >= inserted.get((int) w) || Float.isNaN(x.score())) phantoms.incrementAndGet();
                        }
                    }
                } catch (Throwable t) {
                    errors.incrementAndGet();
                    firstError.set(t);
                }
            }, "reader-" + rd));
        }
        long t0 = System.nanoTime();
        for (Thread t : threads) t.start();
        try {
            Thread.sleep(seconds * 1000L);
            stop.set(true);
            for (Thread t : threads) t.join();
        } catch (InterruptedException e) {
            throw new IllegalStateException(e);
        }
        double secs = (System.nanoTime() - t0) / 1e9;
        TierT.check(ix.awaitIdle(180_000), "conc: idle after the run");
        TieredIndex.Stats s = ix.stats();
        System.out.printf("  conc: %.1f s, %d writes (%.0f/s), %d reads (%.0f/s)%n", secs, writes.get(), writes.get() / secs, reads.get(), reads.get() / secs);
        System.out.println("  conc: " + s);
        System.out.printf("  conc: seal %.0f ms per 2000 vectors (buildThreads %d), merge %.0f ms per 1000 vectors%n",
                s.sealMillis() * 2000.0 / Math.max(1, s.sealedVectors()), c.buildThreads(), s.mergeMillis() * 1000.0 / Math.max(1, s.mergedVectors()));
        if (firstError.get() != null) firstError.get().printStackTrace(System.out);
        TierT.check(errors.get() == 0 && s.failures() == 0 && ix.lastError() == null, "conc: zero exceptions in threads and in the worker (errors=" + errors.get() + ")");
        TierT.check(phantoms.get() == 0, "conc: no phantom id returned (" + phantoms.get() + ")");
        TierT.check(badOrder.get() == 0, "conc: every result sorted and distinct (" + badOrder.get() + ")");
        TierT.check(rywMiss.get() == 0, "conc: writers always find the vector they just added (miss=" + rywMiss.get() + ")");
        TierT.check(wrongDelete.get() == 0, "conc: delete of an own live id always returns true (" + wrongDelete.get() + ")");
        int expect = 0;
        for (HashMap<Long, float[]> m : models) expect += m.size();
        TierT.check(ix.size() == expect, "conc: size equals the model (" + ix.size() + " vs " + expect + ")");
        TierT.check(s.seals() >= 1 && s.merges() >= 1, "conc: seals and merges both ran (" + s.seals() + ", " + s.merges() + ")");
        // every writer finds its last 100 inserts
        int missLast = 0, lastChecked = 0;
        for (int w = 0; w < writers; w++) {
            List<Long> keys = new ArrayList<>(models.get(w).keySet());
            keys.sort(null);
            for (int i = Math.max(0, keys.size() - 100); i < keys.size(); i++) {
                long id = keys.get(i);
                lastChecked++;
                if (!TierT.containsId(ix.search(models.get(w).get(id), 10, null), id)) missLast++;
            }
        }
        TierT.check(missLast == 0 && lastChecked >= 100 * writers, "conc: every writer finds its last 100 inserts (miss=" + missLast + " of " + lastChecked + ")");
        // sampled membership: current vector found, deleted ids not found, newest score
        int missSample = 0, staleSample = 0, delSeen = 0, sampled = 0;
        SplittableRandom r = new SplittableRandom(99);
        for (int w = 0; w < writers; w++) {
            List<Long> keys = new ArrayList<>(models.get(w).keySet());
            for (int i = 0; i < 250 && !keys.isEmpty(); i++) {
                long id = keys.get(r.nextInt(keys.size()));
                float[] cur = models.get(w).get(id);
                Hit[] h = ix.search(cur, 10, null);
                sampled++;
                if (!TierT.containsId(h, id)) missSample++;
                else for (Hit x : h) if (x.id() == id && x.score() < 0.999f) staleSample++;
            }
            List<Long> dl = new ArrayList<>(deletedSets.get(w));
            for (int i = 0; i < 150 && !dl.isEmpty(); i++) {
                long id = dl.get(r.nextInt(dl.size()));
                Hit[] h = ix.search(TierT.vec(w + 10, id, dim), 10, null);
                if (TierT.containsId(h, id)) delSeen++;
            }
        }
        TierT.check(missSample == 0 && staleSample == 0, "conc: sampled ids found with the newest vector (miss=" + missSample + ", stale=" + staleSample + " of " + sampled + ")");
        TierT.check(delSeen == 0, "conc: sampled deleted ids never returned (" + delSeen + ")");
        // per-phase reader latency
        String[] names = {"no seal/merge", "during seal", "during merge"};
        for (int ph = 0; ph < 3; ph++) {
            int total = 0;
            for (int ri = 0; ri < readers; ri++) total += cnt[ri][ph];
            long[] all = new long[total];
            int k = 0;
            for (int ri = 0; ri < readers; ri++) {
                System.arraycopy(lat[ri][ph], 0, all, k, cnt[ri][ph]);
                k += cnt[ri][ph];
            }
            System.out.printf("  conc: reader latency %-14s n=%d p50=%d us p99=%d us max=%d us%n", names[ph], total,
                    TierT.pct(all, total, 0.50) / 1000, TierT.pct(all, total, 0.99) / 1000, TierT.pct(all, total, 1.0) / 1000);
        }
    }

    /** a tiny holder (java.util.concurrent.atomic.AtomicReference would do; kept local to keep the imports short) */
    static final class AtomicReference1 {
        private volatile Throwable t;
        void set(Throwable x) { if (t == null) t = x; }
        Throwable get() { return t; }
    }

    // ------------------------------------------------------------------------------------------------------------------

    public static void main(String[] args) {
        full = args.length > 0 && args[0].equals("full");
        String only = args.length > 1 ? args[1] : "";
        System.out.println("TierTests mode=" + (full ? "full" : "quick") + " filter='" + only + "' java=" + System.getProperty("java.version"));
        String[] names = {"tailOnly", "noGraphBelowSmall", "segmentsRecall", "filterScanBelow", "metrics", "swapInstantHooks", "sealStress", "merge", "deleteReplace", "rebuild", "close", "agentRate", "concurrent"};
        Runnable[] tests = {TierTests::testTailOnly, TierTests::testNoGraphBelowSmall, TierTests::testSegmentsRecall, TierTests::testFilterScanBelow, TierTests::testMetrics,
                TierTests::testSwapInstantHooks, TierTests::testSealStress, TierTests::testMerge, TierTests::testDeleteReplace,
                TierTests::testRebuild, TierTests::testClose, TierTests::testAgentRate, () -> testConcurrent(full ? 60 : 10)};
        for (int i = 0; i < tests.length; i++) {
            if (!only.isEmpty() && !names[i].contains(only)) continue;
            long t0 = System.nanoTime();
            int f0 = TierT.fail, p0 = TierT.pass;
            System.out.println("TEST " + names[i]);
            try {
                tests[i].run();
            } catch (Throwable t) {
                TierT.fail++;
                System.out.println("  EXCEPTION " + t);
                t.printStackTrace(System.out);
            }
            System.out.printf("%s %s (%d checks, %d failed, %.1f s)%n", TierT.fail == f0 ? "PASS" : "FAIL", names[i], TierT.pass - p0 + TierT.fail - f0, TierT.fail - f0, (System.nanoTime() - t0) / 1e9);
            for (TieredIndex ix : open) ix.close();
            open.clear();
        }
        TierT.done("TierTests");
    }
}
