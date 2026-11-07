package com.humemai.vec;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLongArray;
import java.util.function.LongPredicate;

/**
 * A model-based probe for record-id reuse on a {@link TieredIndex} (issue #5). A host database frees a record id and later hands it to a new record, so the tier
 * sees {@code delete(id)} followed by {@code add(id, newVector)} with the same id. The probe drives an index with that pattern in the places where a stale vector
 * or a lost delete could hide: the old copy in the active tail, in the frozen tail while a seal publishes, in a sealed segment, in a segment that a merge is
 * rebuilding, after the merge, after the version floor moved several times, and under the background worker. Every answer is compared with a model that holds
 * the current vector of every id and keeps the older vectors, so a resurrected vector is recognised by its score. Synthetic data, fixed seeds, no timing claims:
 * only counts and exact score checks.
 *
 * <p>The class is not a JUnit test. {@link RidReuseTest} and the other wrappers call {@link #runOneMap}. It takes and returns JDK types only, so that the same
 * class can be loaded by a second class loader against a modified copy of {@code TieredIndex} (see {@link RidReuseRig}).
 *
 * <p>A run passes when these counters are exactly 0: {@code lostDeletes} (a deleted id was returned), {@code staleHits} (a live reused id was returned with an
 * older vector's score), {@code wrongScore}, {@code oldRank1} (a reused id at rank 1 for a query equal to one of its old vectors), {@code dups} (an id twice in
 * one result), {@code unsorted}, {@code phantoms} (an id never added), {@code sizeMismatch} ({@code size()} or {@code stats().live()} differs from the model),
 * {@code badDeleteReturn} ({@code delete} returned something other than "the id was live"), {@code floorLeft} (the floor map was not empty after a full merge in
 * a deterministic case), and the coverage, worker and timeout counters. A case that did not reach its window counts as {@code coverageFail}: a run with a void
 * case fails, it is never skipped. The self-query miss rates are ANN allowances and must stay within the bound given in {@link #rateBad}.
 */
final class RidReuseProbe {
    static final boolean VERBOSE = Boolean.getBoolean("ridreuse.verbose");
    static final int K = 10;
    static final int EF = 200;
    static final int EFT = 400; // the allow-one search: the graph budget then covers a segment of 20000 rows, so it ends in an exact scan

    record Cfg(String name, int n, int dim, Metric metric, int sample) {}

    static final Cfg[] ALL = {
        new Cfg("n3000d32", 3000, 32, Metric.L2, 1200),
        new Cfg("n3000d128", 3000, 128, Metric.COSINE, 1200),
        new Cfg("n3000d32c", 3000, 32, Metric.COSINE, 1200),
        new Cfg("n20000d32", 20000, 32, Metric.COSINE, 800),
        new Cfg("n20000d128", 20000, 128, Metric.L2, 800),
    };

    /** every case, in the order a run executes them */
    static final List<String> CASES = List.of("a_tail", "a2_tail_and_segments", "b_sealed", "c1_reuse_in_merge", "c2_delete_in_merge", "e_floor_chain", "s_seal_window",
            "f_fuzz_background");

    /** the cases without the background fuzz: all of them are deterministic in their sequencing */
    static final List<String> FAST_CASES = CASES.subList(0, CASES.size() - 1);

    static final List<String> FUZZ_ONLY = List.of("f_fuzz_background");

    /** operation groups that run inside a running merge: c1 has 2, c2 has 5 */
    static int expectedWindowGroups(List<String> cases) {
        return (cases.contains("c1_reuse_in_merge") ? 2 : 0) + (cases.contains("c2_delete_in_merge") ? 5 : 0);
    }

    static Cfg cfgByName(String name) {
        for (Cfg c : ALL) if (c.name().equals(name)) return c;
        throw new IllegalArgumentException("unknown config " + name);
    }

    /** where the class under test was loaded from (the rig checks that a mutant really was loaded) */
    static String tierClassLocation() {
        return String.valueOf(TieredIndex.class.getProtectionDomain().getCodeSource().getLocation());
    }

    // ------------------------------------------------------------------------------------------------------------------
    // counters

    static final class Cnt {
        long searches, hitsChecked, idsChecked;
        long lostDeletes, staleHits, wrongScore, dups, unsorted, phantoms, oldRank1, sizeMismatch, badDeleteReturn, floorLeft;
        long tierFailures, coverageFail, timeouts, readerErrors;
        long floorNonEmpty; // informational: the background case, where a floor entry may legitimately outlive the last merge
        final long[] selfQ = new long[2], selfMiss1 = new long[2], selfMissK = new long[2], exQ = new long[2], exMiss = new long[2]; // index 1 = touched live ids (reused or replaced), 0 = untouched
        long cSelfQ, cSelfMiss1;                    // the concurrent reader's self queries
        long opsTotal, opsInSeal, opsInMerge, windowGroups, windowGroupsConfirmed;
        final List<String> notes = new ArrayList<>();  // the first wrong answers
        final List<String> events = new ArrayList<>(); // void windows, exceptions and worker errors, kept apart so that a flood of wrong answers cannot push them out
        String where = "";

        void note(String s) {
            if (notes.size() < 25) notes.add(where + ": " + s);
        }

        void event(String s) {
            if (events.size() < 25) events.add(where + ": " + s);
        }

        long exact() {
            return lostDeletes + staleHits + wrongScore + dups + unsorted + phantoms + oldRank1 + sizeMismatch + badDeleteReturn;
        }

        long hard() {
            return exact() + floorLeft + tierFailures + coverageFail + timeouts + readerErrors;
        }

        void add(Cnt o) {
            searches += o.searches; hitsChecked += o.hitsChecked; idsChecked += o.idsChecked;
            lostDeletes += o.lostDeletes; staleHits += o.staleHits; wrongScore += o.wrongScore; dups += o.dups; unsorted += o.unsorted; phantoms += o.phantoms;
            oldRank1 += o.oldRank1; sizeMismatch += o.sizeMismatch; badDeleteReturn += o.badDeleteReturn; floorLeft += o.floorLeft;
            tierFailures += o.tierFailures; coverageFail += o.coverageFail; timeouts += o.timeouts; readerErrors += o.readerErrors; floorNonEmpty += o.floorNonEmpty;
            for (int i = 0; i < 2; i++) {
                selfQ[i] += o.selfQ[i]; selfMiss1[i] += o.selfMiss1[i]; selfMissK[i] += o.selfMissK[i]; exQ[i] += o.exQ[i]; exMiss[i] += o.exMiss[i];
            }
            cSelfQ += o.cSelfQ; cSelfMiss1 += o.cSelfMiss1;
            opsTotal += o.opsTotal; opsInSeal += o.opsInSeal; opsInMerge += o.opsInMerge; windowGroups += o.windowGroups; windowGroupsConfirmed += o.windowGroupsConfirmed;
            for (String s : o.notes) if (notes.size() < 25) notes.add(s);
            for (String s : o.events) if (events.size() < 25) events.add(s);
        }
    }

    // ------------------------------------------------------------------------------------------------------------------
    // model: the truth the index is compared with

    static final class Model {
        final int cap, dim;
        final Metric metric;
        final float[][][] hist; // hist[id] = every vector ever added under id, oldest first (copy on write: the concurrent reader may read the reference)
        final int[] cur;        // index into hist[id] of the live vector, -1 when the id is not live
        final boolean[] touched; // deleted or replaced at least once
        int live;

        Model(int cap, int dim, Metric metric) {
            this.cap = cap;
            this.dim = dim;
            this.metric = metric;
            hist = new float[cap][][];
            cur = new int[cap];
            Arrays.fill(cur, -1);
            touched = new boolean[cap];
        }

        void put(int id, float[] v) {
            float[][] h = hist[id];
            int n = h == null ? 0 : h.length;
            float[][] nh = h == null ? new float[1][] : Arrays.copyOf(h, n + 1);
            nh[n] = v;
            hist[id] = nh;
            if (n > 0) touched[id] = true;
            if (cur[id] < 0) live++;
            cur[id] = n;
        }

        boolean kill(int id) {
            if (cur[id] < 0) return false;
            cur[id] = -1;
            touched[id] = true;
            live--;
            return true;
        }
    }

    /** deterministic unit vector by (seed, id, generation) */
    static float[] vec(long seed, long id, int gen, int dim) {
        SplittableRandom r = new SplittableRandom(seed * 0x9E3779B97F4A7C15L + id * 0xD1B54A32D192ED03L + gen * 0x2545F4914F6CDD1DL + 17L);
        float[] v = new float[dim];
        double nn = 0;
        for (int i = 0; i < dim; i++) {
            v[i] = (float) r.nextGaussian();
            nn += (double) v[i] * v[i];
        }
        float inv = (float) (1.0 / Math.sqrt(nn));
        for (int i = 0; i < dim; i++) v[i] *= inv;
        return v;
    }

    /** higher is better for every metric, the convention of Hit.score */
    static float score(Metric m, float[] q, float[] v) {
        double dot = 0, nq = 0, nv = 0, l2 = 0;
        for (int i = 0; i < q.length; i++) {
            dot += (double) q[i] * v[i];
            nq += (double) q[i] * q[i];
            nv += (double) v[i] * v[i];
            double d = (double) q[i] - v[i];
            l2 += d * d;
        }
        if (m == Metric.L2) return (float) -l2;
        if (m == Metric.DOT) return (float) dot;
        return (float) (dot / Math.sqrt(nq * nv));
    }

    static boolean close(float got, float want) {
        return Math.abs(got - want) <= 1e-3f * (1f + Math.abs(want));
    }

    static boolean has(Hit[] r, long id) {
        for (Hit h : r) if (h.id() == id) return true;
        return false;
    }

    /**
     * Verify one result against the model. before == null: quiet index, every id counts. Otherwise (concurrent reader): before[id] is the id's sequence number read before the search; a hit counts only when
     * that number was even and is still the same after the model entry was read (the id was untouched for the whole search).
     */
    static boolean verifyHits(Cnt c, Model m, float[] q, Hit[] h, long[] before, AtomicLongArray st) {
        boolean ok = true;
        c.searches++;
        float prev = Float.POSITIVE_INFINITY;
        HashSet<Long> seen = new HashSet<>();
        for (Hit hit : h) {
            if (!(hit.score() <= prev)) { c.unsorted++; c.note("scores not sorted at id " + hit.id()); ok = false; }
            prev = hit.score();
            if (!seen.add(hit.id())) { c.dups++; c.note("id " + hit.id() + " twice in one result"); ok = false; }
            long idl = hit.id();
            if (idl < 0 || idl >= m.cap) { c.phantoms++; c.note("id " + idl + " is outside the id range"); ok = false; continue; }
            int id = (int) idl;
            long a = 0;
            if (before != null) {
                a = before[id];
                if ((a & 1L) != 0) continue;
            }
            int cu = m.cur[id];
            float[][] hl = m.hist[id];
            if (before != null && st.get(id) != a) continue;
            c.hitsChecked++;
            if (hl == null) { c.phantoms++; c.note("id " + id + " was never added"); ok = false; continue; }
            if (cu < 0) { c.lostDeletes++; c.note("deleted id " + id + " returned with score " + hit.score()); ok = false; continue; }
            float want = score(m.metric, q, hl[cu]);
            if (close(hit.score(), want)) continue;
            boolean stale = false;
            for (int g = 0; g < hl.length; g++) {
                if (g != cu && close(hit.score(), score(m.metric, q, hl[g]))) { stale = true; break; }
            }
            if (stale) {
                c.staleHits++;
                c.note("id " + id + " returned with the score of an older vector (" + hit.score() + ", current vector would score " + want + ")");
            } else {
                c.wrongScore++;
                c.note("id " + id + " returned with score " + hit.score() + ", current vector scores " + want);
            }
            ok = false;
        }
        return ok;
    }

    static void sleepMs(long ms) {
        try {
            Thread.sleep(ms);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    static int[] slice(int[] a, int from, int to) {
        return Arrays.copyOfRange(a, Math.min(from, a.length), Math.min(to, a.length));
    }

    static int[] range(int from, int to) {
        int[] r = new int[Math.max(0, to - from)];
        for (int i = 0; i < r.length; i++) r[i] = from + i;
        return r;
    }

    // ------------------------------------------------------------------------------------------------------------------
    // one index with its model

    static TieredIndex.Config tcfg(int tailLimit, int smallLimit, int maxSeg, int fan, double dead, int rate) {
        TieredIndex.Config d = TieredIndex.Config.defaults();
        return new TieredIndex.Config(tailLimit, 0L, smallLimit, maxSeg, fan, dead, rate, 4, d.tailQuant(), d.graph(), 2);
    }

    /** only sealNow and mergeNow seal and merge: no tail limit, no age seal, no segment cap, no dead-fraction compaction */
    static TieredIndex.Config detCfg(int mergeRate) {
        return tcfg(1 << 28, 1000, 1000, 4, 1.01, mergeRate);
    }

    static final class Ctx {
        final Cfg cfg;
        final String name;
        final long seed;
        final Model m;
        final TieredIndex ix;
        final Cnt c;
        final AtomicLongArray st;
        final SplittableRandom rng;
        final boolean reader; // a concurrent reader thread in the merge cases and the background case

        Ctx(Cfg cfg, String name, long seed, TieredIndex.Config tc, Cnt c, boolean reader) {
            this.reader = reader;
            this.cfg = cfg;
            this.name = name;
            this.seed = seed;
            this.c = c;
            int cap = cfg.n + cfg.n / 10 + 16;
            this.m = new Model(cap, cfg.dim, cfg.metric);
            this.st = new AtomicLongArray(cap);
            this.rng = new SplittableRandom(seed * 7919L + name.hashCode());
            this.ix = new TieredIndex(cfg.dim, cfg.metric, tc);
        }

        void log(String s) {
            System.out.println("[" + cfg.name + " seed=" + seed + " " + name + "] " + s);
        }

        /** a message that also goes into the result, for the assertion message of the test */
        void note(String s) {
            log(s);
            c.event(s);
        }

        // ---- writer operations: model first, then the index, with the id's sequence number odd in between ----

        void add(int id) {
            int g = m.hist[id] == null ? 0 : m.hist[id].length;
            float[] v = vec(seed, id, g, m.dim);
            st.incrementAndGet(id);
            m.put(id, v);
            ix.add(id, v);
            st.incrementAndGet(id);
            c.opsTotal++;
        }

        void del(int id) {
            st.incrementAndGet(id);
            boolean existed = m.kill(id);
            boolean r = ix.delete(id);
            st.incrementAndGet(id);
            if (r != existed) {
                c.badDeleteReturn++;
                c.note("delete(" + id + ") returned " + r + ", the model says " + existed);
            }
            c.opsTotal++;
        }

        void addAll(int[] ids) {
            for (int id : ids) add(id);
        }

        void delAll(int[] ids) {
            for (int id : ids) del(id);
        }

        void fill(int from, int to) {
            for (int id = from; id < to; id++) add(id);
        }

        /** all live ids in random order */
        int[] liveShuffled() {
            int[] buf = new int[m.cap];
            int k = 0;
            for (int id = 0; id < m.cap; id++) if (m.cur[id] >= 0) buf[k++] = id;
            for (int i = k - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                int t = buf[i];
                buf[i] = buf[j];
                buf[j] = t;
            }
            return Arrays.copyOf(buf, k);
        }

        /** null when the case runs without a reader */
        Reader startReader(long readerSeed) {
            if (!reader) return null;
            Reader rd = new Reader(this, readerSeed);
            rd.start();
            return rd;
        }

        void quiet() {
            if (!ix.awaitIdle(120_000)) {
                c.timeouts++;
                note("awaitIdle timed out");
            }
        }

        void precond(String what, boolean ok) {
            if (!ok) {
                c.coverageFail++;
                note("VOID: precondition not met: " + what);
            }
        }

        /** after a full mergeNow with at least two segments and no seal running, no stale copy is left and the floor map must be empty (strict); the background case only reports */
        void floorCheck(String label, boolean strict) {
            int fe = ix.stats().floorEntries();
            if (fe == 0) return;
            if (strict) {
                c.floorLeft++;
                note("FLOOR " + label + ": the floor map holds " + fe + " entries after a full merge with no seal running");
            } else {
                c.floorNonEmpty++;
                log("FLOOR " + label + ": the floor map holds " + fe + " entries after a full merge (informational)");
            }
        }

        // ---- checks ----

        /** category 2 = dead ids, 1 = touched live ids, 0 = untouched live ids; a random sample of at most max */
        int[] sampleOf(int cat, int max) {
            int[] buf = new int[m.cap];
            int k = 0;
            for (int id = 0; id < m.cap; id++) {
                if (m.hist[id] == null) continue;
                boolean live = m.cur[id] >= 0;
                boolean ok = cat == 2 ? !live : cat == 1 ? (live && m.touched[id]) : (live && !m.touched[id]);
                if (ok) buf[k++] = id;
            }
            int take = Math.min(k, max);
            for (int i = 0; i < take; i++) {
                int j = i + rng.nextInt(k - i);
                int t = buf[i];
                buf[i] = buf[j];
                buf[j] = t;
            }
            return Arrays.copyOf(buf, take);
        }

        void checkId(int id) {
            float[][] h = m.hist[id];
            int cu = m.cur[id];
            boolean live = cu >= 0;
            int grp = m.touched[id] ? 1 : 0;
            final long tid = id;
            LongPredicate only = z -> z == tid;
            float[] q = live ? h[cu] : h[h.length - 1];
            c.idsChecked++;
            Hit[] r = ix.search(q, K, null, EF);
            verifyHits(c, m, q, r, null, null);
            Hit[] t = ix.search(q, 1, only, EFT); // the id alone is allowed: a live id must come back, a dead id must still not
            verifyHits(c, m, q, t, null, null);
            // sealed-only: the new vector may be absent (it may sit in the tail), but a deleted id or an older vector must never appear
            verifyHits(c, m, q, ix.search(q, K, null, TieredIndex.Freshness.SEALED_ONLY).hits(), null, null);
            verifyHits(c, m, q, ix.search(q, 1, only, TieredIndex.Freshness.SEALED_ONLY).hits(), null, null);
            if (live) {
                c.selfQ[grp]++;
                if (r.length == 0 || r[0].id() != id) c.selfMiss1[grp]++;
                if (!has(r, id)) c.selfMissK[grp]++;
                c.exQ[grp]++;
                if (t.length == 0 || t[0].id() != id) c.exMiss[grp]++;
            }
            if (h.length >= 2) { // an OLDER vector of the same id as the query: the id must not come back with that vector
                int og = rng.nextInt(h.length);
                if (og == cu) og = (og + 1) % h.length;
                float[] qo = h[og];
                Hit[] ro = ix.search(qo, K, null, EF);
                verifyHits(c, m, qo, ro, null, null);
                if (ro.length > 0 && ro[0].id() == id && RidReuseProbe.close(ro[0].score(), score(m.metric, qo, qo))) {
                    c.oldRank1++;
                    c.note("id " + id + " is at rank 1 with the score of its old vector (generation " + og + ", current " + cu + ")");
                }
                Hit[] to = ix.search(qo, 1, only, EFT);
                verifyHits(c, m, qo, to, null, null);
                Hit[] so = ix.search(qo, K, null, TieredIndex.Freshness.SEALED_ONLY).hits();
                verifyHits(c, m, qo, so, null, null);
                if (so.length > 0 && so[0].id() == id && RidReuseProbe.close(so[0].score(), score(m.metric, qo, qo))) {
                    c.oldRank1++;
                    c.note("id " + id + " is at rank 1 of a sealed-only search with the score of its old vector (generation " + og + ", current " + cu + ")");
                }
            }
        }

        void checkpoint(String label) {
            long h0 = c.hard();
            int sz = ix.size();
            int lv = ix.stats().live();
            if (sz != m.live || lv != m.live) {
                c.sizeMismatch++;
                note(label + ": size() " + sz + " and stats().live() " + lv + " must both equal the model " + m.live);
            }
            int[] dead = sampleOf(2, cfg.sample / 4);
            int[] tl = sampleOf(1, cfg.sample / 2);
            int[] pr = sampleOf(0, cfg.sample / 4);
            for (int id : dead) checkId(id);
            for (int id : tl) checkId(id);
            for (int id : pr) checkId(id);
            TieredIndex.Stats s = ix.stats();
            if (VERBOSE || c.hard() != h0) {
                log(String.format("CK %s | live=%d segs=%d tail=%d floor=%d checked dead/touched/untouched=%d/%d/%d | hard+%d", label, m.live, s.segments(), s.tailSize(), s.floorEntries(),
                        dead.length, tl.length, pr.length, c.hard() - h0));
            }
        }

        void close() {
            try {
                Throwable le = ix.lastError();
                TieredIndex.Stats s = ix.stats();
                if (le != null || s.failures() > 0) {
                    c.tierFailures++;
                    note("tier worker error: failures=" + s.failures() + " lastError=" + le);
                }
            } finally {
                ix.close();
            }
        }
    }

    // ------------------------------------------------------------------------------------------------------------------
    // the concurrent reader and the merge window

    static final class Reader extends Thread {
        final Ctx x;
        final Cnt rc = new Cnt();
        final AtomicBoolean stop = new AtomicBoolean();
        final SplittableRandom r;
        final long[] before;
        volatile Throwable err;

        Reader(Ctx x, long seed) {
            super("ridreuse-reader");
            setDaemon(true);
            this.x = x;
            this.r = new SplittableRandom(seed);
            this.before = new long[x.m.cap];
            this.rc.where = x.c.where;
        }

        @Override
        public void run() {
            try {
                Model m = x.m;
                AtomicLongArray st = x.st;
                while (!stop.get()) {
                    for (int i = 0; i < before.length; i++) before[i] = st.get(i);
                    int id = r.nextInt(m.cap);
                    long a = before[id];
                    if ((a & 1L) != 0) continue;
                    float[][] hl = m.hist[id];
                    int cu = m.cur[id];
                    if (hl == null) continue;
                    if (st.get(id) != a) continue;
                    boolean useOld = hl.length >= 2 && r.nextInt(2) == 0;
                    int g;
                    if (useOld) {
                        g = r.nextInt(hl.length);
                        if (g == cu) g = (g + 1) % hl.length;
                    } else {
                        g = cu >= 0 ? cu : hl.length - 1;
                    }
                    boolean current = !useOld && cu >= 0;
                    boolean sealedOnly = r.nextInt(4) == 0;
                    float[] q = hl[g];
                    Hit[] h = sealedOnly ? x.ix.search(q, K, null, TieredIndex.Freshness.SEALED_ONLY).hits() : x.ix.search(q, K, null, EF);
                    verifyHits(rc, m, q, h, before, st);
                    if (st.get(id) == a) {
                        if (current && !sealedOnly) {
                            rc.cSelfQ++;
                            if (h.length == 0 || h[0].id() != id) rc.cSelfMiss1++;
                        }
                        if (useOld && h.length > 0 && h[0].id() == id && RidReuseProbe.close(h[0].score(), score(m.metric, q, q))) {
                            rc.oldRank1++;
                            rc.note("reader: id " + id + " at rank 1 with the score of its old vector");
                        }
                    }
                }
            } catch (Throwable t) {
                err = t;
            }
        }

        /** stop, join and fold the reader's counters into the case's */
        void finish() {
            stop.set(true);
            try {
                join(60_000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            if (isAlive()) {
                x.c.timeouts++;
                x.note("reader did not stop within 60 s");
            }
            if (err != null) {
                x.c.readerErrors++;
                x.note("reader failed: " + err);
            }
            x.c.add(rc);
        }
    }

    /** mergeNow on a helper thread, and the means to run writer operations inside its build phase */
    static final class Mrg {
        final Ctx x;
        final Throwable[] err = new Throwable[1];
        Thread t;
        long m0;

        Mrg(Ctx x) {
            this.x = x;
        }

        boolean inWindow() {
            return x.ix.activity() == 2 && x.ix.stats().merges() == m0;
        }

        /** true when the merge was found in its build phase (activity 2) and still was 250 ms later; the collect loop over the rows is done by then */
        boolean start() {
            m0 = x.ix.stats().merges();
            t = new Thread(() -> {
                try {
                    x.ix.mergeNow();
                } catch (Throwable e) {
                    err[0] = e;
                }
            }, "ridreuse-merge");
            t.setDaemon(true);
            t.start();
            long end = System.nanoTime() + 20_000_000_000L;
            while (System.nanoTime() < end) {
                if (x.ix.activity() == 2) {
                    sleepMs(250);
                    return inWindow();
                }
                sleepMs(1);
            }
            return false;
        }

        void group(String what, Runnable ops) {
            boolean b = inWindow();
            ops.run();
            boolean a = inWindow();
            x.c.windowGroups++;
            if (b && a) {
                x.c.windowGroupsConfirmed++;
            } else {
                x.c.coverageFail++;
                x.note("VOID: group '" + what + "' not inside the running merge (running before=" + b + ", after=" + a + "); the merge throttle is too fast for this size");
            }
        }

        void join() {
            try {
                t.join(180_000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            if (t.isAlive()) {
                x.c.timeouts++;
                x.note("mergeNow thread still running after 180 s");
            }
            if (err[0] != null) {
                x.c.tierFailures++;
                x.note("mergeNow failed: " + err[0]);
            }
        }
    }

    // ------------------------------------------------------------------------------------------------------------------
    // cases

    static void threeSegments(Ctx x) {
        int n = x.cfg.n;
        for (int b = 0; b < 3; b++) {
            x.fill(b * n / 3, (b + 1) * n / 3);
            x.ix.sealNow();
        }
        x.precond("3 sealed segments and an empty tail", x.ix.segmentCount() == 3 && x.ix.pendingTail() == 0);
    }

    /** the old copy sits in the active tail */
    static void caseTail(Ctx x) {
        int n = x.cfg.n;
        x.fill(0, n);
        x.precond("all n vectors in the tail, no segment", x.ix.segmentCount() == 0 && x.ix.pendingTail() == n);
        int[] lv = x.liveShuffled();
        int nd = (int) (0.30 * lv.length);
        int[] dead = slice(lv, 0, nd);
        x.delAll(dead);
        x.checkpoint("a/30 percent deleted, all copies in the tail");
        x.addAll(slice(dead, 0, (int) (0.80 * nd)));
        x.checkpoint("a/80 percent of the deleted ids reused with new vectors B");
        int[] lv2 = x.liveShuffled();
        int[] g2 = slice(lv2, 0, (int) (0.30 * lv2.length));
        x.delAll(g2);
        x.addAll(g2);
        x.checkpoint("a/30 percent deleted and reused again (vectors C)");
        x.ix.sealNow();
        x.checkpoint("a/sealed after the reuse");
        x.quiet();
    }

    /** the old copies sit in one segment and in the tail */
    static void caseTailAndSegments(Ctx x) {
        int n = x.cfg.n;
        x.fill(0, n / 2);
        x.ix.sealNow();
        x.fill(n / 2, n);
        x.precond("one segment and n/2 vectors in the tail", x.ix.segmentCount() == 1 && x.ix.pendingTail() == n - n / 2);
        int[] lv = x.liveShuffled();
        int nd = (int) (0.30 * lv.length);
        int[] dead = slice(lv, 0, nd);
        x.delAll(dead);
        x.checkpoint("a2/30 percent deleted across the segment and the tail");
        x.addAll(slice(dead, 0, (int) (0.80 * nd)));
        x.checkpoint("a2/80 percent reused (B in the tail, old copies in the segment or the tail)");
        x.ix.sealNow();
        x.checkpoint("a2/sealed");
        x.ix.mergeNow();
        x.checkpoint("a2/merged");
        x.floorCheck("a2", true);
    }

    /** the old copies sit in sealed segments */
    static void caseSealed(Ctx x) {
        threeSegments(x);
        int[] lv = x.liveShuffled();
        int nd = (int) (0.30 * lv.length);
        int[] dead = slice(lv, 0, nd);
        x.delAll(dead);
        x.checkpoint("b/30 percent deleted, old copies in sealed segments");
        x.addAll(slice(dead, 0, (int) (0.80 * nd)));
        x.checkpoint("b/80 percent reused, B in the tail and A in the segments");
        x.ix.sealNow();
        x.precond("a 4th segment holds the reused copies", x.ix.segmentCount() == 4);
        x.checkpoint("b/B sealed beside the old copies (a 4th segment)");
        int[] lv2 = x.liveShuffled();
        int[] g2 = slice(lv2, 0, (int) (0.30 * lv2.length));
        x.delAll(g2);
        x.addAll(g2);
        x.checkpoint("b/30 percent deleted and reused again (C)");
        x.ix.sealNow();
        x.checkpoint("b/sealed again (5th segment)");
        x.ix.mergeNow();
        x.checkpoint("b/merged");
        x.floorCheck("b", true);
    }

    /** ids are re-added while a merge rebuilds the segments that hold their old copies, then again after the merge */
    static void caseReuseInMerge(Ctx x) {
        threeSegments(x);
        int[] lv = x.liveShuffled();
        int nd = (int) (0.30 * lv.length);
        int[] dead = slice(lv, 0, nd);
        int[] redo = slice(lv, nd, nd + lv.length / 10);
        x.delAll(dead);
        x.checkpoint("c1/30 percent deleted before the merge");
        final int[] reuse = slice(dead, 0, (int) (0.80 * nd));
        final int[] rest = slice(dead, (int) (0.80 * nd), nd);
        Reader rd = x.startReader(x.seed * 31 + 7);
        Mrg mg = new Mrg(x);
        boolean ok = mg.start();
        x.precond("mergeNow found in its build phase", ok);
        if (ok) {
            mg.group("re-add 80 percent of the deleted ids", () -> x.addAll(reuse));
            mg.group("delete then re-add 10 percent of the live ids", () -> {
                x.delAll(redo);
                x.addAll(redo);
            });
        }
        mg.join();
        if (rd != null) rd.finish();
        x.precond("exactly one merge ran", x.ix.stats().merges() == mg.m0 + 1 && x.ix.segmentCount() == 1);
        x.checkpoint("c1/after the merge swap");
        x.ix.sealNow();
        x.checkpoint("c1/B copies sealed beside the merged segment");
        // after a merge completed
        x.addAll(rest);
        x.checkpoint("d/dead ids reused after the merge dropped their stale copies");
        int[] lv2 = x.liveShuffled();
        int[] g2 = slice(lv2, 0, (int) (0.30 * lv2.length));
        x.delAll(g2);
        x.addAll(g2);
        x.checkpoint("d/30 percent of the merged segment deleted and reused");
        x.ix.sealNow();
        x.ix.mergeNow();
        x.checkpoint("d/sealed and merged again");
        x.floorCheck("d", true);
    }

    /** deletes, replaces, reuse and new ids inside a running merge: no delete may be lost when the merged segment is swapped in */
    static void caseDeleteInMerge(Ctx x) {
        threeSegments(x);
        int n = x.cfg.n;
        int[] lv = x.liveShuffled();
        final int[] dDel = slice(lv, 0, n / 10);
        final int[] dDr = slice(lv, n / 10, 2 * (n / 10));
        final int[] dDdd = slice(lv, 2 * (n / 10), 2 * (n / 10) + n / 20);
        final int[] dRep = slice(lv, 2 * (n / 10) + n / 20, 2 * (n / 10) + 2 * (n / 20));
        final int[] dNew = range(n, n + n / 20);
        Reader rd = x.startReader(x.seed * 31 + 11);
        Mrg mg = new Mrg(x);
        boolean ok = mg.start();
        x.precond("mergeNow found in its build phase", ok);
        if (ok) {
            mg.group("delete 10 percent", () -> x.delAll(dDel));
            mg.group("delete then re-add 10 percent", () -> {
                x.delAll(dDr);
                sleepMs(20);
                x.addAll(dDr);
            });
            mg.group("delete, re-add, delete 5 percent", () -> {
                for (int id : dDdd) {
                    x.del(id);
                    x.add(id);
                    x.del(id);
                }
            });
            mg.group("replace 5 percent", () -> x.addAll(dRep));
            mg.group("add 5 percent brand-new ids", () -> x.addAll(dNew));
        }
        mg.join();
        if (rd != null) rd.finish();
        x.precond("exactly one merge ran", x.ix.stats().merges() == mg.m0 + 1 && x.ix.segmentCount() == 1);
        x.checkpoint("c2/after the merge swap");
        x.ix.sealNow();
        x.checkpoint("c2/sealed");
        x.ix.mergeNow();
        x.checkpoint("c2/merged again");
        x.floorCheck("c2", true);
    }

    /** one id goes through four generations, each in its own segment, so the floor moves up four times */
    static void caseFloorChain(Ctx x) {
        int n = x.cfg.n;
        x.fill(0, n);
        x.ix.sealNow();
        int[] lv = x.liveShuffled();
        int e = Math.max(60, n / 50);
        int[] chain = slice(lv, 0, e);
        for (int gen = 1; gen <= 4; gen++) {
            x.delAll(chain);
            x.checkpoint("e/generation " + gen + ": chain ids deleted (the floor moves up)");
            x.addAll(chain);
            x.ix.sealNow();
            x.checkpoint("e/generation " + gen + ": re-added and sealed (older generations sit in older segments)");
            if (gen == 2) x.precond("floor holds at least n/100 entries after generation 2", x.ix.stats().floorEntries() >= n / 100);
        }
        x.ix.mergeNow();
        x.checkpoint("e/merged (every older generation dropped)");
        x.floorCheck("e", true);
        x.delAll(chain);
        x.addAll(chain);
        x.checkpoint("e/one more delete and re-add after the merge");
    }

    /** operations inside a seal's publish window: before the swap the frozen tail holds every old copy, right after the swap the new segment is visible */
    static void caseSealWindow(Ctx x) {
        int n = x.cfg.n;
        x.fill(0, n);
        int[] lv = x.liveShuffled();
        final int a = n / 10, b = 2 * (n / 10), cc = 3 * (n / 10), d = cc + n / 20;
        final int[] dDel = slice(lv, 0, a);
        final int[] dDr = slice(lv, a, b);
        final int[] dRep = slice(lv, b, cc);
        final int[] dDdd = slice(lv, cc, d);
        final int[] dLate = slice(lv, d, d + n / 20);
        final int[] dNew = range(n, n + n / 20);
        final int[] seen = new int[2];
        final long[] pend = new long[2];
        x.ix.setSealHooks(() -> {
            pend[0] = x.ix.pendingTail();
            x.delAll(dDel);
            x.delAll(dDr);
            x.addAll(dDr);
            x.addAll(dRep);
            for (int id : dDdd) {
                x.del(id);
                x.add(id);
                x.del(id);
            }
            x.addAll(dNew);
            x.checkpoint("s/inside the hook before the publish (the frozen tail still holds the old copies)");
            seen[0]++;
        }, () -> {
            pend[1] = x.ix.pendingTail();
            x.delAll(dLate);
            x.checkpoint("s/inside the hook right after the swap");
            seen[1]++;
        });
        x.ix.sealNow();
        x.ix.setSealHooks(null, null);
        x.precond("both seal hooks ran once", seen[0] == 1 && seen[1] == 1);
        x.precond("the frozen tail held all n vectors at hook entry", pend[0] == n);
        x.checkpoint("s/after the seal");
        x.ix.sealNow();
        x.checkpoint("s/second seal (the new copies)");
        x.ix.mergeNow();
        x.checkpoint("s/merged");
        x.floorCheck("s", true);
    }

    /** a background worker seals and merges while one writer reuses ids and one reader searches all the time */
    static void caseFuzz(Ctx x) {
        int n = x.cfg.n;
        x.fill(0, n);
        x.quiet();
        x.checkpoint("f/after the initial fill and settle");
        Reader rd = x.startReader(x.seed * 31 + 13);
        SplittableRandom r = new SplittableRandom(x.seed * 977L + 5L);
        int nextNew = n;
        int ops = n;
        for (int i = 0; i < ops; i++) {
            int act0 = x.ix.activity();
            int kind = r.nextInt(100);
            if (kind < 35) {
                for (int t = 0; t < 6; t++) {
                    int id = r.nextInt(nextNew);
                    if (x.m.cur[id] >= 0) { x.del(id); break; }
                }
            } else if (kind < 65) {
                for (int t = 0; t < 10; t++) {
                    int id = r.nextInt(nextNew);
                    if (x.m.cur[id] < 0) { x.add(id); break; }
                }
            } else if (kind < 80) {
                for (int t = 0; t < 6; t++) {
                    int id = r.nextInt(nextNew);
                    if (x.m.cur[id] >= 0) { x.add(id); break; }
                }
            } else if (kind < 90) {
                for (int t = 0; t < 6; t++) {
                    int id = r.nextInt(nextNew);
                    if (x.m.cur[id] >= 0) { x.del(id); x.add(id); break; }
                }
            } else if (kind < 95) {
                for (int t = 0; t < 6; t++) {
                    int id = r.nextInt(nextNew);
                    if (x.m.cur[id] >= 0) { x.del(id); x.add(id); x.del(id); break; }
                }
            } else if (nextNew < x.m.cap) {
                x.add(nextNew++);
            }
            int act1 = x.ix.activity();
            if (act0 == 1 && act1 == 1) x.c.opsInSeal++;
            if (act0 == 2 && act1 == 2) x.c.opsInMerge++;
        }
        if (rd != null) rd.finish();
        x.quiet();
        x.checkpoint("f/final, quiet, after the background seals and merges");
        TieredIndex.Stats s = x.ix.stats();
        x.precond("the background worker sealed at least 3 times and merged at least once (seals=" + s.seals() + ", merges=" + s.merges() + ")", s.seals() >= 3 && s.merges() >= 1);
        x.ix.sealNow();
        x.ix.mergeNow();
        x.checkpoint("f/after sealNow and mergeNow");
        x.floorCheck("f", false);
    }

    // ------------------------------------------------------------------------------------------------------------------
    // driver

    static TieredIndex.Config configFor(String name, Cfg cfg) {
        int n = cfg.n;
        switch (name) {
            case "c1_reuse_in_merge": return detCfg(Math.max(300, n / 5));
            case "c2_delete_in_merge": return detCfg(Math.max(300, n / 5));
            case "f_fuzz_background": return tcfg(Math.max(200, n / 12), 100, 3, 2, 0.3, Math.max(1000, n * 2));
            default: return detCfg(0);
        }
    }

    static void runCase(String name, Ctx x) {
        switch (name) {
            case "a_tail" -> caseTail(x);
            case "a2_tail_and_segments" -> caseTailAndSegments(x);
            case "b_sealed" -> caseSealed(x);
            case "c1_reuse_in_merge" -> caseReuseInMerge(x);
            case "c2_delete_in_merge" -> caseDeleteInMerge(x);
            case "e_floor_chain" -> caseFloorChain(x);
            case "s_seal_window" -> caseSealWindow(x);
            case "f_fuzz_background" -> caseFuzz(x);
            default -> throw new IllegalStateException("unknown case " + name);
        }
    }

    /** the ANN allowance: the miss share of reused ids may be at most max(floor, 3 x the share of untouched ids); a rate that was never exercised is bad */
    static boolean rateBad(long missR, long qR, long missP, long qP, double floor) {
        if (qR == 0) return true;
        double rr = (double) missR / qR;
        double rp = qP == 0 ? 0.0 : (double) missP / qP;
        return rr > Math.max(floor, 3.0 * rp);
    }

    /**
     * One run: a fresh index per case, in the given order. Takes and returns JDK types only. The result has every counter as a Long under its own name, "case_NAME" with
     * the violations of each case, "pass" (Boolean), "line" (one summary line), "notes" (the first violations, as strings) and "tierClass".
     */
    static Map<String, Object> runOneMap(String cfgName, long seed, List<String> cases, boolean reader) {
        Cfg cfg = cfgByName(cfgName);
        for (String cs : cases) if (!CASES.contains(cs)) throw new IllegalArgumentException("unknown case " + cs);
        long t0 = System.nanoTime();
        Cnt c = new Cnt();
        LinkedHashMap<String, Long> per = new LinkedHashMap<>();
        for (String name : CASES) {
            if (!cases.contains(name)) continue;
            long h0 = c.hard();
            c.where = cfg.name() + " seed=" + seed + " " + name;
            Ctx x = new Ctx(cfg, name, seed, configFor(name, cfg), c, reader);
            try {
                runCase(name, x);
            } catch (Throwable t) {
                c.tierFailures++;
                x.note("EXCEPTION " + t);
                t.printStackTrace(System.out);
            } finally {
                x.close();
            }
            long dv = c.hard() - h0;
            per.put(name, dv);
            System.out.println("CASE cfg=" + cfg.name() + " seed=" + seed + " case=" + name + " violations=" + dv + " hard_total=" + c.hard());
        }
        boolean readerCase = reader && (cases.contains("c1_reuse_in_merge") || cases.contains("c2_delete_in_merge") || cases.contains("f_fuzz_background"));
        int rateFail = 0;
        if (rateBad(c.exMiss[1], c.exQ[1], c.exMiss[0], c.exQ[0], 0.002)) rateFail++;
        if (rateBad(c.selfMiss1[1], c.selfQ[1], c.selfMiss1[0], c.selfQ[0], 0.01)) rateFail++;
        if (rateBad(c.selfMissK[1], c.selfQ[1], c.selfMissK[0], c.selfQ[0], 0.005)) rateFail++;
        if (readerCase && rateBad(c.cSelfMiss1, c.cSelfQ, 0, 0, 0.01)) rateFail++;
        boolean pass = c.hard() == 0 && rateFail == 0;
        StringBuilder sb = new StringBuilder();
        sb.append("RUN cfg=").append(cfg.name()).append(" seed=").append(seed).append(" cases=").append(cases.size()).append(" pass=").append(pass);
        sb.append(" hard=").append(c.hard()).append(" lostDeletes=").append(c.lostDeletes).append(" staleHits=").append(c.staleHits).append(" wrongScore=").append(c.wrongScore);
        sb.append(" dups=").append(c.dups).append(" unsorted=").append(c.unsorted).append(" phantoms=").append(c.phantoms).append(" oldRank1=").append(c.oldRank1);
        sb.append(" sizeMismatch=").append(c.sizeMismatch).append(" badDeleteReturn=").append(c.badDeleteReturn).append(" floorLeft=").append(c.floorLeft);
        sb.append(" tierFailures=").append(c.tierFailures).append(" coverageFail=").append(c.coverageFail).append(" timeouts=").append(c.timeouts);
        sb.append(" readerErrors=").append(c.readerErrors).append(" rateFail=").append(rateFail);
        sb.append(" exMissReused=").append(c.exMiss[1]).append("/").append(c.exQ[1]).append(" exMissUntouched=").append(c.exMiss[0]).append("/").append(c.exQ[0]);
        sb.append(" self1MissReused=").append(c.selfMiss1[1]).append("/").append(c.selfQ[1]).append(" self1MissUntouched=").append(c.selfMiss1[0]).append("/").append(c.selfQ[0]);
        sb.append(" selfKMissReused=").append(c.selfMissK[1]).append("/").append(c.selfQ[1]).append(" selfKMissUntouched=").append(c.selfMissK[0]).append("/").append(c.selfQ[0]);
        sb.append(" concSelfMiss=").append(c.cSelfMiss1).append("/").append(c.cSelfQ);
        sb.append(" windowGroups=").append(c.windowGroupsConfirmed).append("/").append(c.windowGroups);
        sb.append(" opsTotal=").append(c.opsTotal).append(" opsInSeal=").append(c.opsInSeal).append(" opsInMerge=").append(c.opsInMerge);
        sb.append(" floorNonEmpty=").append(c.floorNonEmpty).append(" searches=").append(c.searches).append(" hitsChecked=").append(c.hitsChecked).append(" idsChecked=").append(c.idsChecked);
        sb.append(String.format(" secs=%.1f", (System.nanoTime() - t0) / 1e9));
        for (Map.Entry<String, Long> e : per.entrySet()) sb.append(" case_").append(e.getKey()).append("=").append(e.getValue());
        System.out.println(sb);

        LinkedHashMap<String, Object> out = new LinkedHashMap<>();
        out.put("pass", pass);
        out.put("line", sb.toString());
        List<String> all = new ArrayList<>(c.events);
        all.addAll(c.notes);
        out.put("notes", all);
        out.put("tierClass", tierClassLocation());
        out.put("hard", c.hard());
        out.put("exact", c.exact());
        out.put("lostDeletes", c.lostDeletes);
        out.put("staleHits", c.staleHits);
        out.put("wrongScore", c.wrongScore);
        out.put("oldRank1", c.oldRank1);
        out.put("dups", c.dups);
        out.put("unsorted", c.unsorted);
        out.put("phantoms", c.phantoms);
        out.put("sizeMismatch", c.sizeMismatch);
        out.put("badDeleteReturn", c.badDeleteReturn);
        out.put("floorLeft", c.floorLeft);
        out.put("tierFailures", c.tierFailures);
        out.put("coverageFail", c.coverageFail);
        out.put("timeouts", c.timeouts);
        out.put("readerErrors", c.readerErrors);
        out.put("rateFail", (long) rateFail);
        out.put("windowGroups", c.windowGroups);
        out.put("windowGroupsConfirmed", c.windowGroupsConfirmed);
        out.put("searches", c.searches);
        out.put("hitsChecked", c.hitsChecked);
        out.put("idsChecked", c.idsChecked);
        for (Map.Entry<String, Long> e : per.entrySet()) out.put("case_" + e.getKey(), e.getValue());
        return out;
    }
}
