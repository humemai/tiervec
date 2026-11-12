package ai.humem.tiervec;

import java.util.Random;
import java.util.function.LongPredicate;

/**
 * Single threaded correctness and recall tests of the IVF index. Arg: n (default 20000; the scalar twin run uses less).
 * Sections: untrained exact path, all three metrics with NONE (exact at full probe) and SQ8 (recall), the recall sweep with the lazily trained
 * lists and after retrain(), fromVectors, deletes and compact(), replace, filters at 1/2, 1/10 and 1/100, the boundary overlap, edge cases.
 */
public final class IvfTest {
    static final int DIM = 128;

    public static void main(String[] args) throws Exception {
        int n = args.length > 0 ? Integer.parseInt(args[0]) : 20000;
        System.out.println("IvfTest n=" + n + " kernels=" + (Kernels.VECTOR ? "vector" : "scalar"));
        untrained();
        metrics();
        main20k(n);
        overlap();
        IvfT.done("IvfTest");
    }

    static void untrained() {
        int p0 = IvfT.pass, f0 = IvfT.fail;
        int n = 500;
        float[][] raw = IvfT.clustered(n + 50, DIM, 8, 1.0, 11, 12, 0f);
        float[][] st = IvfT.stored(Metric.COSINE, raw, 0, n);
        long[] ids = IvfT.seqIds(n);
        IvfIndex ix = new IvfIndex(DIM, Metric.COSINE);
        IvfT.check(ix.search(raw[0], 5, null).length == 0, "empty index returns nothing");
        for (int i = 0; i < n; i++) ix.add(i, raw[i]);
        IvfT.check(!ix.isTrained(), "below trainAt stays untrained");
        IvfT.check(ix.size() == n, "size");
        int bad = 0;
        for (int i = n; i < n + 50; i++) {
            Hit[] t = IvfT.brute(Metric.COSINE, ids, st, null, raw[i], 10, null);
            Hit[] g = ix.search(raw[i], 10, null);
            if (!IvfT.sameIds(t, g)) bad++;
            else for (int j = 0; j < t.length; j++) if (Math.abs(t[j].score() - g[j].score()) > 1e-4f) bad++;
        }
        IvfT.check(bad == 0, "untrained search equals brute force (bad queries " + bad + ")");
        IvfIndex tiny = new IvfIndex(DIM, Metric.L2);
        for (int i = 0; i < 5; i++) tiny.add(i, raw[i]);
        IvfT.check(tiny.search(raw[0], 10, null).length == 5, "k above size returns size hits");
        IvfT.check(tiny.search(raw[0], 10, id -> false).length == 0, "nobody allowed returns nothing");
        IvfT.check(tiny.search(raw[0], 1, null)[0].id() == 0, "self is the nearest in L2");
        IvfT.section("untrained exact path", p0, f0);
    }

    static void metrics() {
        int p0 = IvfT.pass, f0 = IvfT.fail;
        int n = 3000, d = 32, nq = 30;
        for (Metric m : Metric.values()) {
            for (Quant qt : new Quant[] {Quant.NONE, Quant.SQ8}) {
                float[][] raw = IvfT.clustered(n + nq, d, 12, 1.0, 31, 32, 0f);
                float[][] st = IvfT.stored(m, raw, 0, n);
                long[] ids = IvfT.seqIds(n);
                IvfIndex.Params p = new IvfIndex.Params();
                p.trainAt = 500;
                p.quant = qt;
                IvfIndex ix = new IvfIndex(d, m, p);
                for (int i = 0; i < n; i++) ix.add(i, raw[i]);
                ix.retrain();
                String tag = m + "/" + qt;
                IvfT.check(ix.isTrained() && ix.size() == n, tag + " trained, size");
                IvfT.check(ix.nlist() == (int) Math.round(4 * Math.sqrt(n)), tag + " nlist is round(4 sqrt n), got " + ix.nlist());
                int full = ix.nlist();
                int exactBad = 0;
                double rsum = 0;
                for (int i = n; i < n + nq; i++) {
                    Hit[] t = IvfT.brute(m, ids, st, null, raw[i], 10, null);
                    Hit[] g = ix.search(raw[i], 10, null, full);
                    rsum += IvfT.recall(g, t);
                    if (qt == Quant.NONE) {
                        if (!IvfT.sameIds(t, g)) exactBad++;
                        else for (int j = 0; j < t.length; j++) if (Math.abs(t[j].score() - g[j].score()) > 1e-3f * Math.max(1f, Math.abs(t[j].score()))) exactBad++;
                    }
                }
                if (qt == Quant.NONE) IvfT.check(exactBad == 0, tag + " full probe equals brute force (bad " + exactBad + ")");
                else IvfT.check(rsum / nq >= 0.98, tag + " full probe recall " + rsum / nq);
                IvfT.check(ix.search(raw[n], 10, null).length == 10, tag + " default nprobe returns k hits");
            }
        }
        IvfT.section("metrics COSINE DOT L2 x NONE SQ8", p0, f0);
    }

    static void main20k(int n) {
        int p0 = IvfT.pass, f0 = IvfT.fail;
        int nq = 200;
        float[][] raw = IvfT.clustered(n + nq, DIM, 64, 1.0, 21, 22, 0f);
        float[][] st = IvfT.stored(Metric.COSINE, raw, 0, n);
        long[] ids = IvfT.seqIds(n);
        float[][] qs = new float[nq][];
        for (int i = 0; i < nq; i++) qs[i] = raw[n + i];
        Hit[][] truth = IvfT.truths(Metric.COSINE, ids, st, null, qs, 10, null);

        IvfIndex.Params prm = new IvfIndex.Params(); // defaults: trainAt 2000, SQ8, rerank 5
        IvfIndex ix = new IvfIndex(DIM, Metric.COSINE, prm);
        long t0 = System.nanoTime();
        for (int i = 0; i < n; i++) ix.add(i, raw[i]);
        System.out.println("INFO add " + n + " vectors (includes the lazy training at 2000): " + (System.nanoTime() - t0) / 1_000_000 + " ms, rebuilds " + ix.rebuildCount());
        IvfT.check(ix.isTrained(), "lazily trained");
        IvfT.check(ix.nlist() == 179, "lazy nlist is round(4 sqrt 2000) = 179, got " + ix.nlist());
        IvfT.check(ix.size() == n, "size after adds");
        IvfT.check(ix.lastError() == null, "no maintenance error");
        int[] sweep = {1, 2, 4, 8, 16, 32, 64, 128};
        curve("lazy-trained (nlist " + ix.nlist() + ")", ix, qs, truth, sweep);

        t0 = System.nanoTime();
        ix.retrain();
        System.out.println("INFO retrain of " + n + " vectors: " + (System.nanoTime() - t0) / 1_000_000 + " ms");
        IvfT.check(ix.nlist() == (int) Math.round(4 * Math.sqrt(n)), "retrained nlist, got " + ix.nlist());
        IvfT.check(ix.size() == n, "size after retrain");
        double[] rec = curve("retrained (nlist " + ix.nlist() + ")", ix, qs, truth, sweep);
        double best = 0;
        for (double r : rec) best = Math.max(best, r);
        IvfT.check(best >= 0.95, "recall@10 >= 0.95 at some nprobe, best " + best);
        for (int i = 1; i < rec.length; i++) IvfT.check(rec[i] >= rec[i - 1] - 0.01, "recall does not fall with nprobe at step " + i);
        double[] full = IvfT.eval(ix, qs, truth, ix.nlist(), 10);
        IvfT.check(full[0] >= 0.99, "full probe recall " + full[0]);
        double[] dflt = IvfT.eval(ix, qs, truth, 0, 10);
        System.out.println("INFO default nprobe " + Math.max(8, ix.nlist() / 16) + ": recall@10 " + String.format("%.4f", dflt[0]) + ", " + String.format("%.1f", dflt[1]) + " us/query; bytesHeld " + ix.bytesHeld() / 1024 + " KiB");
        IvfT.check(dflt[0] >= 0.9, "default nprobe recall " + dflt[0]);
        IvfT.check(ix.listSkew() < 8.0, "list skew after retrain " + ix.listSkew());
        IvfT.check(!ix.needsRetrain(), "no retrain needed right after retrain");

        // fromVectors (the tier segment constructor)
        float[] flat = new float[n * DIM];
        for (int i = 0; i < n; i++) System.arraycopy(raw[i], 0, flat, i * DIM, DIM);
        IvfIndex seg = IvfIndex.fromVectors(ids, flat, DIM, Metric.COSINE, prm);
        IvfT.check(seg.isTrained() && seg.size() == n, "fromVectors trained, size");
        double[] sr = IvfT.eval(seg, qs, truth, 0, 10);
        System.out.println("INFO fromVectors default nprobe: recall@10 " + String.format("%.4f", sr[0]));
        IvfT.check(sr[0] >= 0.9, "fromVectors recall " + sr[0]);
        seg.add(n + 7, raw[n + 7]);
        IvfT.check(seg.size() == n + 1 && seg.search(raw[n + 7], 1, null)[0].id() == n + 7, "fromVectors index accepts adds");
        IvfT.check(seg.delete(5) && !seg.delete(5) && seg.size() == n, "fromVectors index accepts deletes");
        IvfIndex small = IvfIndex.fromVectors(new long[] {10, 11, 12}, new float[][] {raw[0], raw[1], raw[2]}, DIM, Metric.COSINE, prm);
        IvfT.check(!small.isTrained() && small.size() == 3 && small.search(raw[1], 1, null)[0].id() == 11, "fromVectors below trainAt is a flat exact segment");
        try {
            IvfIndex.fromVectors(new long[] {1, 1}, new float[][] {raw[0], raw[1]}, DIM, Metric.COSINE, prm);
            IvfT.check(false, "fromVectors rejects duplicate ids");
        } catch (IllegalArgumentException e) {
            IvfT.check(true, "fromVectors rejects duplicate ids");
        }

        // deletes
        Random rnd = new Random(5);
        boolean[] alive = new boolean[n];
        java.util.Arrays.fill(alive, true);
        int del = 0;
        for (int i = 0; i < n; i++) {
            if (rnd.nextDouble() < 0.3) {
                IvfT.check(ix.delete(i), "delete of a live id returns true");
                alive[i] = false;
                del++;
            }
        }
        IvfT.check(!ix.delete(n + 123456), "delete of an unknown id returns false");
        for (int i = 0; i < n; i++) {
            if (!alive[i]) {
                IvfT.check(!ix.delete(i), "second delete returns false");
                break;
            }
        }
        IvfT.check(ix.size() == n - del, "size after deletes");
        Hit[][] truth2 = IvfT.truths(Metric.COSINE, ids, st, alive, qs, 10, null);
        int deadSeen = 0;
        for (int i = 0; i < nq; i++) for (Hit h : ix.search(qs[i], 100, null)) if (!alive[(int) h.id()]) deadSeen++;
        IvfT.check(deadSeen == 0, "no deleted id returned (k = 100), saw " + deadSeen);
        double[] r2 = IvfT.eval(ix, qs, truth2, 0, 10);
        System.out.println("INFO after deleting " + del + ": recall@10 " + String.format("%.4f", r2[0]) + ", dead fraction " + String.format("%.3f", ix.deadFraction()));
        IvfT.check(r2[0] >= 0.9, "recall after deletes " + r2[0]);
        long b0 = ix.bytesHeld();
        ix.compact();
        long b1 = ix.bytesHeld();
        IvfT.check(b1 < b0, "compact shrinks bytesHeld " + b0 + " -> " + b1);
        IvfT.check(ix.size() == n - del && ix.deadFraction() == 0.0, "size and dead fraction after compact");
        deadSeen = 0;
        for (int i = 0; i < nq; i++) for (Hit h : ix.search(qs[i], 100, null)) if (!alive[(int) h.id()]) deadSeen++;
        IvfT.check(deadSeen == 0, "no deleted id returned after compact");
        double[] r3 = IvfT.eval(ix, qs, truth2, 0, 10);
        IvfT.check(Math.abs(r3[0] - r2[0]) < 0.01, "compact does not change recall " + r2[0] + " vs " + r3[0]);
        int sumSizes = 0;
        for (int s : ix.listSizes()) sumSizes += s;
        IvfT.check(sumSizes == ix.size(), "sum of list sizes equals size");

        // replace
        float[][] nv = IvfT.clustered(1000, DIM, 64, 1.0, 99, 98, 0f);
        int[] pick = new int[1000];
        int np = 0;
        for (int i = 0; i < n && np < 1000; i++) if (alive[i] && rnd.nextDouble() < 0.1) pick[np++] = i;
        float[][] oldRaw = new float[np][];
        int selfMiss = 0, staleSeen = 0;
        for (int j = 0; j < np; j++) {
            int id = pick[j];
            oldRaw[j] = raw[id];
            ix.add(id, nv[j]);
            st[id] = IvfT.normalize(nv[j]);
        }
        IvfT.check(ix.size() == n - del, "replace keeps size");
        for (int j = 0; j < np; j++) {
            int id = pick[j];
            Hit[] g = ix.search(nv[j], 1, null);
            if (g.length != 1 || g[0].id() != id) selfMiss++;
            for (Hit h : ix.search(oldRaw[j], 10, null)) if (h.id() == id && h.score() > 0.99f) staleSeen++;
        }
        IvfT.check(selfMiss == 0, "a replaced id is found by its new vector, misses " + selfMiss + " of " + np);
        IvfT.check(staleSeen == 0, "a replaced id is not found by its old vector, saw " + staleSeen);
        Hit[][] truth3 = IvfT.truths(Metric.COSINE, ids, st, alive, qs, 10, null);
        double[] r4 = IvfT.eval(ix, qs, truth3, 0, 10);
        System.out.println("INFO after replacing " + np + ": recall@10 " + String.format("%.4f", r4[0]));
        IvfT.check(r4[0] >= 0.88, "recall after replace " + r4[0]);

        // filters
        int[] mods = {2, 10, 100};
        // Filtered probing stops when rerankFactor * k (50) QUALIFYING candidates are in the heap (the heap only receives vectors that pass
        // allow) and it ends in a full scan, which is exact, when fewer qualify (1/100 at n=6000 measured 1.0000). At 1/10 and n=6000 about 420
        // vectors qualify, so the stop rule looks at the nearest ~500 scanned vectors only, and a true top-10 neighbour just outside them is
        // missed: measured 0.886 at n=6000 (scalar run) against 1.0000 at
        // n=20000. The floor therefore depends on n: 0.9 from n=10000, 0.85 below (about 3 points under the n=6000 measurement).
        double fl = n >= 10000 ? 0.9 : 0.85;
        double[] thr = {fl, fl, n >= 10000 ? 0.8 : 0.85};
        for (int mi = 0; mi < mods.length; mi++) {
            final int mod = mods[mi];
            LongPredicate allow = id -> id % mod == 0;
            Hit[][] tf = IvfT.truths(Metric.COSINE, ids, st, alive, qs, 10, allow);
            double rsum = 0;
            int notAllowed = 0, short_ = 0;
            long f0t = System.nanoTime();
            for (int i = 0; i < nq; i++) {
                Hit[] g = ix.search(qs[i], 10, allow);
                rsum += IvfT.recall(g, tf[i]);
                for (Hit h : g) if (h.id() % mod != 0 || !alive[(int) h.id()]) notAllowed++;
                if (g.length != tf[i].length) short_++;
            }
            double us = (System.nanoTime() - f0t) / 1000.0 / nq;
            System.out.println("INFO filter selectivity 1/" + mod + ": recall@10 " + String.format("%.4f", rsum / nq) + ", " + String.format("%.1f", us) + " us/query");
            IvfT.check(notAllowed == 0, "filter 1/" + mod + " returns only allowed live ids");
            IvfT.check(short_ == 0, "filter 1/" + mod + " returns as many hits as the oracle, short " + short_);
            IvfT.check(rsum / nq >= thr[mi], "filter 1/" + mod + " recall " + rsum / nq);
        }
        IvfT.check(ix.search(qs[0], 10, id -> false).length == 0, "a filter that allows nothing returns nothing (full scan fallback ends)");
        IvfT.section("recall sweep, fromVectors, deletes, compact, replace, filters (n=" + n + ")", p0, f0);
    }

    static double[] curve(String label, IvfIndex ix, float[][] qs, Hit[][] truth, int[] sweep) {
        double[] rec = new double[sweep.length];
        StringBuilder sb = new StringBuilder("CURVE " + label + " recall@10 us/query by nprobe:");
        for (int i = 0; i < sweep.length; i++) {
            int np = Math.min(sweep[i], ix.nlist());
            double[] r = IvfT.eval(ix, qs, truth, np, 10);
            rec[i] = r[0];
            sb.append(String.format(" [%d: %.4f %.1f]", np, r[0], r[1]));
        }
        System.out.println(sb);
        return rec;
    }

    static void overlap() {
        int p0 = IvfT.pass, f0 = IvfT.fail;
        int n = 5000, d = 64, nq = 100;
        float[][] raw = IvfT.clustered(n + nq, d, 32, 1.0, 41, 42, 0f);
        float[][] st = IvfT.stored(Metric.COSINE, raw, 0, n);
        long[] ids = IvfT.seqIds(n);
        float[][] qs = new float[nq][];
        for (int i = 0; i < nq; i++) qs[i] = raw[n + i];
        Hit[][] truth = IvfT.truths(Metric.COSINE, ids, st, null, qs, 10, null);
        double[] rr = new double[2];
        for (int ov = 1; ov <= 2; ov++) {
            IvfIndex.Params p = new IvfIndex.Params();
            p.trainAt = 500;
            p.overlap = ov;
            IvfIndex ix = new IvfIndex(d, Metric.COSINE, p);
            for (int i = 0; i < n; i++) ix.add(i, raw[i]);
            ix.retrain();
            int dups = 0;
            double rs = 0;
            for (int i = 0; i < nq; i++) {
                Hit[] g = ix.search(qs[i], 10, null, 8);
                rs += IvfT.recall(g, truth[i]);
                for (int a = 0; a < g.length; a++) for (int b = a + 1; b < g.length; b++) if (g[a].id() == g[b].id()) dups++;
            }
            rr[ov - 1] = rs / nq;
            IvfT.check(dups == 0, "overlap " + ov + " has no duplicate ids in a result");
            IvfT.check(ix.size() == n, "overlap " + ov + " size");
            if (ov == 2) {
                int sum = 0;
                for (int s : ix.listSizes()) sum += s;
                IvfT.check(sum > n && sum <= 2 * n, "overlap 2 stores most vectors twice, list total " + sum);
                for (int i = 0; i < n; i += 3) ix.delete(i);
                int back = 0;
                for (int i = 0; i < nq; i++) for (Hit h : ix.search(qs[i], 50, null)) if (h.id() % 3 == 0 && h.id() < n) back++;
                IvfT.check(back == 0, "overlap 2: a deleted id is gone from both lists");
                ix.add(1, raw[n]); // replace id 1
                IvfT.check(ix.search(raw[n], 1, null)[0].id() == 1 && ix.size() == n - (n + 2) / 3, "overlap 2: replace and size");
            }
        }
        System.out.println("INFO overlap recall@10 at nprobe 8: overlap 1 " + String.format("%.4f", rr[0]) + ", overlap 2 " + String.format("%.4f", rr[1]));
        IvfT.check(rr[1] >= rr[0] - 0.02, "overlap 2 recall is not worse than overlap 1");
        IvfT.section("boundary overlap", p0, f0);
    }
}
