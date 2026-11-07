package com.humemai.vec;

import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/** Functional tests of the graph tier. Usage: GraphTest [quick|all|bench]. Sizes are capped at 100000 vectors. */
public final class GraphTest {
    static final int DIM = 128;

    static float[][] data(GraphT.Gen g, int n) { float[][] d = new float[n][]; for (int i = 0; i < n; i++) d[i] = g.vec(i); return d; }

    static void buildParallel(GraphIndex idx, float[][] d, int threads) throws Exception {
        ExecutorService ex = Executors.newFixedThreadPool(threads);
        AtomicInteger next = new AtomicInteger();
        for (int t = 0; t < threads; t++) ex.submit(() -> { int i; while ((i = next.getAndIncrement()) < d.length) idxAdd(idx, i, d[i]); });
        ex.shutdown();
        if (!ex.awaitTermination(300, TimeUnit.SECONDS)) throw new IllegalStateException("build timeout");
    }
    static void idxAdd(GraphIndex idx, int id, float[] v) { idx.add(id, v); }

    static void recallCurve(String label, GraphIndex.Params p, GraphT.Gen g, int n, int nq, boolean[] needTarget) {
        float[][] d = data(g, n);
        GraphIndex idx = new GraphIndex(DIM, Metric.COSINE, p);
        long t0 = System.nanoTime();
        for (int i = 0; i < n; i++) idx.add(i, d[i]);
        double buildS = (System.nanoTime() - t0) / 1e9;
        System.out.printf("  [%s] n=%d build(1 thread)=%.1fs (%.0f inserts/s) size=%d bytesHeld=%.1f MB%n", label, n, buildS, n / buildS, idx.size(), idx.bytesHeld() / 1e6);
        GraphT.Gen qg = new GraphT.Gen(DIM, g.seed, g.cent.length, 0.4f); // same centroids, different noise
        float[][] qs = new float[nq][]; long[][] truth = new long[nq][];
        for (int i = 0; i < nq; i++) { qs[i] = qg.vec(1_000_000L + i); truth[i] = GraphT.brute(d, null, qs[i], 10, null, Metric.COSINE); }
        int[] efs = { 10, 20, 40, 80, 100, 200, 400, 800, 1600 };
        double best = 0;
        for (int ef : efs) {
            double r = 0; long[] lat = new long[nq];
            for (int i = 0; i < nq; i++) { long a = System.nanoTime(); Hit[] h = idx.search(qs[i], 10, null, ef); lat[i] = System.nanoTime() - a; r += GraphT.recall(h, truth[i]); }
            r /= nq; best = Math.max(best, r);
            long[] s = GraphT.sorted(lat, nq);
            System.out.printf("    ef=%-4d recall@10=%.4f  p50=%d us  p99=%d us%n", ef, r, GraphT.pct(s, 0.5) / 1000, GraphT.pct(s, 0.99) / 1000);
        }
        GraphT.check(best >= 0.95, label + ": recall@10 >= 0.95 at some ef (best " + best + ")");
    }

    static void testRecall() {
        recallCurve("clustered fp32", GraphIndex.Params.defaults(), new GraphT.Gen(DIM, 7, 50, 0.4f), 20000, 300, null);
        recallCurve("uniform fp32", GraphIndex.Params.defaults(), new GraphT.Gen(DIM, 11, 1, 100f), 20000, 300, null);
        recallCurve("clustered sq8", GraphIndex.Params.defaults().withSq8(true), new GraphT.Gen(DIM, 7, 50, 0.4f), 20000, 300, null);
    }

    static void testL2Dot() {
        for (Metric m : new Metric[] { Metric.L2, Metric.DOT }) {
            int n = 5000; GraphT.Gen g = new GraphT.Gen(DIM, 3, 20, 0.4f); float[][] d = data(g, n);
            GraphIndex idx = new GraphIndex(DIM, m);
            for (int i = 0; i < n; i++) idx.add(i, d[i]);
            double r = 0; int nq = 100;
            for (int i = 0; i < nq; i++) { float[] q = g.vec(900000 + i); Hit[] h = idx.search(q, 10, null, 200); r += GraphT.recall(h, GraphT.brute(d, null, q, 10, null, m)); }
            r /= nq; System.out.printf("  %s recall@10 (ef 200, n=5000) = %.4f%n", m, r);
            GraphT.check(r >= 0.95, m + " recall >= 0.95 (" + r + ")");
            Hit[] h = idx.search(d[42], 3, null);
            GraphT.check(h[0].id() == 42, m + " self is best (got " + h[0].id() + ")");
            if (m == Metric.L2) GraphT.check(Math.abs(h[0].score()) < 1e-4, "L2 self score ~ 0, got " + h[0].score());
        }
    }

    static void testReadYourWritesSingleThread() {
        GraphT.Gen g = new GraphT.Gen(DIM, 5, 30, 0.4f); GraphIndex idx = new GraphIndex(DIM, Metric.COSINE);
        int miss = 0;
        for (int i = 0; i < 3000; i++) {
            float[] v = g.vec(i); idx.add(i, v);
            Hit[] h = idx.search(v, 5, null, 50);
            boolean f = false; for (Hit x : h) if (x.id() == i) f = true;
            if (!f) miss++;
        }
        GraphT.check(miss == 0, "every add immediately findable, misses=" + miss);
    }

    static void testDeleteConsolidate() {
        int n = 6000; GraphT.Gen g = new GraphT.Gen(DIM, 9, 40, 0.4f); float[][] d = data(g, n);
        GraphIndex idx = new GraphIndex(DIM, Metric.COSINE);
        for (int i = 0; i < n; i++) idx.add(i, d[i]);
        boolean[] del = new boolean[n]; Set<Long> delIds = new HashSet<>();
        for (int i = 0; i < n; i++) if (i % 10 < 4) { GraphT.check(idx.delete(i), "delete existing " + i); del[i] = true; delIds.add((long) i); }
        GraphT.check(!idx.delete(0), "second delete of 0 returns false");
        GraphT.check(!idx.delete(99999999L), "delete of unknown id returns false");
        GraphT.check(idx.size() == n - delIds.size(), "size after deletes " + idx.size());
        GraphT.check(idx.tombstoneCount() == delIds.size(), "tombstone count " + idx.tombstoneCount());
        int nq = 200; double rBefore = 0; boolean leaked = false;
        float[][] qs = new float[nq][]; long[][] truth = new long[nq][];
        for (int i = 0; i < nq; i++) { qs[i] = g.vec(800000 + i); truth[i] = GraphT.brute(d, del, qs[i], 10, null, Metric.COSINE); }
        for (int i = 0; i < nq; i++) { Hit[] h = idx.search(qs[i], 10, null, 100); rBefore += GraphT.recall(h, truth[i]); for (Hit x : h) if (delIds.contains(x.id())) leaked = true; }
        rBefore /= nq;
        GraphT.check(!leaked, "deleted ids never returned (before consolidate)");
        int hiBefore = idx.ordinalsAllocated();
        long b0 = idx.bytesHeld();
        long t0 = System.nanoTime();
        int recycled = idx.consolidate();
        double ms = (System.nanoTime() - t0) / 1e6;
        double rAfter = 0; leaked = false;
        for (int i = 0; i < nq; i++) { Hit[] h = idx.search(qs[i], 10, null, 100); rAfter += GraphT.recall(h, truth[i]); for (Hit x : h) if (delIds.contains(x.id())) leaked = true; }
        rAfter /= nq;
        System.out.printf("  recall@10 ef=100 before consolidate=%.4f after=%.4f; recycled=%d in %.0f ms; orphansRepaired=%d; bytes %.1f -> %.1f MB%n",
                rBefore, rAfter, recycled, ms, idx.orphansRepaired(), b0 / 1e6, idx.bytesHeld() / 1e6);
        GraphT.check(recycled == delIds.size(), "all tombstones recycled: " + recycled);
        GraphT.check(idx.tombstoneCount() == 0, "no tombstones after consolidate: " + idx.tombstoneCount());
        GraphT.check(!leaked, "deleted ids never returned (after consolidate)");
        GraphT.check(rAfter >= 0.93, "recall after consolidate >= 0.93 (" + rAfter + ")");
        // no list may reference a recycled (REMOVED) slot: reinsert and check slot reuse
        int extra = 2000; float[][] d2 = new float[extra][];
        for (int i = 0; i < extra; i++) { d2[i] = g.vec(n + i); idx.add(n + i, d2[i]); }
        GraphT.check(idx.ordinalsAllocated() == hiBefore, "freed slots reused: ordinals " + hiBefore + " -> " + idx.ordinalsAllocated());
        int miss = 0; for (int i = 0; i < extra; i++) { Hit[] h = idx.search(d2[i], 10, null, 100); boolean f = false; for (Hit x : h) if (x.id() == n + i) f = true; if (!f) miss++; }
        GraphT.check(miss == 0, "recycled-slot inserts all findable, misses=" + miss);
        // second consolidate with nothing to do is a no-op
        GraphT.check(idx.consolidate() == 0, "idempotent consolidate");
        // delete everything and the index must keep working
        for (int i = 0; i < n + extra; i++) idx.delete(i);
        GraphT.check(idx.size() == 0, "size 0 after deleting all");
        GraphT.check(idx.search(qs[0], 10, null).length == 0, "empty search after deleting all");
        idx.consolidate();
        idx.add(1, d[1]); idx.add(2, d[2]);
        Hit[] h = idx.search(d[2], 2, null);
        GraphT.check(h.length == 2 && h[0].id() == 2, "index usable after full delete + consolidate");
    }

    static void testFilter() {
        int n = 10000; GraphT.Gen g = new GraphT.Gen(DIM, 13, 40, 0.4f); float[][] d = data(g, n);
        GraphIndex idx = new GraphIndex(DIM, Metric.COSINE);
        for (int i = 0; i < n; i++) idx.add(i, d[i]);
        for (int mod : new int[] { 2, 10, 100, 1000 }) {
            java.util.function.LongPredicate allow = id -> id % mod == 0;
            double r = 0; int nq = 100; boolean bad = false; int short_ = 0;
            for (int i = 0; i < nq; i++) {
                float[] q = g.vec(700000 + i);
                Hit[] h = idx.search(q, 10, allow, 100);
                long[] t = GraphT.brute(d, null, q, 10, allow, Metric.COSINE);
                r += GraphT.recall(h, t);
                for (Hit x : h) if (x.id() % mod != 0) bad = true;
                if (h.length < t.length) short_++;
                for (int j = 1; j < h.length; j++) if (h[j].score() > h[j - 1].score() + 1e-6f) bad = true;
            }
            r /= nq;
            System.out.printf("  filter 1/%d: recall@10=%.4f%n", mod, r);
            GraphT.check(!bad, "filter 1/" + mod + ": only allowed ids, best first");
            GraphT.check(short_ == 0, "filter 1/" + mod + ": returns k when k qualify, short=" + short_);
            GraphT.check(r >= (mod >= 100 ? 0.99 : 0.9), "filter 1/" + mod + " recall " + r);
        }
        GraphT.check(idx.search(g.vec(1), 10, id -> false).length == 0, "reject-all filter returns nothing");
    }

    static void testReplace() {
        GraphT.Gen g = new GraphT.Gen(DIM, 17, 20, 0.4f); GraphIndex idx = new GraphIndex(DIM, Metric.COSINE);
        for (int i = 0; i < 2000; i++) idx.add(i, g.vec(i));
        float[] nv = g.vec(555555);
        idx.add(7, nv);
        GraphT.check(idx.size() == 2000, "replace keeps size: " + idx.size());
        Hit[] h = idx.search(nv, 1, null, 100);
        GraphT.check(h[0].id() == 7 && h[0].score() > 0.999f, "replaced id found with its new vector, got " + h[0]);
        Hit[] h2 = idx.search(g.vec(7), 20, null, 100);
        int c = 0; for (Hit x : h2) if (x.id() == 7) c++;
        GraphT.check(c <= 1, "id 7 appears at most once: " + c);
        for (Hit x : h2) if (x.id() == 7) GraphT.check(x.score() < 0.99f, "old vector of 7 no longer matches strongly: " + x.score());
    }

    static void testSealAndRoundTrip() {
        int n = 8000; GraphT.Gen g = new GraphT.Gen(DIM, 19, 40, 0.4f); float[][] d = data(g, n);
        for (boolean sq8 : new boolean[] { false, true }) {
            GraphIndex idx = new GraphIndex(DIM, Metric.COSINE, GraphIndex.Params.defaults().withSq8(sq8));
            for (int i = 0; i < n; i++) idx.add(i, d[i]);
            boolean[] del = new boolean[n];
            for (int i = 0; i < n; i += 7) { idx.delete(i); del[i] = true; }
            SealedGraph sg = idx.seal();
            GraphT.check(sg.size() == idx.size(), "sealed size " + sg.size() + " vs " + idx.size());
            int nq = 150; double rm = 0, rs = 0; boolean leak = false;
            for (int i = 0; i < nq; i++) {
                float[] q = g.vec(600000 + i); long[] t = GraphT.brute(d, del, q, 10, null, Metric.COSINE);
                rm += GraphT.recall(idx.search(q, 10, null, 100), t);
                Hit[] h = sg.search(q, 10, null, 100); rs += GraphT.recall(h, t);
                for (Hit x : h) if (del[(int) x.id()]) leak = true;
            }
            rm /= nq; rs /= nq;
            System.out.printf("  sq8=%b recall@10 mutable=%.4f sealed=%.4f; bytes mutable=%.1f MB sealed=%.1f MB%n", sq8, rm, rs, idx.bytesHeld() / 1e6, sg.bytesHeld() / 1e6);
            GraphT.check(!leak, "sealed never returns deleted ids");
            GraphT.check(rs >= 0.93, "sealed recall " + rs);
            GraphT.check(sg.bytesHeld() < idx.bytesHeld(), "sealed form is smaller");
            // soft delete on sealed
            float[] q0 = d[1]; Hit[] h = sg.search(q0, 1, null);
            GraphT.check(h[0].id() == 1, "sealed finds id 1");
            GraphT.check(sg.delete(1), "sealed delete returns true");
            GraphT.check(!sg.delete(1), "sealed second delete false");
            for (Hit x : sg.search(q0, 10, null)) GraphT.check(x.id() != 1, "sealed delete honoured");
            try { sg.add(1, q0); GraphT.check(false, "sealed add must throw"); } catch (UnsupportedOperationException ok) { }
            java.util.function.LongPredicate allow = id -> id % 100 == 0;
            Hit[] fh = sg.search(g.vec(3), 10, allow); boolean bad = fh.length != 10; for (Hit x : fh) if (x.id() % 100 != 0) bad = true;
            GraphT.check(!bad, "sealed filter works incl. exact fallback");
            // fromSealed round trip
            GraphIndex back = GraphIndex.fromSealed(sg);
            GraphT.check(back.size() == sg.size(), "fromSealed size");
            double rb = 0;
            for (int i = 0; i < nq; i++) { float[] q = g.vec(600000 + i); long[] t = GraphT.brute(d, del, q, 10, null, Metric.COSINE); rb += GraphT.recall(back.search(q, 10, null, 100), t); }
            rb /= nq;
            GraphT.check(rb >= 0.93, "re-inflated recall " + rb);
            back.add(5_000_000L, d[3]);
            GraphT.check(back.search(d[3], 5, null)[0].score() > 0.999f && back.size() == sg.size() + 1, "add after fromSealed works");
        }
    }

    static void testKernels() {
        java.util.Random r = new java.util.Random(1);
        for (int dim : new int[] { 1, 7, 8, 33, 128, 768, 1536 }) {
            float[] a = new float[dim], b = new float[dim]; byte[] c = new byte[dim];
            for (int i = 0; i < dim; i++) { a[i] = r.nextFloat() - .5f; b[i] = r.nextFloat() - .5f; c[i] = (byte) r.nextInt(256); }
            double dot = 0, l2 = 0, dq = 0; for (int i = 0; i < dim; i++) { dot += (double) a[i] * b[i]; l2 += (double) (a[i] - b[i]) * (a[i] - b[i]); dq += (double) a[i] * c[i]; }
            GraphT.check(Math.abs(Kern.dot(a, 0, b, 0, dim) - dot) < 1e-3 * (1 + Math.abs(dot)), "dot dim " + dim);
            GraphT.check(Math.abs(Kern.l2(a, 0, b, 0, dim) - l2) < 1e-3 * (1 + l2), "l2 dim " + dim);
            GraphT.check(Math.abs(Kern.dotQ8(a, c, 0, dim) - dq) < 1e-3 * (1 + Math.abs(dq)), "dotQ8 dim " + dim);
        }
    }

    /** descriptive timings: build (1 and 8 threads) and query latency at n=20000 and 100000 */
    static void bench(int n, int threads) throws Exception {
        GraphT.Gen g = new GraphT.Gen(DIM, 23, 100, 0.4f); float[][] d = data(g, n);
        GraphIndex idx = new GraphIndex(DIM, Metric.COSINE);
        long t0 = System.nanoTime();
        if (threads == 1) for (int i = 0; i < n; i++) idx.add(i, d[i]); else buildParallel(idx, d, threads);
        double s = (System.nanoTime() - t0) / 1e9;
        System.out.printf("  BENCH n=%d dim=%d threads=%d build=%.1fs (%.0f inserts/s) bytesHeld=%.1f MB%n", n, DIM, threads, s, n / s, idx.bytesHeld() / 1e6);
        int nq = 300; GraphT.Gen qg = new GraphT.Gen(DIM, 23, 100, 0.4f); float[][] qs = new float[nq][]; long[][] truth = new long[nq][];
        for (int i = 0; i < nq; i++) { qs[i] = qg.vec(2_000_000L + i); truth[i] = GraphT.brute(d, null, qs[i], 10, null, Metric.COSINE); }
        for (int ef : new int[] { 20, 50, 100, 200, 400 }) {
            double r = 0; long[] lat = new long[nq];
            for (int rep = 0; rep < 2; rep++) for (int i = 0; i < nq; i++) { long a = System.nanoTime(); Hit[] h = idx.search(qs[i], 10, null, ef); if (rep == 1) { lat[i] = System.nanoTime() - a; r += GraphT.recall(h, truth[i]); } }
            long[] so = GraphT.sorted(lat, nq);
            System.out.printf("  BENCH n=%d ef=%-4d recall@10=%.4f p50=%d us p99=%d us (single thread, warm, 2nd pass)%n", n, ef, r / nq, GraphT.pct(so, .5) / 1000, GraphT.pct(so, .99) / 1000);
        }
        long a = System.nanoTime(); SealedGraph sg = idx.seal(); double sealMs = (System.nanoTime() - a) / 1e6;
        System.out.printf("  BENCH sealed in %.0f ms, bytes %.1f MB (mutable %.1f MB)%n", sealMs, sg.bytesHeld() / 1e6, idx.bytesHeld() / 1e6);
        for (int ef : new int[] { 50, 100, 200 }) {
            double r = 0; long[] lat = new long[nq];
            for (int rep = 0; rep < 2; rep++) for (int i = 0; i < nq; i++) { long t = System.nanoTime(); Hit[] h = sg.search(qs[i], 10, null, ef); if (rep == 1) { lat[i] = System.nanoTime() - t; r += GraphT.recall(h, truth[i]); } }
            long[] so = GraphT.sorted(lat, nq);
            System.out.printf("  BENCH sealed n=%d ef=%-4d recall@10=%.4f p50=%d us p99=%d us%n", n, ef, r / nq, GraphT.pct(so, .5) / 1000, GraphT.pct(so, .99) / 1000);
        }
        GraphT.check(true, "bench ran");
    }

    public static void main(String[] a) throws Exception {
        String mode = a.length > 0 ? a[0] : "all";
        System.out.println("jdk " + System.getProperty("java.version") + " lanes=" + Kern.lanes());
        if (mode.equals("bench")) {
            GraphT.run("bench 20000 1 thread", () -> { try { bench(20000, 1); } catch (Exception e) { throw new RuntimeException(e); } });
            GraphT.run("bench 20000 8 threads", () -> { try { bench(20000, 8); } catch (Exception e) { throw new RuntimeException(e); } });
            GraphT.run("bench 100000 8 threads", () -> { try { bench(100000, 8); } catch (Exception e) { throw new RuntimeException(e); } });
        } else {
            GraphT.run("kernels", GraphTest::testKernels);
            GraphT.run("read-your-writes (single thread)", GraphTest::testReadYourWritesSingleThread);
            GraphT.run("replace", GraphTest::testReplace);
            GraphT.run("L2 and DOT", GraphTest::testL2Dot);
            GraphT.run("filter", GraphTest::testFilter);
            GraphT.run("delete + consolidate", GraphTest::testDeleteConsolidate);
            GraphT.run("seal + fromSealed", GraphTest::testSealAndRoundTrip);
            if (mode.equals("all")) GraphT.run("recall curve n=20000", GraphTest::testRecall);
        }
        System.out.printf("RESULT checks=%d failures=%d%n", GraphT.checks, GraphT.failures);
        if (GraphT.exitOnFailure) System.exit(GraphT.failures == 0 ? 0 : 1);
    }
}
