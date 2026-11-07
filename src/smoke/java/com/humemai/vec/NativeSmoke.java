package com.humemai.vec;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

/**
 * Smoke program for the GraalVM native-image job of the CI (see .github/workflows/native-image.yml). It is plain Java, no JUnit, and it
 * lives in the package of the library because it reads the kernel switch ({@code Kernels.VECTOR} and {@code Kernels.scalarReason}), which
 * is package-private. Maven does not compile it: the workflow compiles it with the library sources and builds the image from the classes.
 *
 * <p>Argument: {@code native} (the program runs as a native image) or {@code jvm} (the control run on a JVM without the incubator module,
 * the same checks except for the kernel reason). The program checks that the scalar kernels are in use and that no class of the Vector API
 * is present. Then it builds a flat index (all three metrics, fp32 and SQ8), an IVF index (fp32 and SQ8), a graph index (fp32 and SQ8)
 * and a tier (fp32 and SQ8: add, background seal, tail, delete, seal, replace ids, merge, delete after the merge), and after every step
 * it compares the answers with an exact scan over a model of the live rows.
 *
 * <p>How an answer is compared. The exact scan runs in double precision. A hit counts as correct when its exact score is at least the
 * exact k-th best score minus a tolerance, so a tie at the boundary does not fail a run. The tolerance is 1e-5 times the sum of the
 * absolute terms of the score (the one of {@code KernDiffTest}). Every hit must be a live id, once. For fp32 paths the reported score must
 * also match the exact score within the tolerance. Flat fp32 must have recall 1.0 (it is an exact scan); the other fp32 paths need at
 * least 0.9 and the SQ8 paths at least 0.8 (the bar of {@code NoIncubatorProbe}). The data is a fixed-seed Gaussian mixture, so a run
 * is the same every time. A failed check is printed and counted, the program goes on, and it exits 1 at the end when any check failed,
 * so one run shows every failure.
 */
public final class NativeSmoke {
    private NativeSmoke() {}

    static final int DIM = 32, N = 3000, QUERIES = 40, K = 10, CENTERS = 24;
    static final double REL_TOL = 1e-5;
    static final double MIN_RECALL = 0.9;
    static final double MIN_RECALL_SQ8 = 0.8;
    static final long ID0 = 100_000L, ID_STEP = 13L;

    static int failures;

    static void fail(String msg) {
        failures++;
        System.out.println("FAIL " + msg);
        System.out.flush();
    }

    static boolean check(boolean ok, String msg) {
        if (!ok) fail(msg);
        return ok;
    }

    // ---- the model and the exact scan ----

    static long idOf(int row) {
        return ID0 + ID_STEP * row;
    }

    /** the row of an id this program added, or -1 */
    static int rowOf(long id) {
        long d = id - ID0;
        if (d < 0 || d % ID_STEP != 0 || d / ID_STEP >= N) return -1;
        return (int) (d / ID_STEP);
    }

    /** the vector of every row, whether the row is live in the index under test, and the queries */
    static final class Data {
        final float[][] centers = new float[CENTERS][DIM];
        final float[][] vec = new float[N][];
        final boolean[] live = new boolean[N];
        final float[][] queries = new float[QUERIES][];
        final Random fresh = new Random(13);

        Data() {
            Random r = new Random(11);
            for (float[] c : centers) for (int j = 0; j < DIM; j++) c[j] = 2f * r.nextFloat() - 1f;
            for (int i = 0; i < N; i++) vec[i] = point(r);
            Random qr = new Random(12);
            for (int i = 0; i < QUERIES; i++) queries[i] = point(qr);
        }

        float[] point(Random r) {
            float[] c = centers[r.nextInt(CENTERS)];
            float[] x = new float[DIM];
            for (int j = 0; j < DIM; j++) x[j] = c[j] + 0.3f * (float) r.nextGaussian();
            return x;
        }

        int liveCount() {
            int n = 0;
            for (boolean b : live) if (b) n++;
            return n;
        }
    }

    /** the exact score of a vector against a query, in double precision; higher is better for every metric */
    static double exactScore(Metric metric, float[] v, float[] q) {
        double dot = 0, vv = 0, qq = 0, l2 = 0;
        for (int j = 0; j < DIM; j++) {
            double a = v[j], b = q[j];
            dot += a * b;
            vv += a * a;
            qq += b * b;
            l2 += (a - b) * (a - b);
        }
        if (metric == Metric.COSINE) return dot / Math.sqrt(vv * qq);
        if (metric == Metric.DOT) return dot;
        return -l2;
    }

    /** 1e-5 times the sum of the absolute terms of the score (for cosine the terms sum to at most 1) */
    static double tolerance(Metric metric, float[] v, float[] q, double score) {
        if (metric == Metric.COSINE) return REL_TOL;
        if (metric == Metric.DOT) {
            double s = 0;
            for (int j = 0; j < DIM; j++) s += Math.abs((double) v[j] * q[j]);
            return REL_TOL * Math.max(1.0, s);
        }
        return REL_TOL * Math.max(1.0, Math.abs(score));
    }

    interface Searcher {
        Hit[] search(float[] q);
    }

    /** runs every query, checks each answer against the exact scan of the live rows, prints the mean recall at K and returns it */
    static double compare(String label, Data d, Metric metric, Searcher s, boolean checkScores, double minRecall) {
        int liveN = d.liveCount();
        if (!check(liveN >= K, label + ": fewer than " + K + " live rows")) return 0;
        double sum = 0;
        for (float[] q : d.queries) {
            double[] sc = new double[N];
            double[] sorted = new double[liveN];
            int m = 0;
            for (int i = 0; i < N; i++) {
                if (!d.live[i]) continue;
                sc[i] = exactScore(metric, d.vec[i], q);
                sorted[m++] = sc[i];
            }
            Arrays.sort(sorted);
            double kth = sorted[liveN - K];
            Hit[] hits = s.search(q);
            check(hits.length == K, label + ": " + hits.length + " hits, expected " + K);
            Set<Long> seen = new HashSet<>();
            int good = 0;
            for (int h = 0; h < hits.length; h++) {
                Hit x = hits[h];
                int row = rowOf(x.id());
                if (!check(row >= 0, label + ": id " + x.id() + " was never added")) continue;
                if (!check(d.live[row], label + ": id " + x.id() + " is deleted but was returned")) continue;
                check(seen.add(x.id()), label + ": id " + x.id() + " is returned twice");
                double tol = tolerance(metric, d.vec[row], q, sc[row]);
                if (checkScores) {
                    check(Math.abs(x.score() - sc[row]) <= tol, label + ": id " + x.id() + " has score " + x.score() + ", exact " + sc[row] + ", tolerance " + tol);
                }
                if (h > 0) check(hits[h - 1].score() + tol >= x.score(), label + ": hits are not in score order at rank " + h);
                if (sc[row] >= kth - tol) good++;
            }
            sum += (double) good / K;
        }
        double recall = sum / d.queries.length;
        System.out.println(label + ": recall@" + K + " " + Math.round(recall * 10000) / 10000.0 + " over " + d.queries.length + " queries, " + liveN + " live rows");
        check(recall >= minRecall, label + ": recall@" + K + " " + recall + " is below " + minRecall);
        return recall;
    }

    static void addRow(VecIndex ix, Data d, int row) {
        ix.add(idOf(row), d.vec[row]);
        d.live[row] = true;
    }

    /** deletes every step-th live row from first on; returns how many */
    static int deleteRows(VecIndex ix, Data d, String label, int first, int step) {
        int n = 0;
        for (int i = first; i < N; i += step) {
            if (!d.live[i]) continue;
            check(ix.delete(idOf(i)), label + ": delete of the live id " + idOf(i) + " returned false");
            d.live[i] = false;
            n++;
        }
        return n;
    }

    static void checkSize(String label, VecIndex ix, Data d) {
        check(ix.size() == d.liveCount(), label + ": size " + ix.size() + ", expected " + d.liveCount());
    }

    // ---- phases ----

    static void kernels(String mode) {
        boolean nat = mode.equals("native");
        String image = System.getProperty("org.graalvm.nativeimage.imagecode");
        System.out.println("imagecode=" + image + ", Kernels.VECTOR=" + Kernels.VECTOR + ", scalarReason=" + Kernels.scalarReason);
        check(!Kernels.VECTOR, "the Vector API kernels are in use, the scalar twins must be");
        if (nat) {
            check("runtime".equals(image), "the program is not running as a native image (imagecode " + image + ")");
            check("native image".equals(Kernels.scalarReason), "scalarReason is " + Kernels.scalarReason + ", expected native image");
        } else {
            check(image == null, "the control run is inside a native image");
            check(Kernels.scalarReason != null && Kernels.scalarReason.startsWith("Vector API unavailable"), "scalarReason is " + Kernels.scalarReason + ", expected Vector API unavailable");
        }
        try {
            Class.forName("jdk.incubator.vector.FloatVector");
            fail("the class jdk.incubator.vector.FloatVector is present, the Vector API must not be");
        } catch (ClassNotFoundException | LinkageError e) {
            System.out.println("jdk.incubator.vector.FloatVector is not present: " + e.getClass().getSimpleName());
        }
    }

    static void flat() {
        for (Metric metric : Metric.values()) flatRun(metric, Quant.NONE, 1.0);
        flatRun(Metric.COSINE, Quant.SQ8, MIN_RECALL);
    }

    static void flatRun(Metric metric, Quant quant, double minRecall) {
        String label = "flat " + metric + " " + quant;
        Data d = new Data();
        FlatIndex f = new FlatIndex(DIM, metric, quant);
        try {
            Searcher s = q -> f.search(q, K, null);
            for (int i = 0; i < N; i++) addRow(f, d, i);
            checkSize(label, f, d);
            compare(label + " after adds", d, metric, s, true, minRecall);
            int gone = deleteRows(f, d, label, 0, 7);
            System.out.println(label + ": deleted " + gone);
            checkSize(label, f, d);
            compare(label + " after deletes", d, metric, s, true, minRecall);
            f.compact();
            checkSize(label, f, d);
            compare(label + " after compact", d, metric, s, true, minRecall);
        } finally {
            f.close();
        }
    }

    static void ivf() {
        ivfRun(Quant.NONE);
        ivfRun(Quant.SQ8);
    }

    static void ivfRun(Quant quant) {
        String label = "ivf " + quant;
        Data d = new Data();
        IvfIndex.Params p = new IvfIndex.Params();
        p.nlist = 16;
        p.nprobe = 8;
        p.trainAt = 500;
        p.quant = quant;
        p.threads = 2;
        p.seed = 7;
        IvfIndex ix = new IvfIndex(DIM, Metric.COSINE, p);
        try {
            Searcher s = q -> ix.search(q, K, null);
            for (int i = 0; i < N; i++) addRow(ix, d, i);
            ix.build();
            check(ix.isTrained(), label + ": not trained after build");
            System.out.println(label + ": nlist " + ix.nlist() + ", rebuilds " + ix.rebuildCount());
            checkSize(label, ix, d);
            compare(label + " after build", d, Metric.COSINE, s, quant == Quant.NONE, MIN_RECALL);
            int gone = deleteRows(ix, d, label, 0, 7);
            System.out.println(label + ": deleted " + gone);
            checkSize(label, ix, d);
            compare(label + " after deletes", d, Metric.COSINE, s, quant == Quant.NONE, MIN_RECALL);
            ix.compact();
            checkSize(label, ix, d);
            compare(label + " after compact", d, Metric.COSINE, s, quant == Quant.NONE, MIN_RECALL);
            check(ix.lastError() == null, label + ": lastError " + ix.lastError());
        } finally {
            ix.close();
        }
    }

    static void graph() {
        graphRun(false);
        graphRun(true);
    }

    static void graphRun(boolean sq8) {
        String label = "graph " + (sq8 ? "sq8" : "fp32");
        double minRecall = sq8 ? MIN_RECALL_SQ8 : MIN_RECALL;
        Data d = new Data();
        GraphIndex g = new GraphIndex(DIM, Metric.COSINE, GraphIndex.Params.defaults().withEfConstruction(100).withSq8(sq8));
        Searcher s = q -> g.search(q, K, null, 100);
        for (int i = 0; i < N; i++) addRow(g, d, i);
        checkSize(label, g, d);
        compare(label + " after adds", d, Metric.COSINE, s, !sq8, minRecall);
        int gone = deleteRows(g, d, label, 0, 7);
        System.out.println(label + ": deleted " + gone + ", tombstones " + g.tombstoneCount());
        checkSize(label, g, d);
        compare(label + " after deletes", d, Metric.COSINE, s, !sq8, minRecall);
        int removed = g.consolidate();
        System.out.println(label + ": consolidate removed " + removed + ", tombstones " + g.tombstoneCount());
        checkSize(label, g, d);
        compare(label + " after consolidate", d, Metric.COSINE, s, !sq8, minRecall);
    }

    static void tier() {
        tierRun(false);
        tierRun(true);
    }

    static void tierRun(boolean sq8) {
        String label = "tier " + (sq8 ? "sq8" : "fp32");
        double minRecall = sq8 ? MIN_RECALL_SQ8 : MIN_RECALL;
        boolean scores = !sq8;
        Data d = new Data();
        TieredIndex.Config cfg = new TieredIndex.Config(500, 0L, 300, 8, 4, 0.3, 0, 4, sq8 ? Quant.SQ8 : Quant.NONE,
                GraphIndex.Params.defaults().withEfConstruction(100).withSq8(sq8), 2);
        TieredIndex ix = new TieredIndex(DIM, Metric.COSINE, cfg);
        try {
            Searcher s = q -> ix.search(q, K, null, 200);

            // 1. adds: the background worker seals the tail every 500 vectors
            for (int i = 0; i < 2000; i++) addRow(ix, d, i);
            check(ix.awaitIdle(120_000), label + ": the tier did not become idle after the adds");
            describe(label, ix);
            check(ix.stats().segments() >= 1 && ix.stats().seals() >= 1, label + ": no sealed segment after 2000 adds");
            checkSize(label, ix, d);
            compare(label + " sealed segments", d, Metric.COSINE, s, scores, minRecall);

            // 2. a tail on top of the segments
            for (int i = 2000; i < 2300; i++) addRow(ix, d, i);
            check(ix.stats().tailSize() > 0, label + ": expected a non-empty tail");
            compare(label + " segments and tail", d, Metric.COSINE, s, scores, minRecall);

            // 3. deletes in the segments and in the tail
            int gone = deleteRows(ix, d, label, 0, 7);
            System.out.println(label + ": deleted " + gone);
            check(!ix.delete(idOf(0)), label + ": a second delete of id " + idOf(0) + " returned true");
            checkSize(label, ix, d);
            compare(label + " after deletes", d, Metric.COSINE, s, scores, minRecall);

            // 4. seal the tail now
            ix.sealNow();
            describe(label, ix);
            check(ix.stats().tailSize() == 0, label + ": the tail is not empty after sealNow");
            compare(label + " after sealNow", d, Metric.COSINE, s, scores, minRecall);

            // 5. id reuse: every 20th id gets a new vector (a live id is replaced, a deleted id comes back)
            for (int i = 0; i < 2300; i += 20) {
                d.vec[i] = d.point(d.fresh);
                addRow(ix, d, i);
            }
            checkSize(label, ix, d);
            compare(label + " after id reuse", d, Metric.COSINE, s, scores, minRecall);

            // 6. merge every segment into one
            ix.mergeNow();
            check(ix.awaitIdle(120_000), label + ": the tier did not become idle after mergeNow");
            describe(label, ix);
            check(ix.segmentCount() == 1, label + ": " + ix.segmentCount() + " segments after mergeNow, expected 1");
            check(ix.stats().merges() >= 1, label + ": no merge was counted");
            checkSize(label, ix, d);
            compare(label + " after merge", d, Metric.COSINE, s, scores, minRecall);

            // 7. deletes after the merge
            gone = deleteRows(ix, d, label, 3, 11);
            System.out.println(label + ": deleted " + gone);
            checkSize(label, ix, d);
            compare(label + " after deletes on the merged tier", d, Metric.COSINE, s, scores, minRecall);

            check(ix.lastError() == null, label + ": lastError " + ix.lastError());
            check(ix.stats().failures() == 0, label + ": " + ix.stats().failures() + " failed background tasks");
        } finally {
            ix.close();
        }
    }

    static void describe(String label, TieredIndex ix) {
        TieredIndex.Stats st = ix.stats();
        System.out.println(label + ": segments " + st.segments() + " " + Arrays.toString(st.segmentSizes()) + ", tail " + st.tailSize() + ", seals " + st.seals() + ", merges " + st.merges());
    }

    // ---- main ----

    static void phase(String name, Runnable r) {
        long t = System.nanoTime();
        System.out.println("== " + name);
        try {
            r.run();
        } catch (Throwable e) {
            failures++;
            System.out.println("FAIL " + name + ": " + e);
            e.printStackTrace(System.out);
        }
        System.out.println("== " + name + " done in " + (System.nanoTime() - t) / 1_000_000 + " ms");
        System.out.flush();
    }

    public static void main(String[] args) {
        final String mode = args.length > 0 ? args[0] : "";
        if (!mode.equals("native") && !mode.equals("jvm")) {
            System.out.println("usage: NativeSmoke native|jvm");
            System.exit(2);
        }
        long t0 = System.nanoTime();
        System.out.println("tiervec smoke program, mode " + mode + ", dim " + DIM + ", n " + N + ", k " + K + ", " + QUERIES + " queries, cpus " + Runtime.getRuntime().availableProcessors());
        System.out.println("covered: flat, ivf, graph, tier (seal, merge, delete, id reuse); not covered: segment files (checkpoint, open), rebuildFrom, threaded writers");
        phase("kernels", () -> kernels(mode));
        phase("flat", NativeSmoke::flat);
        phase("ivf", NativeSmoke::ivf);
        phase("graph", NativeSmoke::graph);
        phase("tier", NativeSmoke::tier);
        long ms = (System.nanoTime() - t0) / 1_000_000;
        if (failures > 0) {
            System.out.println("NATIVE-SMOKE FAILED: " + failures + " failed check(s), " + ms + " ms");
            System.exit(1);
        }
        System.out.println("NATIVE-SMOKE OK (" + mode + ") in " + ms + " ms");
        System.exit(0);
    }
}
