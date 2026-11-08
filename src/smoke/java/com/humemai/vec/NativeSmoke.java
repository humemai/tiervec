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
 * <p>Argument, the mode (two image variants and two control runs on a JVM):
 * <ul>
 * <li>{@code native}: the scalar image, built without the module jdk.incubator.vector. The program checks that the scalar kernels are in use
 * and that no class of the Vector API is present.</li>
 * <li>{@code native-vector}: the vector image, built with {@code --add-modules jdk.incubator.vector -H:+VectorAPISupport} and the properties file
 * of the jar on the class path. The program checks that {@code Kernels.VECTOR} is true in the image, that the species of {@code NKernels} was
 * fixed at image build time and has the lanes of the architecture, that {@code Kernels.JVM_VECTOR} is false and {@code Kernels.NATIVE_VECTOR} is true, and that
 * every kernel except hamming (fp32 dot and L2, int8 dot, the SQ8 dots, block16) returns the bits of its {@code NKernels} method and agrees with its scalar twin.</li>
 * <li>{@code jvm}: the control run on a JVM without the module (the same checks as {@code native} except for the kernel reason).</li>
 * <li>{@code jvm-vector}: the control run on a JVM with the module (the Vector API kernels of the JVM, no image).</li>
 * </ul>
 * Every mode then prints informational {@code SPEED} lines with the median time of 1,000,000 calls of {@code Kernels.dot} on 768 dimensions and of the scalar
 * twin, and the same for {@code dotI8}, {@code dotF8}, {@code dotF8x4} and {@code block16} (per call on a block of 16 vectors); a line is never a pass or fail bar.
 * Then the program builds a flat index (all three metrics, fp32, and SQ8 with each of the four SQ8 kernels), an IVF index (fp32 and SQ8), a graph index (fp32 and SQ8)
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
        boolean nat = mode.startsWith("native");
        boolean vec = mode.endsWith("-vector");
        String image = System.getProperty("org.graalvm.nativeimage.imagecode");
        System.out.println("imagecode=" + image + ", Kernels.VECTOR=" + Kernels.VECTOR + ", Kernels.JVM_VECTOR=" + Kernels.JVM_VECTOR + ", scalarReason=" + Kernels.scalarReason + ", os.arch=" + System.getProperty("os.arch"));
        if (nat) check("runtime".equals(image), "the program is not running as a native image (imagecode " + image + ")");
        else check(image == null, "the control run is inside a native image");
        if (!vec) {
            check(!Kernels.VECTOR, "the Vector API kernels are in use, the scalar twins must be");
            if (nat) {
                check(Kernels.scalarReason != null && Kernels.scalarReason.startsWith("native image without the Vector API"), "scalarReason is " + Kernels.scalarReason + ", expected native image without the Vector API");
            } else {
                check(Kernels.scalarReason != null && Kernels.scalarReason.startsWith("Vector API unavailable"), "scalarReason is " + Kernels.scalarReason + ", expected Vector API unavailable");
            }
            try {
                Class.forName("jdk.incubator.vector.FloatVector");
                fail("the class jdk.incubator.vector.FloatVector is present, the Vector API must not be");
            } catch (ClassNotFoundException | LinkageError e) {
                System.out.println("jdk.incubator.vector.FloatVector is not present: " + e.getClass().getSimpleName());
            }
            return;
        }
        check(Kernels.VECTOR, "Kernels.VECTOR is false, scalarReason " + Kernels.scalarReason);
        check(Kernels.scalarReason == null, "scalarReason is " + Kernels.scalarReason + ", expected none");
        try {
            Class.forName("jdk.incubator.vector.FloatVector");
            System.out.println("jdk.incubator.vector.FloatVector is present");
        } catch (ClassNotFoundException | LinkageError e) {
            fail("the class jdk.incubator.vector.FloatVector is not present: " + e);
        }
        if (nat) {
            String arch = System.getProperty("os.arch", "");
            int lanes = arch.equals("aarch64") || arch.equals("arm64") ? 4 : 8;
            check(!Kernels.JVM_VECTOR, "Kernels.JVM_VECTOR is true in an image, the kernels of VKernels must not be in use there");
            check(Kernels.NATIVE_VECTOR, "Kernels.NATIVE_VECTOR is false in an image with the Vector API, the kernels of NKernels must be in use");
            check("buildtime".equals(NKernels.INIT_IMAGECODE), "NKernels was initialized at " + NKernels.INIT_IMAGECODE + ", expected buildtime (is native-image.properties on the class path?)");
            check(NKernels.L == lanes, "NKernels has " + NKernels.L + " lanes, expected " + lanes + " for os.arch " + arch);
            check(NKernels.WIDE == (lanes == 8), "NKernels.WIDE is " + NKernels.WIDE + " for os.arch " + arch);
            System.out.println("NKernels: species " + NKernels.S + ", lanes " + NKernels.L + ", wide " + NKernels.WIDE + ", initialized at " + NKernels.INIT_IMAGECODE);
        } else {
            check(Kernels.JVM_VECTOR, "Kernels.JVM_VECTOR is false on a JVM with the module");
            check(!Kernels.NATIVE_VECTOR, "Kernels.NATIVE_VECTOR is true on a JVM");
        }
    }

    /**
     * The kernels in use against their scalar twins on random data, for the lengths around the lane widths and the 768 and 770 of the speed lines. In an
     * image (vector) each result must also be the exact bits of the NKernels method, which proves that Kernels routes to NKernels there. The integer
     * kernel must agree exactly; a float kernel within 1e-4 times the sum of the absolute terms plus 1e-6.
     */
    static void kernelsAgree(String mode) {
        boolean nat = mode.equals("native-vector");
        Random r = new Random(31);
        int[] dims = {1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 128, 768, 770};
        for (int n : dims) {
            String m = " n=" + n;
            float[] q = new float[n];
            byte[] a = new byte[n + 3], b = new byte[n + 5], blk = new byte[n * 16 + 7];
            float[] c = new float[n], d = new float[n];
            for (int i = 0; i < n; i++) {
                q[i] = r.nextFloat() * 2f - 1f;
                c[i] = r.nextFloat() * 2f - 1f;
                d[i] = r.nextFloat() * 2f - 1f;
            }
            for (int i = 0; i < a.length; i++) a[i] = (byte) (r.nextInt(255) - 127);
            for (int i = 0; i < b.length; i++) b[i] = (byte) (r.nextInt(255) - 127);
            for (int i = 0; i < blk.length; i++) blk[i] = (byte) (r.nextInt(255) - 127);
            check(Kernels.dotI8(a, 3, b, 5, n) == Kernels.dotI8Scalar(a, 3, b, 5, n), "dotI8 against the scalar twin" + m);
            if (nat) check(Kernels.dotI8(a, 3, b, 5, n) == NKernels.dotI8(a, 3, b, 5, n), "dotI8 is not NKernels.dotI8" + m);
            double dq = 0, dqAbs = 0, dd = 0, ddAbs = 0;
            for (int i = 0; i < n; i++) {
                dq += (double) q[i] * a[3 + i];
                dqAbs += Math.abs((double) q[i] * a[3 + i]);
                dd += (double) c[i] * d[i];
                ddAbs += Math.abs((double) c[i] * d[i]);
            }
            float f8 = Kernels.dotF8(q, 0, a, 3, n), f84 = Kernels.dotF8x4(q, 0, a, 3, n), f8s = Kernels.dotF8Scalar(q, 0, a, 3, n);
            float kd = Kernels.dot(c, 0, d, 0, n), kdS = Kernels.dotScalar(c, 0, d, 0, n);
            double tolQ = 1e-4 * dqAbs + 1e-6, tolD = 1e-4 * ddAbs + 1e-6;
            check(Math.abs(f8 - f8s) <= tolQ && Math.abs(f8 - dq) <= tolQ, "dotF8 against the scalar twin and the exact value" + m);
            check(Math.abs(f84 - f8s) <= tolQ && Math.abs(f84 - dq) <= tolQ, "dotF8x4 against the scalar twin and the exact value" + m);
            check(Math.abs(kd - kdS) <= tolD && Math.abs(kd - dd) <= tolD, "dot against the scalar twin and the exact value" + m);
            float[] out = new float[16], outS = new float[16];
            Kernels.block16(q, 0, blk, 5, n, out);
            Kernels.block16Scalar(q, 0, blk, 5, n, outS);
            for (int l = 0; l < 16; l++) {
                double ex = 0, exAbs = 0;
                for (int j = 0; j < n; j++) {
                    ex += (double) q[j] * blk[5 + j * 16 + l];
                    exAbs += Math.abs((double) q[j] * blk[5 + j * 16 + l]);
                }
                double tol = 1e-4 * exAbs + 1e-6;
                check(Math.abs(out[l] - outS[l]) <= tol && Math.abs(out[l] - ex) <= tol, "block16 lane " + l + " against the scalar twin and the exact value" + m);
            }
            if (nat) {
                check(Float.floatToIntBits(f8) == Float.floatToIntBits(NKernels.dotF8(q, 0, a, 3, n)), "dotF8 is not NKernels.dotF8" + m);
                check(Float.floatToIntBits(f84) == Float.floatToIntBits(NKernels.dotF8x4(q, 0, a, 3, n)), "dotF8x4 is not NKernels.dotF8x4" + m);
                check(Float.floatToIntBits(kd) == Float.floatToIntBits(NKernels.dot(c, 0, d, 0, n)), "dot is not NKernels.dot" + m);
                float[] outN = new float[16];
                NKernels.block16(q, 0, blk, 5, n, outN);
                for (int l = 0; l < 16; l++) check(Float.floatToIntBits(out[l]) == Float.floatToIntBits(outN[l]), "block16 lane " + l + " is not NKernels.block16" + m);
                check(Float.floatToIntBits(Kern.dotQ8(q, a, 3, n)) == Float.floatToIntBits(NKernels.dotF8(q, 0, a, 3, n)), "Kern.dotQ8 is not NKernels.dotF8" + m);
            }
        }
        System.out.println("kernels agree with their scalar twins at " + dims.length + " lengths" + (nat ? ", and are the NKernels methods" : ""));
    }

    /**
     * Informational: median ns per call of Kernels.dot and of the scalar twin on 768 dimensions, over {@code REPS} runs of {@code CALLS} calls each (one
     * run to warm up first). The calls rotate over a few pairs of vectors, and every result goes into a sum that is printed, so the loop stays alive.
     * Never a pass or fail bar: the line shows in the CI log what the kernels of this image or JVM cost.
     */
    static void speed(String mode) {
        final int dim = 768, pairs = 16, calls = 1_000_000, reps = 5;
        Random r = new Random(5);
        float[][] a = new float[pairs][dim], b = new float[pairs][dim];
        for (int p = 0; p < pairs; p++) {
            for (int j = 0; j < dim; j++) {
                a[p][j] = r.nextFloat() * 2f - 1f;
                b[p][j] = r.nextFloat() * 2f - 1f;
            }
        }
        double[] kern = new double[reps], scal = new double[reps];
        float sink = 0f;
        for (int rep = -1; rep < reps; rep++) {
            long t = System.nanoTime();
            for (int c = 0; c < calls; c++) sink += Kernels.dot(a[c % pairs], 0, b[c % pairs], 0, dim);
            double kn = (System.nanoTime() - t) / (double) calls;
            t = System.nanoTime();
            for (int c = 0; c < calls; c++) sink += Kernels.dotScalar(a[c % pairs], 0, b[c % pairs], 0, dim);
            double sn = (System.nanoTime() - t) / (double) calls;
            if (rep >= 0) {
                kern[rep] = kn;
                scal[rep] = sn;
            }
        }
        Arrays.sort(kern);
        Arrays.sort(scal);
        double mk = kern[reps / 2], ms = scal[reps / 2];
        System.out.println("SPEED (informational) mode " + mode + ", dot " + dim + " dims, median of " + reps + " runs of " + calls + " calls: Kernels.dot "
                + Math.round(mk * 10) / 10.0 + " ns, scalar twin " + Math.round(ms * 10) / 10.0 + " ns, scalar twin / Kernels.dot " + Math.round(ms / mk * 100) / 100.0 + ", VECTOR "
                + Kernels.VECTOR + ", sink " + sink);
    }

    /**
     * Informational: one SPEED line each for dotI8, dotF8, dotF8x4 and block16 (per call on a block of 16 vectors) on 768 dimensions: median ns per
     * call of the kernel in use and of its scalar twin, over {@code REPS} runs of {@code CALLS} calls (one run to warm up first). Never a bar.
     */
    static void speedQuant(String mode) {
        final int dim = 768, pairs = 16, calls = 1_000_000, reps = 5;
        Random r = new Random(6);
        byte[][] ca = new byte[pairs][dim], cb = new byte[pairs][dim];
        byte[][] blocks = new byte[pairs][dim * 16];
        float[][] q = new float[pairs][dim];
        for (int p = 0; p < pairs; p++) {
            for (int j = 0; j < dim; j++) {
                ca[p][j] = (byte) (r.nextInt(255) - 127);
                cb[p][j] = (byte) (r.nextInt(255) - 127);
                q[p][j] = r.nextFloat() * 2f - 1f;
            }
            for (int j = 0; j < blocks[p].length; j++) blocks[p][j] = (byte) (r.nextInt(255) - 127);
        }
        float[] out = new float[16];
        double[][] t = new double[8][reps];
        float sink = 0f;
        for (int rep = -1; rep < reps; rep++) {
            double[] ns = new double[8];
            long t0 = System.nanoTime();
            for (int c = 0; c < calls; c++) sink += Kernels.dotI8(ca[c % pairs], 0, cb[c % pairs], 0, dim);
            ns[0] = (System.nanoTime() - t0) / (double) calls;
            t0 = System.nanoTime();
            for (int c = 0; c < calls; c++) sink += Kernels.dotI8Scalar(ca[c % pairs], 0, cb[c % pairs], 0, dim);
            ns[1] = (System.nanoTime() - t0) / (double) calls;
            t0 = System.nanoTime();
            for (int c = 0; c < calls; c++) sink += Kernels.dotF8(q[c % pairs], 0, cb[c % pairs], 0, dim);
            ns[2] = (System.nanoTime() - t0) / (double) calls;
            t0 = System.nanoTime();
            for (int c = 0; c < calls; c++) sink += Kernels.dotF8Scalar(q[c % pairs], 0, cb[c % pairs], 0, dim);
            ns[3] = (System.nanoTime() - t0) / (double) calls;
            t0 = System.nanoTime();
            for (int c = 0; c < calls; c++) sink += Kernels.dotF8x4(q[c % pairs], 0, cb[c % pairs], 0, dim);
            ns[4] = (System.nanoTime() - t0) / (double) calls;
            t0 = System.nanoTime();
            for (int c = 0; c < calls; c++) sink += Kernels.dotF8Scalar(q[c % pairs], 0, cb[c % pairs], 0, dim);
            ns[5] = (System.nanoTime() - t0) / (double) calls;
            t0 = System.nanoTime();
            for (int c = 0; c < calls; c++) {
                Kernels.block16(q[c % pairs], 0, blocks[c % pairs], 0, dim, out);
                sink += out[c & 15];
            }
            ns[6] = (System.nanoTime() - t0) / (double) calls;
            t0 = System.nanoTime();
            for (int c = 0; c < calls; c++) {
                Kernels.block16Scalar(q[c % pairs], 0, blocks[c % pairs], 0, dim, out);
                sink += out[c & 15];
            }
            ns[7] = (System.nanoTime() - t0) / (double) calls;
            if (rep >= 0) for (int i = 0; i < 8; i++) t[i][rep] = ns[i];
        }
        String[] names = {"dotI8", "dotF8", "dotF8x4", "block16"};
        for (int k = 0; k < 4; k++) {
            double[] kern = t[k * 2].clone(), scal = t[k * 2 + 1].clone();
            Arrays.sort(kern);
            Arrays.sort(scal);
            double mk = kern[reps / 2], ms = scal[reps / 2];
            System.out.println("SPEED (informational) mode " + mode + ", " + names[k] + " " + dim + " dims" + (k == 3 ? " (per call, a block of 16 vectors)" : "")
                    + ", median of " + reps + " runs of " + calls + " calls: Kernels." + names[k] + " " + Math.round(mk * 10) / 10.0 + " ns, scalar twin "
                    + Math.round(ms * 10) / 10.0 + " ns, scalar twin / Kernels." + names[k] + " " + Math.round(ms / mk * 100) / 100.0 + ", VECTOR " + Kernels.VECTOR
                    + ", sink " + sink);
        }
    }

    static void flat() {
        for (Metric metric : Metric.values()) flatRun(metric, Quant.NONE, 1.0, Sq8Kernel.DEFAULT);
        // every SQ8 kernel of the flat index: I8 (dotI8), F8 (dotF8), F8X4 (dotF8x4) and BLOCK (block16)
        for (Sq8Kernel k : Sq8Kernel.values()) flatRun(Metric.COSINE, Quant.SQ8, MIN_RECALL, k);
    }

    static void flatRun(Metric metric, Quant quant, double minRecall, Sq8Kernel kernel) {
        String label = "flat " + metric + " " + quant + (quant == Quant.SQ8 ? " " + kernel : "");
        Data d = new Data();
        FlatIndex f = new FlatIndex(DIM, metric, quant, 0, kernel);
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
        if (!mode.equals("native") && !mode.equals("native-vector") && !mode.equals("jvm") && !mode.equals("jvm-vector")) {
            System.out.println("usage: NativeSmoke native|native-vector|jvm|jvm-vector");
            System.exit(2);
        }
        long t0 = System.nanoTime();
        System.out.println("tiervec smoke program, mode " + mode + ", dim " + DIM + ", n " + N + ", k " + K + ", " + QUERIES + " queries, cpus " + Runtime.getRuntime().availableProcessors());
        System.out.println("covered: flat, ivf, graph, tier (seal, merge, delete, id reuse); not covered: segment files (checkpoint, open), rebuildFrom, threaded writers");
        phase("kernels", () -> kernels(mode));
        phase("speed", () -> {
            speed(mode);
            speedQuant(mode);
        });
        phase("kernel agreement", () -> {
            if (mode.endsWith("-vector")) kernelsAgree(mode);
        });
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
