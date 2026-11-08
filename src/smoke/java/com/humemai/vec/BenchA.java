package com.humemai.vec;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Arrays;
import java.util.Random;

/**
 * Experiment round 5 (branch native-perf-experiment only; generated from one template by the experiment script, do not edit by hand): micro-benchmark of the
 * four VKernels kernels that use convertShape, arm64 with the 128-bit float species, in ONE process. Arguments: a label, then optionally variant names (default: all).
 * Kernels (RESULT kernel=): dotI8 (int8 x int8), dotF8 (float query x int8 codes, one accumulator), dotF8x4 (same, four accumulators), block16 (dimension-major
 * block of 16 vectors, one call computes 16 sums, vecs_per_call=16). Variants: cur, tx, cv (Vc8A, see its header), s1 and s4 and s8 (Sb, plain loops).
 * Dimensions 128, 100 and 768 (100 is not a multiple of any step, so the scalar tail runs), 4096 vectors of codes per pass. gbps counts code bytes only.
 * Before timing, EVERY variant is compared with the scalar twin (exact for dotI8; 1e-3 relative plus 1e-5 of the sum of absolute terms for the float kernels);
 * a failure prints MISMATCH name and exits 1, success prints SELFCHECK-OK. The timed loop then prints CHECK lines against a double reference.
 */
public final class BenchA {
    private BenchA() {}

    static final int N = 4096, WARM_PASSES = 3, TIMED_PASSES = 20, CHECKED = 64;
    static final long WARM_NS = 1_000_000_000L;
    static final int[] DIMS = {128, 100, 768};
    static final String[] KERNELS = {"dotI8", "dotF8", "dotF8x4", "block16"};
    static final String[] ALL = {"cur", "tx", "cv", "s1", "s4", "s8"};
    static double sink;

    static int oneI(String v, byte[] qb, byte[] d, int off, int dim) {
        switch (v) {
            case "cur": return Vc8A.dotI8cur(qb, 0, d, off, dim);
            case "tx": return Vc8A.dotI8tx(qb, 0, d, off, dim);
            case "cv": return Vc8A.dotI8cv(qb, 0, d, off, dim);
            case "s1": return Sb.dotI81(qb, 0, d, off, dim);
            case "s4": return Sb.dotI84(qb, 0, d, off, dim);
            case "s8": return Sb.dotI88(qb, 0, d, off, dim);
            default: throw new IllegalStateException(v);
        }
    }

    /** kernel 1 is dotF8, kernel 2 is dotF8x4 */
    static float oneF(String v, int kernel, float[] qf, byte[] d, int off, int dim) {
        switch (v) {
            case "cur": return kernel == 1 ? Vc8A.dotF8cur(qf, 0, d, off, dim) : Vc8A.dotF8x4cur(qf, 0, d, off, dim);
            case "tx": return kernel == 1 ? Vc8A.dotF8tx(qf, 0, d, off, dim) : Vc8A.dotF8x4tx(qf, 0, d, off, dim);
            case "cv": return kernel == 1 ? Vc8A.dotF8cv(qf, 0, d, off, dim) : Vc8A.dotF8x4cv(qf, 0, d, off, dim);
            case "s1": return kernel == 1 ? Sb.dotF81(qf, 0, d, off, dim) : Sb.dotF81(qf, 0, d, off, dim);
            case "s4": return kernel == 1 ? Sb.dotF84(qf, 0, d, off, dim) : Sb.dotF84(qf, 0, d, off, dim);
            case "s8": return kernel == 1 ? Sb.dotF88(qf, 0, d, off, dim) : Sb.dotF88(qf, 0, d, off, dim);
            default: throw new IllegalStateException(v);
        }
    }

    static void oneB(String v, float[] qf, byte[] d, int base, int dim, float[] out) {
        switch (v) {
            case "cur": Vc8A.block16cur(qf, 0, d, base, dim, out); return;
            case "tx": Vc8A.block16tx(qf, 0, d, base, dim, out); return;
            case "cv": Vc8A.block16cv(qf, 0, d, base, dim, out); return;
            case "s1": Sb.blk1(qf, 0, d, base, dim, out); return;
            case "s4": Sb.blk4(qf, 0, d, base, dim, out); return;
            case "s8": Sb.blk8(qf, 0, d, base, dim, out); return;
            default: throw new IllegalStateException(v);
        }
    }

    static double pass_cur_0(float[] qf, byte[] qb, byte[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += Vc8A.dotI8cur(qb, 0, data, i * dim, dim);
        return s;
    }

    static double pass_cur_1(float[] qf, byte[] qb, byte[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += Vc8A.dotF8cur(qf, 0, data, i * dim, dim);
        return s;
    }

    static double pass_cur_2(float[] qf, byte[] qb, byte[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += Vc8A.dotF8x4cur(qf, 0, data, i * dim, dim);
        return s;
    }

    static double pass_cur_3(float[] qf, byte[] qb, byte[] data, int dim) {
        double s = 0;
        float[] out = new float[16];
        for (int b = 0; b < N / 16; b++) {
            Vc8A.block16cur(qf, 0, data, b * dim * 16, dim, out);
            s += out[0] + out[15];
        }
        return s;
    }

    static double pass_tx_0(float[] qf, byte[] qb, byte[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += Vc8A.dotI8tx(qb, 0, data, i * dim, dim);
        return s;
    }

    static double pass_tx_1(float[] qf, byte[] qb, byte[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += Vc8A.dotF8tx(qf, 0, data, i * dim, dim);
        return s;
    }

    static double pass_tx_2(float[] qf, byte[] qb, byte[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += Vc8A.dotF8x4tx(qf, 0, data, i * dim, dim);
        return s;
    }

    static double pass_tx_3(float[] qf, byte[] qb, byte[] data, int dim) {
        double s = 0;
        float[] out = new float[16];
        for (int b = 0; b < N / 16; b++) {
            Vc8A.block16tx(qf, 0, data, b * dim * 16, dim, out);
            s += out[0] + out[15];
        }
        return s;
    }

    static double pass_cv_0(float[] qf, byte[] qb, byte[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += Vc8A.dotI8cv(qb, 0, data, i * dim, dim);
        return s;
    }

    static double pass_cv_1(float[] qf, byte[] qb, byte[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += Vc8A.dotF8cv(qf, 0, data, i * dim, dim);
        return s;
    }

    static double pass_cv_2(float[] qf, byte[] qb, byte[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += Vc8A.dotF8x4cv(qf, 0, data, i * dim, dim);
        return s;
    }

    static double pass_cv_3(float[] qf, byte[] qb, byte[] data, int dim) {
        double s = 0;
        float[] out = new float[16];
        for (int b = 0; b < N / 16; b++) {
            Vc8A.block16cv(qf, 0, data, b * dim * 16, dim, out);
            s += out[0] + out[15];
        }
        return s;
    }

    static double pass_s1_0(float[] qf, byte[] qb, byte[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += Sb.dotI81(qb, 0, data, i * dim, dim);
        return s;
    }

    static double pass_s1_1(float[] qf, byte[] qb, byte[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += Sb.dotF81(qf, 0, data, i * dim, dim);
        return s;
    }

    static double pass_s1_2(float[] qf, byte[] qb, byte[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += Sb.dotF81(qf, 0, data, i * dim, dim);
        return s;
    }

    static double pass_s1_3(float[] qf, byte[] qb, byte[] data, int dim) {
        double s = 0;
        float[] out = new float[16];
        for (int b = 0; b < N / 16; b++) {
            Sb.blk1(qf, 0, data, b * dim * 16, dim, out);
            s += out[0] + out[15];
        }
        return s;
    }

    static double pass_s4_0(float[] qf, byte[] qb, byte[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += Sb.dotI84(qb, 0, data, i * dim, dim);
        return s;
    }

    static double pass_s4_1(float[] qf, byte[] qb, byte[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += Sb.dotF84(qf, 0, data, i * dim, dim);
        return s;
    }

    static double pass_s4_2(float[] qf, byte[] qb, byte[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += Sb.dotF84(qf, 0, data, i * dim, dim);
        return s;
    }

    static double pass_s4_3(float[] qf, byte[] qb, byte[] data, int dim) {
        double s = 0;
        float[] out = new float[16];
        for (int b = 0; b < N / 16; b++) {
            Sb.blk4(qf, 0, data, b * dim * 16, dim, out);
            s += out[0] + out[15];
        }
        return s;
    }

    static double pass_s8_0(float[] qf, byte[] qb, byte[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += Sb.dotI88(qb, 0, data, i * dim, dim);
        return s;
    }

    static double pass_s8_1(float[] qf, byte[] qb, byte[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += Sb.dotF88(qf, 0, data, i * dim, dim);
        return s;
    }

    static double pass_s8_2(float[] qf, byte[] qb, byte[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += Sb.dotF88(qf, 0, data, i * dim, dim);
        return s;
    }

    static double pass_s8_3(float[] qf, byte[] qb, byte[] data, int dim) {
        double s = 0;
        float[] out = new float[16];
        for (int b = 0; b < N / 16; b++) {
            Sb.blk8(qf, 0, data, b * dim * 16, dim, out);
            s += out[0] + out[15];
        }
        return s;
    }

    static double pass(String v, int kernel, float[] qf, byte[] qb, byte[] data, int dim) {
        switch (v) {
            case "cur": return kernel == 0 ? pass_cur_0(qf, qb, data, dim) : kernel == 1 ? pass_cur_1(qf, qb, data, dim) : kernel == 2 ? pass_cur_2(qf, qb, data, dim) : pass_cur_3(qf, qb, data, dim);
            case "tx": return kernel == 0 ? pass_tx_0(qf, qb, data, dim) : kernel == 1 ? pass_tx_1(qf, qb, data, dim) : kernel == 2 ? pass_tx_2(qf, qb, data, dim) : pass_tx_3(qf, qb, data, dim);
            case "cv": return kernel == 0 ? pass_cv_0(qf, qb, data, dim) : kernel == 1 ? pass_cv_1(qf, qb, data, dim) : kernel == 2 ? pass_cv_2(qf, qb, data, dim) : pass_cv_3(qf, qb, data, dim);
            case "s1": return kernel == 0 ? pass_s1_0(qf, qb, data, dim) : kernel == 1 ? pass_s1_1(qf, qb, data, dim) : kernel == 2 ? pass_s1_2(qf, qb, data, dim) : pass_s1_3(qf, qb, data, dim);
            case "s4": return kernel == 0 ? pass_s4_0(qf, qb, data, dim) : kernel == 1 ? pass_s4_1(qf, qb, data, dim) : kernel == 2 ? pass_s4_2(qf, qb, data, dim) : pass_s4_3(qf, qb, data, dim);
            case "s8": return kernel == 0 ? pass_s8_0(qf, qb, data, dim) : kernel == 1 ? pass_s8_1(qf, qb, data, dim) : kernel == 2 ? pass_s8_2(qf, qb, data, dim) : pass_s8_3(qf, qb, data, dim);
            default: throw new IllegalStateException(v);
        }
    }

    static byte[] codes(Random rnd, int len) {
        byte[] r = new byte[len];
        for (int i = 0; i < len; i++) r[i] = (byte) (rnd.nextInt(255) - 127);
        return r;
    }

    static float[] floats(Random rnd, int len) {
        float[] r = new float[len];
        for (int i = 0; i < len; i++) r[i] = 2f * rnd.nextFloat() - 1f;
        return r;
    }

    static void fail(String v, int kernel, int dim, String got, String scalar) {
        System.out.println("MISMATCH " + v + " kernel=" + KERNELS[kernel] + " dim=" + dim + " got=" + got + " scalar=" + scalar);
        System.out.flush();
        System.exit(1);
    }

    /** every variant against Sb.dotI81, Sb.dotF81 and Sb.blk1 (the scalar twins) */
    static void selfCheck() {
        int checked = 0;
        for (int dim : DIMS) {
            Random rnd = new Random(7000 + dim);
            int m = 32;
            byte[] d = codes(rnd, m * dim);
            byte[] qb = codes(rnd, dim);
            float[] qf = floats(rnd, dim);
            float[] out = new float[16], ref = new float[16];
            for (String v : ALL) {
                if (v.equals("s1")) continue;
                for (int i = 0; i < m; i++) {
                    int got = oneI(v, qb, d, i * dim, dim);
                    int exp = Sb.dotI81(qb, 0, d, i * dim, dim);
                    if (got != exp) fail(v, 0, dim, "" + got, "" + exp);
                    checked++;
                }
                for (int kernel = 1; kernel <= 2; kernel++) {
                    for (int i = 0; i < m; i++) {
                        float exp = Sb.dotF81(qf, 0, d, i * dim, dim);
                        double sa = 0;
                        for (int j = 0; j < dim; j++) sa += Math.abs((double) qf[j] * d[i * dim + j]);
                        float got = oneF(v, kernel, qf, d, i * dim, dim);
                        double err = Math.abs((double) got - exp);
                        if (!(err <= 1e-3 * Math.abs(exp) + 1e-5 * sa)) fail(v, kernel, dim, "" + got, "" + exp);
                        checked++;
                    }
                }
                for (int b = 0; b < m / 16; b++) {
                    int base = b * dim * 16;
                    Sb.blk1(qf, 0, d, base, dim, ref);
                    oneB(v, qf, d, base, dim, out);
                    for (int l = 0; l < 16; l++) {
                        double sa = 0;
                        for (int j = 0; j < dim; j++) sa += Math.abs((double) qf[j] * d[base + j * 16 + l]);
                        double err = Math.abs((double) out[l] - ref[l]);
                        if (!(err <= 1e-3 * Math.abs(ref[l]) + 1e-5 * sa)) fail(v, 3, dim, "" + out[l], "" + ref[l]);
                        checked++;
                    }
                }
            }
        }
        System.out.println("SELFCHECK-OK variant_checks=" + checked);
    }

    /** CHECKED values of a variant: vector i for the dot kernels, lane e % 16 of block e / 16 for block16 */
    static void values(String v, int kernel, float[] qf, byte[] qb, byte[] d, int dim, double[] got) {
        float[] out = new float[16];
        if (kernel == 0) {
            for (int i = 0; i < CHECKED; i++) got[i] = oneI(v, qb, d, i * dim, dim);
        } else if (kernel < 3) {
            for (int i = 0; i < CHECKED; i++) got[i] = oneF(v, kernel, qf, d, i * dim, dim);
        } else {
            for (int b = 0; b < CHECKED / 16; b++) {
                oneB(v, qf, d, b * dim * 16, dim, out);
                for (int l = 0; l < 16; l++) got[b * 16 + l] = out[l];
            }
        }
    }

    static void reference(int kernel, float[] qf, byte[] qb, byte[] d, int dim, double[] ref, double[] refAbs) {
        for (int e = 0; e < CHECKED; e++) {
            double s = 0, sa = 0;
            for (int j = 0; j < dim; j++) {
                double t;
                if (kernel == 0) t = (double) qb[j] * d[e * dim + j];
                else if (kernel < 3) t = (double) qf[j] * d[e * dim + j];
                else t = (double) qf[j] * d[(e / 16) * dim * 16 + j * 16 + (e % 16)];
                s += t;
                sa += Math.abs(t);
            }
            ref[e] = s;
            refAbs[e] = sa;
        }
    }

    static String cpu() {
        try (BufferedReader r = new BufferedReader(new FileReader("/proc/cpuinfo"))) {
            String l;
            while ((l = r.readLine()) != null) if (l.startsWith("model name")) return l.substring(l.indexOf(':') + 1).trim().replace(' ', '_');
        } catch (Throwable t) {
            return "unknown";
        }
        return "unknown";
    }

    public static void main(String[] args) {
        String label = args.length > 0 ? args[0] : "none";
        String[] variants = args.length > 1 ? Arrays.copyOfRange(args, 1, args.length) : ALL;
        boolean isNative = System.getProperty("org.graalvm.nativeimage.imagecode") != null;
        System.out.println("ID label=" + label + " vm=" + System.getProperty("java.vm.name").replace(' ', '_') + " version=" + System.getProperty("java.version")
                + " vendor=" + System.getProperty("java.vendor").replace(' ', '_') + " native=" + isNative + " cpu=" + cpu() + " arch=" + System.getProperty("os.arch")
                + " os=" + System.getProperty("os.name").replace(' ', '_'));
        try {
            System.out.println("VC8 float=" + Vc8A.F128 + " int=" + Vc8A.I128 + " short=" + Vc8A.S128 + " b64=" + Vc8A.B64 + " b128=" + Vc8A.B128);
        } catch (Throwable t) {
            System.out.println("VC8-FAILED " + t);
            System.exit(1);
        }
        selfCheck();
        boolean allOk = true;
        for (int dim : DIMS) {
            Random rnd = new Random(1000 + dim);
            byte[] data = codes(rnd, N * dim);
            byte[] qb = codes(rnd, dim);
            float[] qf = floats(rnd, dim);
            for (int kernel = 0; kernel < KERNELS.length; kernel++) {
                double[] ref = new double[CHECKED], refAbs = new double[CHECKED], got = new double[CHECKED];
                reference(kernel, qf, qb, data, dim, ref, refAbs);
                int calls = kernel == 3 ? N / 16 : N;
                int vecs = kernel == 3 ? 16 : 1;
                for (String v : variants) {
                    values(v, kernel, qf, qb, data, dim, got);
                    double maxRel = 0;
                    for (int i = 0; i < CHECKED; i++) {
                        double err = Math.abs(got[i] - ref[i]) / (refAbs[i] == 0 ? 1 : refAbs[i]);
                        if (err > maxRel) maxRel = err;
                    }
                    boolean ok = maxRel <= 1e-5;
                    allOk &= ok;
                    System.out.println(String.format("CHECK%s label=%s variant=%s kernel=%s dim=%d max_rel_err=%.3e", ok ? "" : "-FAIL", label, v, KERNELS[kernel], dim, maxRel));
                    long w0 = System.nanoTime();
                    int w = 0;
                    while (w < WARM_PASSES || System.nanoTime() - w0 < WARM_NS) {
                        sink += pass(v, kernel, qf, qb, data, dim);
                        w++;
                    }
                    long[] t = new long[TIMED_PASSES];
                    double sum = 0;
                    for (int p = 0; p < TIMED_PASSES; p++) {
                        long t0 = System.nanoTime();
                        double r = pass(v, kernel, qf, qb, data, dim);
                        t[p] = System.nanoTime() - t0;
                        sum = r;
                        sink += r;
                    }
                    Arrays.sort(t);
                    double med = (t[TIMED_PASSES / 2 - 1] + t[TIMED_PASSES / 2]) / 2.0;
                    double gbps = (double) N * dim / med;
                    System.out.println(String.format("RESULT label=%s variant=%s kernel=%s dim=%d warm_passes=%d median_ns_per_call=%.2f min_ns_per_call=%.2f gbps=%.3f sum=%.9e vecs_per_call=%d",
                            label, v, KERNELS[kernel], dim, w, med / calls, t[0] / (double) calls, gbps, sum, vecs));
                    System.out.flush();
                }
            }
        }
        System.out.println("DONE label=" + label + " all_checks_ok=" + allOk + " sink=" + sink);
    }
}
