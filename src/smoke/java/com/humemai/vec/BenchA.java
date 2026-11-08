package com.humemai.vec;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Arrays;
import java.util.Random;

/**
 * Experiment round 3 (branch native-perf-experiment only; generated from one template, do not edit by hand): kernel micro-benchmark with
 * several variants of dot and l2sq in ONE process. Arguments: a label, then optionally variant names (default: all of this class).
 * Variants: vk (tiervec VKernels, preferred species), t64, t128, t256, t512 and tp (VcA, static final species, two accumulators, reduceLanes), s1 and s4 and s8 (Sc, plain loops with 1, 4 and 8 accumulators).
 * Round 4 variants (VcA, 128-bit species): L128 (horizontal sum as a pairwise tree over v.lane(i)), A128 (v.intoArray into a per-call float[], then the tree), M128 (reduceLanes, masked tail).
 * Dimensions 128, 100, 768 and 770 (100 and 770 are not multiples of the lane count, so the tail runs), 4096 vectors per pass. Before timing, the first 64 results of every variant are compared with a double
 * reference (tolerance 1e-5 times the sum of absolute terms, as KernDiffTest); a failure prints CHECK-FAIL.
 */
public final class BenchA {
    private BenchA() {}

    static final int N = 4096, WARM_PASSES = 3, TIMED_PASSES = 20, CHECKED = 64;
    static final long WARM_NS = 1_000_000_000L;
    static final int[] DIMS = {128, 100, 768, 770};
    static final String[] NAMES = {"dot", "l2sq"};
    static final String[] ALL = {"vk", "t64", "t128", "t256", "t512", "tp", "s1", "s4", "s8", "L128", "A128", "M128"};
    static final String[] NEWV = {"L128", "A128", "M128"};
    static double sink;

    static float one(String v, int kernel, float[] q, float[] d, int off, int dim) {
        switch (v) {
            case "vk": return kernel == 0 ? VKernels.dot(q, 0, d, off, dim) : VKernels.l2sq(q, 0, d, off, dim);
            case "t64": return kernel == 0 ? VcA.dot64(q, 0, d, off, dim) : VcA.l2sq64(q, 0, d, off, dim);
            case "t128": return kernel == 0 ? VcA.dot128(q, 0, d, off, dim) : VcA.l2sq128(q, 0, d, off, dim);
            case "t256": return kernel == 0 ? VcA.dot256(q, 0, d, off, dim) : VcA.l2sq256(q, 0, d, off, dim);
            case "t512": return kernel == 0 ? VcA.dot512(q, 0, d, off, dim) : VcA.l2sq512(q, 0, d, off, dim);
            case "tp": return kernel == 0 ? VcA.dotp(q, 0, d, off, dim) : VcA.l2sqp(q, 0, d, off, dim);
            case "s1": return kernel == 0 ? Sc.dot1(q, 0, d, off, dim) : Sc.l2sq1(q, 0, d, off, dim);
            case "s4": return kernel == 0 ? Sc.dot4(q, 0, d, off, dim) : Sc.l2sq4(q, 0, d, off, dim);
            case "s8": return kernel == 0 ? Sc.dot8(q, 0, d, off, dim) : Sc.l2sq8(q, 0, d, off, dim);
            case "L128": return kernel == 0 ? VcA.dotL128(q, 0, d, off, dim) : VcA.l2sqL128(q, 0, d, off, dim);
            case "A128": return kernel == 0 ? VcA.dotA128(q, 0, d, off, dim) : VcA.l2sqA128(q, 0, d, off, dim);
            case "M128": return kernel == 0 ? VcA.dotM128(q, 0, d, off, dim) : VcA.l2sqM128(q, 0, d, off, dim);
            default: throw new IllegalStateException(v);
        }
    }

    static double pass_vk_0(float[] q, float[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += VKernels.dot(q, 0, data, i * dim, dim);
        return s;
    }

    static double pass_vk_1(float[] q, float[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += VKernels.l2sq(q, 0, data, i * dim, dim);
        return s;
    }

    static double pass_t64_0(float[] q, float[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += VcA.dot64(q, 0, data, i * dim, dim);
        return s;
    }

    static double pass_t64_1(float[] q, float[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += VcA.l2sq64(q, 0, data, i * dim, dim);
        return s;
    }

    static double pass_t128_0(float[] q, float[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += VcA.dot128(q, 0, data, i * dim, dim);
        return s;
    }

    static double pass_t128_1(float[] q, float[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += VcA.l2sq128(q, 0, data, i * dim, dim);
        return s;
    }

    static double pass_t256_0(float[] q, float[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += VcA.dot256(q, 0, data, i * dim, dim);
        return s;
    }

    static double pass_t256_1(float[] q, float[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += VcA.l2sq256(q, 0, data, i * dim, dim);
        return s;
    }

    static double pass_t512_0(float[] q, float[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += VcA.dot512(q, 0, data, i * dim, dim);
        return s;
    }

    static double pass_t512_1(float[] q, float[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += VcA.l2sq512(q, 0, data, i * dim, dim);
        return s;
    }

    static double pass_tp_0(float[] q, float[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += VcA.dotp(q, 0, data, i * dim, dim);
        return s;
    }

    static double pass_tp_1(float[] q, float[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += VcA.l2sqp(q, 0, data, i * dim, dim);
        return s;
    }

    static double pass_s1_0(float[] q, float[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += Sc.dot1(q, 0, data, i * dim, dim);
        return s;
    }

    static double pass_s1_1(float[] q, float[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += Sc.l2sq1(q, 0, data, i * dim, dim);
        return s;
    }

    static double pass_s4_0(float[] q, float[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += Sc.dot4(q, 0, data, i * dim, dim);
        return s;
    }

    static double pass_s4_1(float[] q, float[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += Sc.l2sq4(q, 0, data, i * dim, dim);
        return s;
    }

    static double pass_s8_0(float[] q, float[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += Sc.dot8(q, 0, data, i * dim, dim);
        return s;
    }

    static double pass_s8_1(float[] q, float[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += Sc.l2sq8(q, 0, data, i * dim, dim);
        return s;
    }

    static double pass_L128_0(float[] q, float[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += VcA.dotL128(q, 0, data, i * dim, dim);
        return s;
    }

    static double pass_L128_1(float[] q, float[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += VcA.l2sqL128(q, 0, data, i * dim, dim);
        return s;
    }

    static double pass_A128_0(float[] q, float[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += VcA.dotA128(q, 0, data, i * dim, dim);
        return s;
    }

    static double pass_A128_1(float[] q, float[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += VcA.l2sqA128(q, 0, data, i * dim, dim);
        return s;
    }

    static double pass_M128_0(float[] q, float[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += VcA.dotM128(q, 0, data, i * dim, dim);
        return s;
    }

    static double pass_M128_1(float[] q, float[] data, int dim) {
        double s = 0;
        for (int i = 0; i < N; i++) s += VcA.l2sqM128(q, 0, data, i * dim, dim);
        return s;
    }

    static double pass(String v, int kernel, float[] q, float[] data, int dim) {
        switch (v) {
            case "vk": return kernel == 0 ? pass_vk_0(q, data, dim) : pass_vk_1(q, data, dim);
            case "t64": return kernel == 0 ? pass_t64_0(q, data, dim) : pass_t64_1(q, data, dim);
            case "t128": return kernel == 0 ? pass_t128_0(q, data, dim) : pass_t128_1(q, data, dim);
            case "t256": return kernel == 0 ? pass_t256_0(q, data, dim) : pass_t256_1(q, data, dim);
            case "t512": return kernel == 0 ? pass_t512_0(q, data, dim) : pass_t512_1(q, data, dim);
            case "tp": return kernel == 0 ? pass_tp_0(q, data, dim) : pass_tp_1(q, data, dim);
            case "s1": return kernel == 0 ? pass_s1_0(q, data, dim) : pass_s1_1(q, data, dim);
            case "s4": return kernel == 0 ? pass_s4_0(q, data, dim) : pass_s4_1(q, data, dim);
            case "s8": return kernel == 0 ? pass_s8_0(q, data, dim) : pass_s8_1(q, data, dim);
            case "L128": return kernel == 0 ? pass_L128_0(q, data, dim) : pass_L128_1(q, data, dim);
            case "A128": return kernel == 0 ? pass_A128_0(q, data, dim) : pass_A128_1(q, data, dim);
            case "M128": return kernel == 0 ? pass_M128_0(q, data, dim) : pass_M128_1(q, data, dim);
            default: throw new IllegalStateException(v);
        }
    }

    static boolean isNew(String v) {
        for (String n : NEWV) if (n.equals(v)) return true;
        return false;
    }

    /** round 4: every new variant must agree with the scalar loop (Sc.dot1 and Sc.l2sq1) within 1e-3 relative (plus 1e-5 of the sum of absolute terms as float noise floor); prints MISMATCH name and exits nonzero otherwise */
    static void selfCheck(String[] variants) {
        int checked = 0;
        for (int dim : DIMS) {
            Random rnd = new Random(7000 + dim);
            int m = 16;
            float[] q = new float[dim];
            float[] d = new float[m * dim];
            for (int i = 0; i < d.length; i++) d[i] = 2f * rnd.nextFloat() - 1f;
            for (int i = 0; i < dim; i++) q[i] = 2f * rnd.nextFloat() - 1f;
            for (String v : variants) {
                if (!isNew(v)) continue;
                for (int kernel = 0; kernel < 2; kernel++) {
                    for (int i = 0; i < m; i++) {
                        float ref = kernel == 0 ? Sc.dot1(q, 0, d, i * dim, dim) : Sc.l2sq1(q, 0, d, i * dim, dim);
                        double sa = 0;
                        for (int j = 0; j < dim; j++) {
                            double a = q[j], b = d[i * dim + j];
                            sa += Math.abs(kernel == 0 ? a * b : (a - b) * (a - b));
                        }
                        float got = one(v, kernel, q, d, i * dim, dim);
                        double err = Math.abs((double) got - ref);
                        if (!(err <= 1e-3 * Math.abs(ref) + 1e-5 * sa)) {
                            System.out.println("MISMATCH " + v + " kernel=" + NAMES[kernel] + " dim=" + dim + " got=" + got + " scalar=" + ref);
                            System.out.flush();
                            System.exit(1);
                        }
                        checked++;
                    }
                }
            }
        }
        System.out.println("SELFCHECK-OK new_variant_checks=" + checked);
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
                + " vendor=" + System.getProperty("java.vendor").replace(' ', '_') + " native=" + isNative + " arch=" + System.getProperty("os.arch") + " os=" + System.getProperty("os.name").replace(' ', '_'));
        try {
            System.out.println("SPECIES preferred=" + jdk.incubator.vector.FloatVector.SPECIES_PREFERRED + " preferredShape=" + jdk.incubator.vector.VectorShape.preferredShape()
                    + " vkLanes=" + VKernels.FL + " vkSpecies=" + VKernels.FS + " tpSpecies=" + VcA.Sp + " tpLanes=" + VcA.Lp
                    + " s64=" + VcA.S64 + " s128=" + VcA.S128 + " s256=" + VcA.S256 + " s512=" + VcA.S512);
        } catch (Throwable t) {
            System.out.println("ARM-SPECIES-FAILED " + t);
            System.exit(1);
        }
        selfCheck(variants);
        boolean allOk = true;
        for (int dim : DIMS) {
            Random rnd = new Random(1000 + dim);
            float[] data = new float[N * dim];
            float[] q = new float[dim];
            for (int i = 0; i < data.length; i++) data[i] = 2f * rnd.nextFloat() - 1f;
            for (int i = 0; i < dim; i++) q[i] = 2f * rnd.nextFloat() - 1f;
            for (int kernel = 0; kernel < 2; kernel++) {
                double[] ref = new double[CHECKED], refAbs = new double[CHECKED];
                for (int i = 0; i < CHECKED; i++) {
                    double s = 0, sa = 0;
                    for (int j = 0; j < dim; j++) {
                        double a = q[j], b = data[i * dim + j];
                        double t = kernel == 0 ? a * b : (a - b) * (a - b);
                        s += t;
                        sa += Math.abs(t);
                    }
                    ref[i] = s;
                    refAbs[i] = sa;
                }
                for (String v : variants) {
                    double maxRel = 0;
                    for (int i = 0; i < CHECKED; i++) {
                        double err = Math.abs(one(v, kernel, q, data, i * dim, dim) - ref[i]) / (refAbs[i] == 0 ? 1 : refAbs[i]);
                        if (err > maxRel) maxRel = err;
                    }
                    boolean ok = maxRel <= 1e-5;
                    allOk &= ok;
                    System.out.println(String.format("CHECK%s label=%s variant=%s kernel=%s dim=%d max_rel_err=%.3e", ok ? "" : "-FAIL", label, v, NAMES[kernel], dim, maxRel));
                    long w0 = System.nanoTime();
                    int w = 0;
                    while (w < WARM_PASSES || System.nanoTime() - w0 < WARM_NS) {
                        sink += pass(v, kernel, q, data, dim);
                        w++;
                    }
                    long[] t = new long[TIMED_PASSES];
                    double sum = 0;
                    for (int p = 0; p < TIMED_PASSES; p++) {
                        long t0 = System.nanoTime();
                        double r = pass(v, kernel, q, data, dim);
                        t[p] = System.nanoTime() - t0;
                        sum = r;
                        sink += r;
                    }
                    Arrays.sort(t);
                    double med = (t[TIMED_PASSES / 2 - 1] + t[TIMED_PASSES / 2]) / 2.0;
                    double gbps = (double) N * dim * 4 / med;
                    System.out.println(String.format("RESULT label=%s variant=%s kernel=%s dim=%d warm_passes=%d median_ns_per_call=%.2f min_ns_per_call=%.2f gbps=%.3f sum=%.9e",
                            label, v, NAMES[kernel], dim, w, med / N, t[0] / (double) N, gbps, sum));
                    System.out.flush();
                }
            }
        }
        System.out.println("DONE label=" + label + " all_checks_ok=" + allOk + " sink=" + sink);
    }
}
