package com.humemai.vec;

import java.util.Arrays;
import java.util.Random;

/**
 * Micro-benchmark of the distance kernels (experiment, branch native-perf-experiment; not part of the library). Lives in the package of
 * the library because the kernels and the kernel switch are package-private.
 *
 * <p>Argument: the arm the program is supposed to be, checked at start (exit 1 when it is not what runs):
 * <ul>
 * <li>{@code a}: JVM, Vector API kernels ({@code Kernels.VECTOR} is true)</li>
 * <li>{@code b}: JVM, scalar twins forced with -Dhumemai.vec.scalar=true</li>
 * <li>{@code c}: native image, scalar twins (the switch picks them in an image)</li>
 * <li>{@code d}: native image, Vector API kernels called directly ({@code VKernels}), because the switch refuses them in an image</li>
 * </ul>
 *
 * <p>For each dimension in 128, 768, 1536 it allocates 4096 vectors (fixed seed, one flat array, as a scan reads a chunk), one query,
 * and runs each kernel (dot, l2sq, dotF8 against int8 codes) over all vectors. Warm-up is at least 5 passes and at least 1.5 s, then 20
 * timed passes; one pass is 4096 kernel calls. It reports the median ns per call, GB/s of vector data read (4 bytes per element for
 * fp32, 1 for the codes; the query stays in cache), and two checksums of the results (sum and sum of absolute values) so that arms can
 * be compared within the tolerance of KernDiffTest (1e-5 times the sum of absolute terms). No assertion on speed.
 */
public final class KernelBench {
    private KernelBench() {}

    static final int N = 4096, WARM_PASSES = 5, TIMED_PASSES = 20;
    static final long WARM_NS = 1_500_000_000L;
    static final int[] DIMS = {128, 768, 1536};
    static final int DOT = 0, L2 = 1, F8 = 2;
    static final String[] NAMES = {"dot", "l2sq", "dotF8"};

    /** true: call the Vector API holder directly (arm d) */
    static boolean direct;
    static double sink;

    static float dot(float[] a, int ao, float[] b, int bo, int n) {
        return direct ? VKernels.dot(a, ao, b, bo, n) : Kernels.dot(a, ao, b, bo, n);
    }

    static float l2sq(float[] a, int ao, float[] b, int bo, int n) {
        return direct ? VKernels.l2sq(a, ao, b, bo, n) : Kernels.l2sq(a, ao, b, bo, n);
    }

    static float dotF8(float[] q, int qo, byte[] c, int co, int n) {
        if (direct) return VKernels.F8_OK ? VKernels.dotF8(q, qo, c, co, n) : Kernels.dotF8Scalar(q, qo, c, co, n);
        return Kernels.dotF8(q, qo, c, co, n);
    }

    /** one pass over all vectors; returns the sum of the results; abs[0] receives the sum of absolute values */
    static double pass(int kernel, float[] q, float[] data, byte[] codes, int dim, double[] abs) {
        double s = 0, sa = 0;
        switch (kernel) {
            case DOT:
                for (int i = 0; i < N; i++) {
                    float r = dot(q, 0, data, i * dim, dim);
                    s += r;
                    sa += Math.abs(r);
                }
                break;
            case L2:
                for (int i = 0; i < N; i++) {
                    float r = l2sq(q, 0, data, i * dim, dim);
                    s += r;
                    sa += Math.abs(r);
                }
                break;
            default:
                for (int i = 0; i < N; i++) {
                    float r = dotF8(q, 0, codes, i * dim, dim);
                    s += r;
                    sa += Math.abs(r);
                }
                break;
        }
        abs[0] = sa;
        return s;
    }

    static void die(String msg) {
        System.out.println("ARM-MISMATCH " + msg);
        System.out.flush();
        System.exit(1);
    }

    public static void main(String[] args) {
        String arm = args.length > 0 ? args[0] : "";
        boolean isNative = System.getProperty("org.graalvm.nativeimage.imagecode") != null;
        boolean vector = Kernels.VECTOR;
        String reason = Kernels.scalarReason;
        System.out.println("ID arm=" + arm + " vm=" + System.getProperty("java.vm.name") + " version=" + System.getProperty("java.version")
                + " vendor=" + System.getProperty("java.vendor") + " arch=" + System.getProperty("os.arch") + " native=" + isNative);
        System.out.println("KERNELS Kernels.VECTOR=" + vector + " scalarReason=" + reason);
        switch (arm) {
            case "a":
                if (isNative || !vector) die("arm a needs a JVM with Kernels.VECTOR true");
                break;
            case "b":
                if (isNative || vector || reason == null || !reason.startsWith("forced")) die("arm b needs a JVM with the scalar twins forced");
                break;
            case "c":
                if (!isNative || vector || !"native image".equals(reason)) die("arm c needs a native image with the scalar twins");
                break;
            case "d":
                if (!isNative) die("arm d needs a native image");
                direct = true;
                try {
                    System.out.println("VKERNELS lanes=" + VKernels.FL + " F8_OK=" + VKernels.F8_OK + " species=" + VKernels.FS);
                } catch (Throwable t) {
                    System.out.println("ARM-D-VKERNELS-FAILED " + t);
                    System.exit(1);
                }
                break;
            default:
                die("unknown arm '" + arm + "'");
        }
        if (arm.equals("a")) System.out.println("VKERNELS lanes=" + VKernels.FL + " F8_OK=" + VKernels.F8_OK + " species=" + VKernels.FS);

        double[] abs = new double[1];
        for (int dim : DIMS) {
            Random rnd = new Random(1000 + dim);
            float[] data = new float[N * dim];
            byte[] codes = new byte[N * dim];
            float[] q = new float[dim];
            for (int i = 0; i < data.length; i++) data[i] = 2f * rnd.nextFloat() - 1f;
            for (int i = 0; i < codes.length; i++) codes[i] = (byte) (rnd.nextInt(255) - 127);
            for (int i = 0; i < dim; i++) q[i] = 2f * rnd.nextFloat() - 1f;
            for (int kernel = 0; kernel < 3; kernel++) {
                long w0 = System.nanoTime();
                int w = 0;
                while (w < WARM_PASSES || System.nanoTime() - w0 < WARM_NS) {
                    sink += pass(kernel, q, data, codes, dim, abs);
                    w++;
                }
                long[] t = new long[TIMED_PASSES];
                double sum = 0, sumAbs = 0;
                for (int p = 0; p < TIMED_PASSES; p++) {
                    long t0 = System.nanoTime();
                    double r = pass(kernel, q, data, codes, dim, abs);
                    t[p] = System.nanoTime() - t0;
                    sum = r;
                    sumAbs = abs[0];
                    sink += r;
                }
                Arrays.sort(t);
                double med = (t[TIMED_PASSES / 2 - 1] + t[TIMED_PASSES / 2]) / 2.0;
                double nsPerCall = med / N;
                int bytesPerElem = kernel == F8 ? 1 : 4;
                double gbps = (double) N * dim * bytesPerElem / med;
                System.out.println(String.format("RESULT arm=%s kernel=%s dim=%d warm_passes=%d median_ns_per_call=%.2f min_ns_per_call=%.2f gbps=%.3f sum=%.9e sumabs=%.9e",
                        arm, NAMES[kernel], dim, w, nsPerCall, t[0] / (double) N, gbps, sum, sumAbs));
                System.out.flush();
            }
        }
        System.out.println("DONE arm=" + arm + " sink=" + sink);
    }
}
