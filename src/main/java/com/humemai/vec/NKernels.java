package com.humemai.vec;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * The fp32 distance kernels for a GraalVM native image (issue #44). {@link Kernels} and {@link Kern} call this class only when they run inside an
 * image and the module jdk.incubator.vector is present; on a JVM {@link VKernels} and {@link VKern} are used.
 *
 * <p>Why a second set of kernels. In a native image a masked step anywhere in a method (the tail of {@link VKernels}) or a slice-based horizontal sum
 * sends the whole kernel through a slow path, and the kernel runs at a few percent of the speed of scalar code. The form here is the textbook one that
 * the image compiles to SIMD instructions: the species is a {@code static final} fixed when the image is built, two accumulators, {@code fma}, a sum
 * of the lanes with {@code reduceLanes(ADD)}, a scalar tail, no masks. A sum in a fixed order (through {@code lane()} or {@code intoArray}) is fast
 * only on Oracle GraalVM, so this class does not use one: the order of the lane sum is the one of the instruction, and a result can differ from the
 * JVM kernels by rounding (the tests state the tolerance).
 *
 * <p>The species is chosen once: 128 bits on aarch64 (NEON), 256 bits everywhere else (AVX2 on x86-64; a CPU with AVX-512 runs the 256-bit
 * species too). The choice is read from {@code os.arch} in the static initializer, so an application must have the image builder run it: the
 * file META-INF/native-image/com.humemai/tiervec/native-image.properties in the jar does that with --initialize-at-build-time. The image builder must
 * also be given {@code --add-modules jdk.incubator.vector -H:+VectorAPISupport} (see docs/DESIGN.md).
 *
 * <p>The initializer catches a failure (the module is missing) and leaves {@link #S} null, so that a build of an image without the module, with the
 * properties file on the class path, does not fail; {@link Kernels} then selects the scalar twins and names the cause as the scalar reason.
 */
final class NKernels {
    private NKernels() {}

    /** the species of every kernel here, or null when it could not be created (the Vector API module is missing) */
    static final VectorSpecies<Float> S;
    /** lanes of {@link #S}, 0 when {@link #S} is null */
    static final int L;
    /** the failure that left {@link #S} null, or null */
    static final Throwable ERROR;
    /** value of org.graalvm.nativeimage.imagecode when this class was initialized: "buildtime" when the image builder did it (what the smoke test wants), "runtime", or null on a JVM */
    static final String INIT_IMAGECODE = System.getProperty("org.graalvm.nativeimage.imagecode");

    static {
        VectorSpecies<Float> s = null;
        Throwable err = null;
        try {
            s = pick();
        } catch (Throwable t) {
            err = t;
        }
        S = s;
        L = s == null ? 0 : s.length();
        ERROR = err;
    }

    /** 128 bits on aarch64, 256 bits otherwise */
    private static VectorSpecies<Float> pick() {
        String arch = System.getProperty("os.arch", "");
        return arch.equals("aarch64") || arch.equals("arm64") ? FloatVector.SPECIES_128 : FloatVector.SPECIES_256;
    }

    static float dot(float[] a, int ao, float[] b, int bo, int n) {
        FloatVector a0 = FloatVector.zero(S), a1 = FloatVector.zero(S);
        int i = 0;
        int end2 = n - (n % (2 * L));
        for (; i < end2; i += 2 * L) {
            a0 = FloatVector.fromArray(S, a, ao + i).fma(FloatVector.fromArray(S, b, bo + i), a0);
            a1 = FloatVector.fromArray(S, a, ao + i + L).fma(FloatVector.fromArray(S, b, bo + i + L), a1);
        }
        int end1 = n - (n % L);
        for (; i < end1; i += L) {
            a0 = FloatVector.fromArray(S, a, ao + i).fma(FloatVector.fromArray(S, b, bo + i), a0);
        }
        float s = a0.add(a1).reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) s += a[ao + i] * b[bo + i];
        return s;
    }

    static float l2sq(float[] a, int ao, float[] b, int bo, int n) {
        FloatVector a0 = FloatVector.zero(S), a1 = FloatVector.zero(S);
        int i = 0;
        int end2 = n - (n % (2 * L));
        for (; i < end2; i += 2 * L) {
            FloatVector d0 = FloatVector.fromArray(S, a, ao + i).sub(FloatVector.fromArray(S, b, bo + i));
            FloatVector d1 = FloatVector.fromArray(S, a, ao + i + L).sub(FloatVector.fromArray(S, b, bo + i + L));
            a0 = d0.fma(d0, a0);
            a1 = d1.fma(d1, a1);
        }
        int end1 = n - (n % L);
        for (; i < end1; i += L) {
            FloatVector d0 = FloatVector.fromArray(S, a, ao + i).sub(FloatVector.fromArray(S, b, bo + i));
            a0 = d0.fma(d0, a0);
        }
        float s = a0.add(a1).reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) {
            float d = a[ao + i] - b[bo + i];
            s += d * d;
        }
        return s;
    }
}
