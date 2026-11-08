package com.humemai.vec;

/**
 * All distance kernels of the graph tier live here. Each one picks, once per JVM, between the Vector API version in {@link VKern} and a
 * scalar twin: {@link Kernels#VECTOR} is the one switch for the flat, the IVF and the graph kernels (-Dhumemai.vec.scalar=true or a missing
 * jdk.incubator.vector module give the scalar twins). In a native image all three use the Vector API, through {@link NKernels} (the SQ8 dot is its dotF8).
 * On an aarch64 JVM the SQ8 dot uses the 128-bit form of NKernels too ({@link Kernels#JVM_NARROW}), so that it is the same kernel as {@code Kernels.dotF8} there; on another
 * JVM with fewer than 8 float lanes it is the scalar twin.
 * The scalar twins are the ones of {@link Kernels}
 * (dotScalar, l2sqScalar, dotF8Scalar), so the graph and the flat tier compute the same score for the same data in scalar mode.
 * VKern is touched only when Kernels.VECTOR is true, and nothing here has a Vector API type in a field or a signature, so loading this
 * class never loads jdk.incubator.vector classes.
 * Plain float arrays with offsets. dotQ8: dot of a float query with SIGNED int8 codes (the query starts at index 0).
 * A float result depends only on n and the mode, never on offsets; the two modes can differ by rounding (the tests state the tolerance).
 */
final class Kern {
    private Kern() {}

    static float dot(float[] a, int ao, float[] b, int bo, int n) {
        if (!Kernels.VECTOR) return Kernels.dotScalar(a, ao, b, bo, n);
        return Kernels.IN_IMAGE ? NKernels.dot(a, ao, b, bo, n) : VKern.dot(a, ao, b, bo, n);
    }

    /** squared L2 distance */
    static float l2(float[] a, int ao, float[] b, int bo, int n) {
        if (!Kernels.VECTOR) return Kernels.l2sqScalar(a, ao, b, bo, n);
        return Kernels.IN_IMAGE ? NKernels.l2sq(a, ao, b, bo, n) : VKern.l2(a, ao, b, bo, n);
    }

    /** sum over i of q[i] * codes[co + i] (codes are signed bytes) */
    static float dotQ8(float[] q, byte[] codes, int co, int n) {
        if (Kernels.JVM_QUANT_VK && VKern.Q8_OK) return VKern.dotQ8(q, codes, co, n);
        if (Kernels.JVM_NARROW) return NKernels.dotF8N(q, 0, codes, co, n);
        return Kernels.NATIVE_VECTOR ? NKernels.dotF8(q, 0, codes, co, n) : Kernels.dotF8Scalar(q, 0, codes, co, n);
    }

    static float sum(float[] a, int n) { float s = 0; for (int i = 0; i < n; i++) s += a[i]; return s; }

    /** float lanes of the Vector API kernels, 0 when the scalar twins are in use */
    static int lanes() {
        return Kernels.VECTOR ? (Kernels.IN_IMAGE ? NKernels.L : VKern.L) : 0;
    }
}
