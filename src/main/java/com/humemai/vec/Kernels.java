package com.humemai.vec;

/**
 * Distance kernels: a run-time choice between the Java Vector API ({@link VKernels}, and {@link VKern} for the graph kernels in {@link Kern})
 * and a scalar twin of every kernel. The scalar twins are used when -Dhumemai.vec.scalar=true, inside a GraalVM native image (the engine forces scalar code there), or when the
 * jdk.incubator.vector module is absent. All take (array, offset) pairs so a scan reads straight out of a chunk array without copying.
 * Any length n is fine. A float kernel result depends only on n and the mode, never on offsets, so two calls on equal data are bitwise
 * equal within one mode; the scalar and the vector mode can differ by rounding (the int8 kernel is exact in both).
 */
final class Kernels {
    private Kernels() {}

    /** why the scalar twins are in use, or null (set before VECTOR; no initializer on purpose) */
    static String scalarReason;

    /** true when the Vector API kernels are used */
    static final boolean VECTOR = decide();

    /**
     * The Vector API holder classes are initialized HERE, inside a try: if one of them cannot load or initialize (module missing,
     * unsupported species, native image) every index (flat, IVF, graph, tier) falls back to the scalar twins instead of failing on the
     * first search, so one run never mixes the two modes. With -Dhumemai.vec.scalar=true neither holder is touched.
     */
    private static boolean decide() {
        if (Boolean.getBoolean("humemai.vec.scalar")) {
            scalarReason = "forced by -Dhumemai.vec.scalar=true";
            return false;
        }
        if (System.getProperty("org.graalvm.nativeimage.imagecode") != null) {
            scalarReason = "native image";
            return false;
        }
        try {
            Class.forName("com.humemai.vec.VKernels", true, Kernels.class.getClassLoader());
            Class.forName("com.humemai.vec.VKern", true, Kernels.class.getClassLoader());
            return true;
        } catch (Throwable t) {
            scalarReason = "Vector API unavailable: " + t;
            return false;
        }
    }

    static float dot(float[] a, int ao, float[] b, int bo, int n) {
        return VECTOR ? VKernels.dot(a, ao, b, bo, n) : dotScalar(a, ao, b, bo, n);
    }

    /** squared Euclidean distance */
    static float l2sq(float[] a, int ao, float[] b, int bo, int n) {
        return VECTOR ? VKernels.l2sq(a, ao, b, bo, n) : l2sqScalar(a, ao, b, bo, n);
    }

    /** int8 dot with an int32 accumulator; codes must be in [-127, 127] */
    static int dotI8(byte[] a, int ao, byte[] b, int bo, int n) {
        return VECTOR ? VKernels.dotI8(a, ao, b, bo, n) : dotI8Scalar(a, ao, b, bo, n);
    }

    /** float query against int8 codes, one accumulator */
    static float dotF8(float[] q, int qo, byte[] c, int co, int n) {
        return VECTOR && VKernels.F8_OK ? VKernels.dotF8(q, qo, c, co, n) : dotF8Scalar(q, qo, c, co, n);
    }

    /** float query against int8 codes, four accumulators */
    static float dotF8x4(float[] q, int qo, byte[] c, int co, int n) {
        return VECTOR && VKernels.F8_OK ? VKernels.dotF8x4(q, qo, c, co, n) : dotF8Scalar(q, qo, c, co, n);
    }

    /** 16 dot products at once over a dimension-major block (element (j, lane) at base + j*16 + lane); out receives 16 sums */
    static void block16(float[] q, int qo, byte[] c, int base, int n, float[] out) {
        if (VECTOR && VKernels.BLOCK_OK) VKernels.block16(q, qo, c, base, n, out);
        else block16Scalar(q, qo, c, base, n, out);
    }

    static int hamming(long[] a, int ao, long[] b, int bo, int words) {
        int h = 0;
        for (int i = 0; i < words; i++) h += Long.bitCount(a[ao + i] ^ b[bo + i]);
        return h;
    }

    /** a normalized copy; the zero vector stays the zero vector */
    static float[] normalized(float[] v) {
        float[] r = new float[v.length];
        float nn = dot(v, 0, v, 0, v.length);
        if (nn > 0f) {
            float inv = (float) (1.0 / Math.sqrt(nn));
            for (int i = 0; i < r.length; i++) r[i] = v[i] * inv;
        }
        return r;
    }

    // ---- scalar twins (package-private so a test can compare them with the vector kernels) ----

    static float dotScalar(float[] a, int ao, float[] b, int bo, int n) {
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int i = 0;
        for (; i + 4 <= n; i += 4) {
            s0 += a[ao + i] * b[bo + i];
            s1 += a[ao + i + 1] * b[bo + i + 1];
            s2 += a[ao + i + 2] * b[bo + i + 2];
            s3 += a[ao + i + 3] * b[bo + i + 3];
        }
        for (; i < n; i++) s0 += a[ao + i] * b[bo + i];
        return (s0 + s1) + (s2 + s3);
    }

    static float l2sqScalar(float[] a, int ao, float[] b, int bo, int n) {
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int i = 0;
        for (; i + 4 <= n; i += 4) {
            float d0 = a[ao + i] - b[bo + i], d1 = a[ao + i + 1] - b[bo + i + 1], d2 = a[ao + i + 2] - b[bo + i + 2], d3 = a[ao + i + 3] - b[bo + i + 3];
            s0 += d0 * d0;
            s1 += d1 * d1;
            s2 += d2 * d2;
            s3 += d3 * d3;
        }
        for (; i < n; i++) {
            float d = a[ao + i] - b[bo + i];
            s0 += d * d;
        }
        return (s0 + s1) + (s2 + s3);
    }

    static int dotI8Scalar(byte[] a, int ao, byte[] b, int bo, int n) {
        int s = 0;
        for (int i = 0; i < n; i++) s += a[ao + i] * b[bo + i];
        return s;
    }

    static float dotF8Scalar(float[] q, int qo, byte[] c, int co, int n) {
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int i = 0;
        for (; i + 4 <= n; i += 4) {
            s0 += q[qo + i] * c[co + i];
            s1 += q[qo + i + 1] * c[co + i + 1];
            s2 += q[qo + i + 2] * c[co + i + 2];
            s3 += q[qo + i + 3] * c[co + i + 3];
        }
        for (; i < n; i++) s0 += q[qo + i] * c[co + i];
        return (s0 + s1) + (s2 + s3);
    }

    static void block16Scalar(float[] q, int qo, byte[] c, int base, int n, float[] out) {
        for (int lane = 0; lane < 16; lane++) out[lane] = 0f;
        for (int j = 0; j < n; j++) {
            float qj = q[qo + j];
            int b = base + j * 16;
            for (int lane = 0; lane < 16; lane++) out[lane] += qj * c[b + lane];
        }
    }
}
