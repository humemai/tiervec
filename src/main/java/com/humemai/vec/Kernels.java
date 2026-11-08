package com.humemai.vec;

/**
 * Distance kernels: a run-time choice between the Java Vector API ({@link VKernels}, and {@link VKern} for the graph kernels in {@link Kern})
 * and a scalar twin of every kernel. The scalar twins are used when -Dhumemai.vec.scalar=true or when the
 * jdk.incubator.vector module is absent. All take (array, offset) pairs so a scan reads straight out of a chunk array without copying.
 * Any length n is fine. A float kernel result depends only on n and the mode, never on offsets, so two calls on equal data are bitwise
 * equal within one mode; the scalar and the vector mode can differ by rounding (the int8 kernel is exact in both).
 *
 * <p>Inside a GraalVM native image ({@link #IN_IMAGE}) the float Vector API kernels of {@link VKernels} are far slower than scalar code (a masked tail
 * and a slice-based sum send the kernel through a slow path), so an image uses {@link NKernels} for every kernel that has a Vector API form (fp32 dot and
 * squared L2, int8 dot, the SQ8 dots, block16) when the module is present, and the scalar twins otherwise. Only hamming is scalar in an image, as it
 * is on a JVM (it has no Vector API form). A result in an image can differ from a result on a JVM by rounding.
 */
final class Kernels {
    private Kernels() {}

    /** why the scalar twins are in use, or null (set before VECTOR; no initializer on purpose) */
    static String scalarReason;

    /** true inside a GraalVM native image (the property org.graalvm.nativeimage.imagecode is set, as it is by the image builder and by a running image) */
    static final boolean IN_IMAGE = System.getProperty("org.graalvm.nativeimage.imagecode") != null;

    /**
     * true when the Vector API kernels are used: on a JVM those of {@link VKernels}, in a native image those of {@link NKernels}.
     * (initialized after {@link #IN_IMAGE}, and before everything that reads it)
     */
    static final boolean VECTOR = decide();

    /** true when the Vector API kernels of {@link VKernels} and {@link VKern} are in use: {@link #VECTOR} on a JVM, never in a native image */
    static final boolean JVM_VECTOR = VECTOR && !IN_IMAGE;

    /** true when the Vector API kernels of {@link NKernels} are in use: {@link #VECTOR} in a native image, never on a JVM */
    static final boolean NATIVE_VECTOR = VECTOR && IN_IMAGE;

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
        if (IN_IMAGE) {
            // NKernels is initialized at image build time (native-image.properties); if the module was missing there, its species is null
            try {
                Class.forName("com.humemai.vec.NKernels", true, Kernels.class.getClassLoader());
                if (NKernels.S == null) throw new IllegalStateException("no species: " + NKernels.ERROR);
                return true;
            } catch (Throwable t) {
                scalarReason = "native image without the Vector API: " + t;
                return false;
            }
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
        if (!VECTOR) return dotScalar(a, ao, b, bo, n);
        return IN_IMAGE ? NKernels.dot(a, ao, b, bo, n) : VKernels.dot(a, ao, b, bo, n);
    }

    /** squared Euclidean distance */
    static float l2sq(float[] a, int ao, float[] b, int bo, int n) {
        if (!VECTOR) return l2sqScalar(a, ao, b, bo, n);
        return IN_IMAGE ? NKernels.l2sq(a, ao, b, bo, n) : VKernels.l2sq(a, ao, b, bo, n);
    }

    /** int8 dot with an int32 accumulator; codes must be in [-127, 127] */
    static int dotI8(byte[] a, int ao, byte[] b, int bo, int n) {
        if (JVM_VECTOR) return VKernels.dotI8(a, ao, b, bo, n);
        return NATIVE_VECTOR ? NKernels.dotI8(a, ao, b, bo, n) : dotI8Scalar(a, ao, b, bo, n);
    }

    /** float query against int8 codes, one accumulator */
    static float dotF8(float[] q, int qo, byte[] c, int co, int n) {
        if (JVM_VECTOR && VKernels.F8_OK) return VKernels.dotF8(q, qo, c, co, n);
        return NATIVE_VECTOR ? NKernels.dotF8(q, qo, c, co, n) : dotF8Scalar(q, qo, c, co, n);
    }

    /** float query against int8 codes, four accumulators */
    static float dotF8x4(float[] q, int qo, byte[] c, int co, int n) {
        if (JVM_VECTOR && VKernels.F8_OK) return VKernels.dotF8x4(q, qo, c, co, n);
        return NATIVE_VECTOR ? NKernels.dotF8x4(q, qo, c, co, n) : dotF8Scalar(q, qo, c, co, n);
    }

    /** 16 dot products at once over a dimension-major block (element (j, lane) at base + j*16 + lane); out receives 16 sums */
    static void block16(float[] q, int qo, byte[] c, int base, int n, float[] out) {
        if (JVM_VECTOR && VKernels.BLOCK_OK) VKernels.block16(q, qo, c, base, n, out);
        else if (NATIVE_VECTOR) NKernels.block16(q, qo, c, base, n, out);
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
