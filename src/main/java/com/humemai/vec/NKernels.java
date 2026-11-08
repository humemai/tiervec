package com.humemai.vec;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * The distance kernels for a GraalVM native image: the fp32 dot and squared L2 (issue #44) and the int8, SQ8 and block kernels (issue #48). {@link Kernels}
 * and {@link Kern} call this class only when they run inside an image and the module jdk.incubator.vector is present; on a JVM {@link VKernels}
 * and {@link VKern} are used. Only hamming has no Vector API form and stays scalar in an image.
 *
 * <p>Why a second set of kernels. In a native image a masked step anywhere in a method (the tail of {@link VKernels}) or a slice-based horizontal sum
 * sends the whole kernel through a slow path, and the kernel runs at a few percent of the speed of scalar code. The form here is the textbook one that
 * the image compiles to SIMD instructions: the species is a {@code static final} fixed when the image is built, two accumulators, {@code fma}, a sum
 * of the lanes with {@code reduceLanes(ADD)}, a scalar tail, no masks. A sum in a fixed order (through {@code lane()} or {@code intoArray}) is fast
 * only on Oracle GraalVM, so this class does not use one: the order of the lane sum is the one of the instruction, and a result can differ from the
 * JVM kernels by rounding (the tests state the tolerance).
 *
 * <p>The int8, SQ8 and block kernels (issue #48) follow the same rules, and each has two forms, one per float species, chosen by {@link #WIDE}. The
 * byte, short and int species of a form have the bit size of its float species. The Vector API has no 32-bit byte species, so on aarch64 (4 float
 * lanes) a byte vector with the lane count of the float species does not exist: the narrow forms load 64 or 128 bits of bytes and split them into parts.
 * The form of each kernel is the one that was fastest in an image on both toolchains (Oracle GraalVM 25 and GraalVM CE 25.0.2): x86-64 (wide) takes
 * the {@code convertShape} form for dotI8 and block16, one wide byte load plus {@code convert} for dotF8, and the textbook form with four accumulators
 * for dotF8x4; aarch64 (narrow) takes the wide-load form for dotI8 and dotF8, the textbook form for dotF8x4, and the {@code convertShape} form for block16.
 * Both forms of every kernel are package-private so that a test can run them on any host.
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

    /** true when {@link #S} has 256 bits (8 float lanes), false on aarch64 (128 bits); the int8, SQ8 and block kernels pick their form with it */
    static final boolean WIDE;
    // the species of the two forms of the int8, SQ8 and block kernels: 256 bits (wide) and 128 bits (narrow), and the byte species 64, 128 and 256 bits
    static final VectorSpecies<Float> F256, F128;
    static final VectorSpecies<Integer> I256, I128;
    static final VectorSpecies<Short> H256, H128;
    static final VectorSpecies<Byte> B64, B128, B256;

    static {
        VectorSpecies<Float> s = null, f256 = null, f128 = null;
        VectorSpecies<Integer> i256 = null, i128 = null;
        VectorSpecies<Short> h256 = null, h128 = null;
        VectorSpecies<Byte> b64 = null, b128 = null, b256 = null;
        boolean wide = true;
        Throwable err = null;
        try {
            String arch = System.getProperty("os.arch", "");
            wide = !(arch.equals("aarch64") || arch.equals("arm64"));
            f256 = FloatVector.SPECIES_256;
            f128 = FloatVector.SPECIES_128;
            i256 = IntVector.SPECIES_256;
            i128 = IntVector.SPECIES_128;
            h256 = ShortVector.SPECIES_256;
            h128 = ShortVector.SPECIES_128;
            b64 = ByteVector.SPECIES_64;
            b128 = ByteVector.SPECIES_128;
            b256 = ByteVector.SPECIES_256;
            s = wide ? f256 : f128; // last: S stays null when anything above failed
        } catch (Throwable t) {
            err = t;
            s = null;
        }
        S = s;
        L = s == null ? 0 : s.length();
        ERROR = err;
        WIDE = wide;
        F256 = f256;
        F128 = f128;
        I256 = i256;
        I128 = i128;
        H256 = h256;
        H128 = h128;
        B64 = b64;
        B128 = b128;
        B256 = b256;
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

    // ---- int8 and SQ8 kernels and the block kernel (issue #48): a form per float species, no mask in any of them ----

    /** int8 dot with an int32 accumulator; codes must be in [-127, 127] (a product then fits a short) */
    static int dotI8(byte[] a, int ao, byte[] b, int bo, int n) {
        return WIDE ? dotI8W(a, ao, b, bo, n) : dotI8N(a, ao, b, bo, n);
    }

    /** float query against int8 codes, one accumulator */
    static float dotF8(float[] q, int qo, byte[] c, int co, int n) {
        return WIDE ? dotF8W(q, qo, c, co, n) : dotF8N(q, qo, c, co, n);
    }

    /** float query against int8 codes, four accumulators */
    static float dotF8x4(float[] q, int qo, byte[] c, int co, int n) {
        return WIDE ? dotF8x4W(q, qo, c, co, n) : dotF8x4N(q, qo, c, co, n);
    }

    /** 16 dot products at once over a dimension-major block (element (j, lane) at base + j*16 + lane); out receives 16 sums */
    static void block16(float[] q, int qo, byte[] c, int base, int n, float[] out) {
        if (WIDE) block16W(q, qo, c, base, n, out);
        else block16N(q, qo, c, base, n, out);
    }

    // wide forms: 256-bit float species, 8 lanes

    /** 16 bytes per step: byte to short with convertShape, one short multiply, short to two int vectors */
    static int dotI8W(byte[] a, int ao, byte[] b, int bo, int n) {
        IntVector acc = IntVector.zero(I256);
        int i = 0;
        int end = n - (n % 16);
        for (; i < end; i += 16) {
            ShortVector as = (ShortVector) ByteVector.fromArray(B128, a, ao + i).convertShape(VectorOperators.B2S, H256, 0);
            ShortVector bs = (ShortVector) ByteVector.fromArray(B128, b, bo + i).convertShape(VectorOperators.B2S, H256, 0);
            ShortVector p = as.mul(bs);
            acc = acc.add((IntVector) p.convertShape(VectorOperators.S2I, I256, 0)).add((IntVector) p.convertShape(VectorOperators.S2I, I256, 1));
        }
        int s = acc.reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) s += a[ao + i] * b[bo + i];
        return s;
    }

    /** 32 bytes per step: one 256-bit byte load, four float parts with convert */
    static float dotF8W(float[] q, int qo, byte[] c, int co, int n) {
        FloatVector a0 = FloatVector.zero(F256);
        int i = 0;
        int end = n - (n % 32);
        for (; i < end; i += 32) {
            ByteVector cb = ByteVector.fromArray(B256, c, co + i);
            a0 = FloatVector.fromArray(F256, q, qo + i).fma((FloatVector) cb.convert(VectorOperators.B2F, 0), a0);
            a0 = FloatVector.fromArray(F256, q, qo + i + 8).fma((FloatVector) cb.convert(VectorOperators.B2F, 1), a0);
            a0 = FloatVector.fromArray(F256, q, qo + i + 16).fma((FloatVector) cb.convert(VectorOperators.B2F, 2), a0);
            a0 = FloatVector.fromArray(F256, q, qo + i + 24).fma((FloatVector) cb.convert(VectorOperators.B2F, 3), a0);
        }
        float s = a0.reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) s += q[qo + i] * c[co + i];
        return s;
    }

    /** 32 elements per step, four accumulators, each from a 64-bit byte load; then 8 per step; reduceLanes; scalar tail */
    static float dotF8x4W(float[] q, int qo, byte[] c, int co, int n) {
        FloatVector a0 = FloatVector.zero(F256), a1 = FloatVector.zero(F256), a2 = FloatVector.zero(F256), a3 = FloatVector.zero(F256);
        int i = 0;
        int end4 = n - (n % 32);
        for (; i < end4; i += 32) {
            FloatVector c0 = (FloatVector) ByteVector.fromArray(B64, c, co + i).convertShape(VectorOperators.B2F, F256, 0);
            FloatVector c1 = (FloatVector) ByteVector.fromArray(B64, c, co + i + 8).convertShape(VectorOperators.B2F, F256, 0);
            FloatVector c2 = (FloatVector) ByteVector.fromArray(B64, c, co + i + 16).convertShape(VectorOperators.B2F, F256, 0);
            FloatVector c3 = (FloatVector) ByteVector.fromArray(B64, c, co + i + 24).convertShape(VectorOperators.B2F, F256, 0);
            a0 = FloatVector.fromArray(F256, q, qo + i).fma(c0, a0);
            a1 = FloatVector.fromArray(F256, q, qo + i + 8).fma(c1, a1);
            a2 = FloatVector.fromArray(F256, q, qo + i + 16).fma(c2, a2);
            a3 = FloatVector.fromArray(F256, q, qo + i + 24).fma(c3, a3);
        }
        int end1 = n - (n % 8);
        for (; i < end1; i += 8) {
            FloatVector cv = (FloatVector) ByteVector.fromArray(B64, c, co + i).convertShape(VectorOperators.B2F, F256, 0);
            a0 = FloatVector.fromArray(F256, q, qo + i).fma(cv, a0);
        }
        float s = a0.add(a1).add(a2.add(a3)).reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) s += q[qo + i] * c[co + i];
        return s;
    }

    /** per dimension one 128-bit byte load, two float parts with convertShape; out is written with intoArray (no tail, no mask) */
    static void block16W(float[] q, int qo, byte[] c, int base, int n, float[] out) {
        FloatVector lo = FloatVector.zero(F256), hi = FloatVector.zero(F256);
        for (int j = 0; j < n; j++) {
            FloatVector qj = FloatVector.broadcast(F256, q[qo + j]);
            ByteVector b = ByteVector.fromArray(B128, c, base + j * 16);
            lo = qj.fma((FloatVector) b.convertShape(VectorOperators.B2F, F256, 0), lo);
            hi = qj.fma((FloatVector) b.convertShape(VectorOperators.B2F, F256, 1), hi);
        }
        lo.intoArray(out, 0);
        hi.intoArray(out, 8);
    }

    // narrow forms: 128-bit float species, 4 lanes; no 32-bit byte species exists, so bytes come in 64 or 128 bits and are split into parts

    /** 16 bytes per step: one 128-bit byte load, two short parts with convert, four int parts */
    static int dotI8N(byte[] a, int ao, byte[] b, int bo, int n) {
        IntVector acc = IntVector.zero(I128);
        int i = 0;
        int end = n - (n % 16);
        for (; i < end; i += 16) {
            ByteVector av = ByteVector.fromArray(B128, a, ao + i);
            ByteVector bv = ByteVector.fromArray(B128, b, bo + i);
            ShortVector p0 = ((ShortVector) av.convert(VectorOperators.B2S, 0)).mul((ShortVector) bv.convert(VectorOperators.B2S, 0));
            ShortVector p1 = ((ShortVector) av.convert(VectorOperators.B2S, 1)).mul((ShortVector) bv.convert(VectorOperators.B2S, 1));
            acc = acc.add((IntVector) p0.convert(VectorOperators.S2I, 0)).add((IntVector) p0.convert(VectorOperators.S2I, 1));
            acc = acc.add((IntVector) p1.convert(VectorOperators.S2I, 0)).add((IntVector) p1.convert(VectorOperators.S2I, 1));
        }
        int s = acc.reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) s += a[ao + i] * b[bo + i];
        return s;
    }

    /** 16 bytes per step: one 128-bit byte load, four float parts with convert */
    static float dotF8N(float[] q, int qo, byte[] c, int co, int n) {
        FloatVector a0 = FloatVector.zero(F128);
        int i = 0;
        int end = n - (n % 16);
        for (; i < end; i += 16) {
            ByteVector cb = ByteVector.fromArray(B128, c, co + i);
            a0 = FloatVector.fromArray(F128, q, qo + i).fma((FloatVector) cb.convert(VectorOperators.B2F, 0), a0);
            a0 = FloatVector.fromArray(F128, q, qo + i + 4).fma((FloatVector) cb.convert(VectorOperators.B2F, 1), a0);
            a0 = FloatVector.fromArray(F128, q, qo + i + 8).fma((FloatVector) cb.convert(VectorOperators.B2F, 2), a0);
            a0 = FloatVector.fromArray(F128, q, qo + i + 12).fma((FloatVector) cb.convert(VectorOperators.B2F, 3), a0);
        }
        float s = a0.reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) s += q[qo + i] * c[co + i];
        return s;
    }

    /** 16 elements per step, four accumulators, two 64-bit byte loads split with convertShape; reduceLanes; scalar tail */
    static float dotF8x4N(float[] q, int qo, byte[] c, int co, int n) {
        FloatVector a0 = FloatVector.zero(F128), a1 = FloatVector.zero(F128), a2 = FloatVector.zero(F128), a3 = FloatVector.zero(F128);
        int i = 0;
        int end = n - (n % 16);
        for (; i < end; i += 16) {
            ByteVector cb0 = ByteVector.fromArray(B64, c, co + i);
            ByteVector cb1 = ByteVector.fromArray(B64, c, co + i + 8);
            FloatVector c0 = (FloatVector) cb0.convertShape(VectorOperators.B2F, F128, 0);
            FloatVector c1 = (FloatVector) cb0.convertShape(VectorOperators.B2F, F128, 1);
            FloatVector c2 = (FloatVector) cb1.convertShape(VectorOperators.B2F, F128, 0);
            FloatVector c3 = (FloatVector) cb1.convertShape(VectorOperators.B2F, F128, 1);
            a0 = FloatVector.fromArray(F128, q, qo + i).fma(c0, a0);
            a1 = FloatVector.fromArray(F128, q, qo + i + 4).fma(c1, a1);
            a2 = FloatVector.fromArray(F128, q, qo + i + 8).fma(c2, a2);
            a3 = FloatVector.fromArray(F128, q, qo + i + 12).fma(c3, a3);
        }
        float s = a0.add(a1).add(a2.add(a3)).reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) s += q[qo + i] * c[co + i];
        return s;
    }

    /** per dimension one 128-bit byte load, four float parts with convertShape; out is written with intoArray (no tail, no mask) */
    static void block16N(float[] q, int qo, byte[] c, int base, int n, float[] out) {
        FloatVector v0 = FloatVector.zero(F128), v1 = FloatVector.zero(F128), v2 = FloatVector.zero(F128), v3 = FloatVector.zero(F128);
        for (int j = 0; j < n; j++) {
            FloatVector qj = FloatVector.broadcast(F128, q[qo + j]);
            ByteVector b = ByteVector.fromArray(B128, c, base + j * 16);
            v0 = qj.fma((FloatVector) b.convertShape(VectorOperators.B2F, F128, 0), v0);
            v1 = qj.fma((FloatVector) b.convertShape(VectorOperators.B2F, F128, 1), v1);
            v2 = qj.fma((FloatVector) b.convertShape(VectorOperators.B2F, F128, 2), v2);
            v3 = qj.fma((FloatVector) b.convertShape(VectorOperators.B2F, F128, 3), v3);
        }
        v0.intoArray(out, 0);
        v1.intoArray(out, 4);
        v2.intoArray(out, 8);
        v3.intoArray(out, 12);
    }
}
