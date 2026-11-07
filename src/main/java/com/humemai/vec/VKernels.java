package com.humemai.vec;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

/**
 * The Java Vector API kernels. Only {@link Kernels} calls this class, and only when {@link Kernels#VECTOR} is true, so a scalar-only run
 * (native image, no jdk.incubator.vector module, -Dhumemai.vec.scalar=true) never loads it. Style: Vector objects are locals of one method,
 * species are static final, no Vector object crosses a call.
 */
final class VKernels {
    private VKernels() {}

    static final VectorSpecies<Float> FS = FloatVector.SPECIES_PREFERRED;
    static final int FL = FS.length();
    private static final VectorSpecies<Byte> B128 = ByteVector.SPECIES_128;
    private static final VectorSpecies<Short> S256 = ShortVector.SPECIES_256;
    private static final VectorSpecies<Integer> I256 = IntVector.SPECIES_256;
    /** bytes species with the same lane count as FS (needs at least 64 bits, i.e. FL >= 8) */
    static final boolean F8_OK = FL >= 8;
    private static final VectorSpecies<Byte> BS = F8_OK ? VectorSpecies.of(byte.class,VectorShape.forBitSize(FL * 8)) : ByteVector.SPECIES_64;
    /** the 16-vector block kernel exists for 256-bit and 512-bit float lanes */
    static final boolean BLOCK_OK = FL == 8 || FL == 16;

    /**
     * Horizontal sum in a FIXED order (log2 steps of shift-and-add). FloatVector.reduceLanes(ADD) leaves the order unspecified and the JIT
     * tiers differ, so the same inputs could give results one ulp apart; a search must rank identically before and after compilation.
     */
    static float hsum(FloatVector v) {
        for (int h = FL / 2; h >= 1; h >>= 1) v = v.add(v.slice(h));
        return v.lane(0);
    }

    static float dot(float[] a, int ao, float[] b, int bo, int n) {
        FloatVector a0 = FloatVector.zero(FS), a1 = FloatVector.zero(FS);
        int i = 0;
        int end2 = n - (n % (2 * FL));
        for (; i < end2; i += 2 * FL) {
            a0 = FloatVector.fromArray(FS, a, ao + i).fma(FloatVector.fromArray(FS, b, bo + i), a0);
            a1 = FloatVector.fromArray(FS, a, ao + i + FL).fma(FloatVector.fromArray(FS, b, bo + i + FL), a1);
        }
        int end1 = n - (n % FL);
        for (; i < end1; i += FL) {
            a0 = FloatVector.fromArray(FS, a, ao + i).fma(FloatVector.fromArray(FS, b, bo + i), a0);
        }
        if (i < n) {
            VectorMask<Float> m = FS.indexInRange(i, n);
            a0 = FloatVector.fromArray(FS, a, ao + i, m).fma(FloatVector.fromArray(FS, b, bo + i, m), a0);
        }
        return hsum(a0.add(a1));
    }

    static float l2sq(float[] a, int ao, float[] b, int bo, int n) {
        FloatVector a0 = FloatVector.zero(FS), a1 = FloatVector.zero(FS);
        int i = 0;
        int end2 = n - (n % (2 * FL));
        for (; i < end2; i += 2 * FL) {
            FloatVector d0 = FloatVector.fromArray(FS, a, ao + i).sub(FloatVector.fromArray(FS, b, bo + i));
            FloatVector d1 = FloatVector.fromArray(FS, a, ao + i + FL).sub(FloatVector.fromArray(FS, b, bo + i + FL));
            a0 = d0.fma(d0, a0);
            a1 = d1.fma(d1, a1);
        }
        int end1 = n - (n % FL);
        for (; i < end1; i += FL) {
            FloatVector d0 = FloatVector.fromArray(FS, a, ao + i).sub(FloatVector.fromArray(FS, b, bo + i));
            a0 = d0.fma(d0, a0);
        }
        if (i < n) {
            VectorMask<Float> m = FS.indexInRange(i, n);
            FloatVector d0 = FloatVector.fromArray(FS, a, ao + i, m).sub(FloatVector.fromArray(FS, b, bo + i, m));
            a0 = d0.fma(d0, a0);
        }
        return hsum(a0.add(a1));
    }

    /** int8 x int8 dot (variant I8): widen to shorts, multiply (exact for codes in [-127,127]), widen to ints, add. */
    static int dotI8(byte[] a, int ao, byte[] b, int bo, int n) {
        IntVector acc = IntVector.zero(I256);
        int i = 0;
        int end = n - (n % 16);
        for (; i < end; i += 16) {
            ShortVector as = (ShortVector) ByteVector.fromArray(B128, a, ao + i).convertShape(VectorOperators.B2S, S256, 0);
            ShortVector bs = (ShortVector) ByteVector.fromArray(B128, b, bo + i).convertShape(VectorOperators.B2S, S256, 0);
            ShortVector p = as.mul(bs);
            acc = acc.add((IntVector) p.convertShape(VectorOperators.S2I, I256, 0)).add((IntVector) p.convertShape(VectorOperators.S2I, I256, 1));
        }
        int s = acc.reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) s += a[ao + i] * b[bo + i];
        return s;
    }

    /** float query x int8 codes (variant F8): codes widened to float, one fma per FL dimensions, one accumulator. Requires F8_OK. */
    static float dotF8(float[] q, int qo, byte[] c, int co, int n) {
        FloatVector a0 = FloatVector.zero(FS);
        int i = 0;
        int end = n - (n % FL);
        for (; i < end; i += FL) {
            FloatVector cv = (FloatVector) ByteVector.fromArray(BS, c, co + i).convertShape(VectorOperators.B2F, FS, 0);
            a0 = FloatVector.fromArray(FS, q, qo + i).fma(cv, a0);
        }
        float s = hsum(a0);
        for (; i < n; i++) s += q[qo + i] * c[co + i];
        return s;
    }

    /** same as dotF8 with four independent accumulators (variant F8X4), to hide the fma latency. Requires F8_OK. */
    static float dotF8x4(float[] q, int qo, byte[] c, int co, int n) {
        FloatVector a0 = FloatVector.zero(FS), a1 = FloatVector.zero(FS), a2 = FloatVector.zero(FS), a3 = FloatVector.zero(FS);
        int i = 0;
        int end4 = n - (n % (4 * FL));
        for (; i < end4; i += 4 * FL) {
            FloatVector c0 = (FloatVector) ByteVector.fromArray(BS, c, co + i).convertShape(VectorOperators.B2F, FS, 0);
            FloatVector c1 = (FloatVector) ByteVector.fromArray(BS, c, co + i + FL).convertShape(VectorOperators.B2F, FS, 0);
            FloatVector c2 = (FloatVector) ByteVector.fromArray(BS, c, co + i + 2 * FL).convertShape(VectorOperators.B2F, FS, 0);
            FloatVector c3 = (FloatVector) ByteVector.fromArray(BS, c, co + i + 3 * FL).convertShape(VectorOperators.B2F, FS, 0);
            a0 = FloatVector.fromArray(FS, q, qo + i).fma(c0, a0);
            a1 = FloatVector.fromArray(FS, q, qo + i + FL).fma(c1, a1);
            a2 = FloatVector.fromArray(FS, q, qo + i + 2 * FL).fma(c2, a2);
            a3 = FloatVector.fromArray(FS, q, qo + i + 3 * FL).fma(c3, a3);
        }
        int end1 = n - (n % FL);
        for (; i < end1; i += FL) {
            FloatVector cv = (FloatVector) ByteVector.fromArray(BS, c, co + i).convertShape(VectorOperators.B2F, FS, 0);
            a0 = FloatVector.fromArray(FS, q, qo + i).fma(cv, a0);
        }
        float s = hsum(a0.add(a1).add(a2.add(a3)));
        for (; i < n; i++) s += q[qo + i] * c[co + i];
        return s;
    }

    /**
     * Dimension-major block (variant BLOCK): codes of 16 vectors are stored so that element (dimension j, vector lane) is at
     * base + j*16 + lane. One query value is broadcast and multiplies 16 vectors at once; there is no horizontal sum. out gets 16 sums.
     * Requires BLOCK_OK.
     */
    static void block16(float[] q, int qo, byte[] c, int base, int n, float[] out) {
        if (FL == 8) {
            FloatVector lo = FloatVector.zero(FS), hi = FloatVector.zero(FS);
            for (int j = 0; j < n; j++) {
                FloatVector qj = FloatVector.broadcast(FS, q[qo + j]);
                ByteVector b = ByteVector.fromArray(B128, c, base + j * 16);
                lo = qj.fma((FloatVector) b.convertShape(VectorOperators.B2F, FS, 0), lo);
                hi = qj.fma((FloatVector) b.convertShape(VectorOperators.B2F, FS, 1), hi);
            }
            lo.intoArray(out, 0);
            hi.intoArray(out, 8);
        } else { // FL == 16
            FloatVector acc = FloatVector.zero(FS);
            for (int j = 0; j < n; j++) {
                FloatVector qj = FloatVector.broadcast(FS, q[qo + j]);
                acc = qj.fma((FloatVector) ByteVector.fromArray(B128, c, base + j * 16).convertShape(VectorOperators.B2F, FS, 0), acc);
            }
            acc.intoArray(out, 0);
        }
    }
}
