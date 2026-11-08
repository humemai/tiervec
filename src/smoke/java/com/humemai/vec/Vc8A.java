package com.humemai.vec;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Experiment round 5 (branch native-perf-experiment only): the four VKernels kernels that use convertShape (dotI8, dotF8, dotF8x4, block16),
 * arm64, 128-bit float species (4 lanes), all species static final. The Vector API has no 32-bit byte species (the smallest is ByteVector.SPECIES_64,
 * 8 lanes), so a byte vector with the lane count of the float species does not exist: the 64-bit byte vector is split with convertShape parts 0 and 1,
 * the 128-bit byte vector with parts 0 to 3. No kernel here contains a mask (VKernels has none in these four).
 * Variants per kernel:
 * cur = the VKernels form with the species fixed, adapted to 128 bits (convertShape from B64 or B128, hsum by slice and lane for the float kernels);
 * tx = textbook (one species per element type, minimal code, reduceLanes, scalar tail), with a different widening step where cur has more than one
 * (dotI8: byte straight to int with B2I; block16: two dimensions per step, eight accumulators);
 * cv = no convertShape anywhere: one 128-bit byte load and ByteVector.convert(B2x, part) (16 bytes = 4 float vectors, or 2 shorts then 4 int vectors).
 * The cv loops step by 16 bytes, so up to 15 trailing elements run in the scalar tail.
 */
final class Vc8A {
    private Vc8A() {}

    static final VectorSpecies<Float> F128 = FloatVector.SPECIES_128;
    static final VectorSpecies<Integer> I128 = IntVector.SPECIES_128;
    static final VectorSpecies<Short> S128 = ShortVector.SPECIES_128;
    static final VectorSpecies<Byte> B64 = ByteVector.SPECIES_64;
    static final VectorSpecies<Byte> B128 = ByteVector.SPECIES_128;
    static final int L = 4;

    /** same as VKernels.hsum: log2 steps of shift-and-add, result in lane 0 */
    static float hsum(FloatVector v) {
        for (int h = L / 2; h >= 1; h >>= 1) v = v.add(v.slice(h));
        return v.lane(0);
    }

    // ---- dotI8: int8 x int8 -> int32 ----

    static int dotI8cur(byte[] a, int ao, byte[] b, int bo, int n) {
        IntVector acc = IntVector.zero(I128);
        int i = 0;
        int end = n - (n % 8);
        for (; i < end; i += 8) {
            ShortVector as = (ShortVector) ByteVector.fromArray(B64, a, ao + i).convertShape(VectorOperators.B2S, S128, 0);
            ShortVector bs = (ShortVector) ByteVector.fromArray(B64, b, bo + i).convertShape(VectorOperators.B2S, S128, 0);
            ShortVector p = as.mul(bs);
            acc = acc.add((IntVector) p.convertShape(VectorOperators.S2I, I128, 0)).add((IntVector) p.convertShape(VectorOperators.S2I, I128, 1));
        }
        int s = acc.reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) s += a[ao + i] * b[bo + i];
        return s;
    }

    static int dotI8tx(byte[] a, int ao, byte[] b, int bo, int n) {
        IntVector acc = IntVector.zero(I128);
        int i = 0;
        int end = n - (n % 8);
        for (; i < end; i += 8) {
            ByteVector av = ByteVector.fromArray(B64, a, ao + i);
            ByteVector bv = ByteVector.fromArray(B64, b, bo + i);
            IntVector a0 = (IntVector) av.convertShape(VectorOperators.B2I, I128, 0);
            IntVector a1 = (IntVector) av.convertShape(VectorOperators.B2I, I128, 1);
            IntVector b0 = (IntVector) bv.convertShape(VectorOperators.B2I, I128, 0);
            IntVector b1 = (IntVector) bv.convertShape(VectorOperators.B2I, I128, 1);
            acc = acc.add(a0.mul(b0)).add(a1.mul(b1));
        }
        int s = acc.reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) s += a[ao + i] * b[bo + i];
        return s;
    }

    static int dotI8cv(byte[] a, int ao, byte[] b, int bo, int n) {
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

    // ---- dotF8: float query x int8 codes, one accumulator ----

    static float dotF8cur(float[] q, int qo, byte[] c, int co, int n) {
        FloatVector a0 = FloatVector.zero(F128);
        int i = 0;
        int end = n - (n % 8);
        for (; i < end; i += 8) {
            ByteVector cb = ByteVector.fromArray(B64, c, co + i);
            FloatVector c0 = (FloatVector) cb.convertShape(VectorOperators.B2F, F128, 0);
            FloatVector c1 = (FloatVector) cb.convertShape(VectorOperators.B2F, F128, 1);
            a0 = FloatVector.fromArray(F128, q, qo + i).fma(c0, a0);
            a0 = FloatVector.fromArray(F128, q, qo + i + 4).fma(c1, a0);
        }
        float s = hsum(a0);
        for (; i < n; i++) s += q[qo + i] * c[co + i];
        return s;
    }

    static float dotF8tx(float[] q, int qo, byte[] c, int co, int n) {
        FloatVector a0 = FloatVector.zero(F128);
        int i = 0;
        int end = n - (n % 8);
        for (; i < end; i += 8) {
            ByteVector cb = ByteVector.fromArray(B64, c, co + i);
            FloatVector c0 = (FloatVector) cb.convertShape(VectorOperators.B2F, F128, 0);
            FloatVector c1 = (FloatVector) cb.convertShape(VectorOperators.B2F, F128, 1);
            a0 = FloatVector.fromArray(F128, q, qo + i).fma(c0, a0);
            a0 = FloatVector.fromArray(F128, q, qo + i + 4).fma(c1, a0);
        }
        float s = a0.reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) s += q[qo + i] * c[co + i];
        return s;
    }

    static float dotF8cv(float[] q, int qo, byte[] c, int co, int n) {
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

    // ---- dotF8x4: same with four accumulators ----

    static float dotF8x4cur(float[] q, int qo, byte[] c, int co, int n) {
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
        float s = hsum(a0.add(a1).add(a2.add(a3)));
        for (; i < n; i++) s += q[qo + i] * c[co + i];
        return s;
    }

    static float dotF8x4tx(float[] q, int qo, byte[] c, int co, int n) {
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

    static float dotF8x4cv(float[] q, int qo, byte[] c, int co, int n) {
        FloatVector a0 = FloatVector.zero(F128), a1 = FloatVector.zero(F128), a2 = FloatVector.zero(F128), a3 = FloatVector.zero(F128);
        int i = 0;
        int end = n - (n % 16);
        for (; i < end; i += 16) {
            ByteVector cb = ByteVector.fromArray(B128, c, co + i);
            a0 = FloatVector.fromArray(F128, q, qo + i).fma((FloatVector) cb.convert(VectorOperators.B2F, 0), a0);
            a1 = FloatVector.fromArray(F128, q, qo + i + 4).fma((FloatVector) cb.convert(VectorOperators.B2F, 1), a1);
            a2 = FloatVector.fromArray(F128, q, qo + i + 8).fma((FloatVector) cb.convert(VectorOperators.B2F, 2), a2);
            a3 = FloatVector.fromArray(F128, q, qo + i + 12).fma((FloatVector) cb.convert(VectorOperators.B2F, 3), a3);
        }
        float s = a0.add(a1).add(a2.add(a3)).reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) s += q[qo + i] * c[co + i];
        return s;
    }

    // ---- block16: dimension-major block of 16 vectors, element (dimension j, lane) at base + j * 16 + lane ----

    static void block16cur(float[] q, int qo, byte[] c, int base, int n, float[] out) {
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

    static void block16tx(float[] q, int qo, byte[] c, int base, int n, float[] out) {
        FloatVector v0 = FloatVector.zero(F128), v1 = FloatVector.zero(F128), v2 = FloatVector.zero(F128), v3 = FloatVector.zero(F128);
        FloatVector w0 = FloatVector.zero(F128), w1 = FloatVector.zero(F128), w2 = FloatVector.zero(F128), w3 = FloatVector.zero(F128);
        int j = 0;
        int end = n - (n % 2);
        for (; j < end; j += 2) {
            FloatVector q0 = FloatVector.broadcast(F128, q[qo + j]);
            FloatVector q1 = FloatVector.broadcast(F128, q[qo + j + 1]);
            ByteVector b0 = ByteVector.fromArray(B128, c, base + j * 16);
            ByteVector b1 = ByteVector.fromArray(B128, c, base + j * 16 + 16);
            v0 = q0.fma((FloatVector) b0.convertShape(VectorOperators.B2F, F128, 0), v0);
            v1 = q0.fma((FloatVector) b0.convertShape(VectorOperators.B2F, F128, 1), v1);
            v2 = q0.fma((FloatVector) b0.convertShape(VectorOperators.B2F, F128, 2), v2);
            v3 = q0.fma((FloatVector) b0.convertShape(VectorOperators.B2F, F128, 3), v3);
            w0 = q1.fma((FloatVector) b1.convertShape(VectorOperators.B2F, F128, 0), w0);
            w1 = q1.fma((FloatVector) b1.convertShape(VectorOperators.B2F, F128, 1), w1);
            w2 = q1.fma((FloatVector) b1.convertShape(VectorOperators.B2F, F128, 2), w2);
            w3 = q1.fma((FloatVector) b1.convertShape(VectorOperators.B2F, F128, 3), w3);
        }
        if (j < n) {
            FloatVector qj = FloatVector.broadcast(F128, q[qo + j]);
            ByteVector b = ByteVector.fromArray(B128, c, base + j * 16);
            v0 = qj.fma((FloatVector) b.convertShape(VectorOperators.B2F, F128, 0), v0);
            v1 = qj.fma((FloatVector) b.convertShape(VectorOperators.B2F, F128, 1), v1);
            v2 = qj.fma((FloatVector) b.convertShape(VectorOperators.B2F, F128, 2), v2);
            v3 = qj.fma((FloatVector) b.convertShape(VectorOperators.B2F, F128, 3), v3);
        }
        v0.add(w0).intoArray(out, 0);
        v1.add(w1).intoArray(out, 4);
        v2.add(w2).intoArray(out, 8);
        v3.add(w3).intoArray(out, 12);
    }

    static void block16cv(float[] q, int qo, byte[] c, int base, int n, float[] out) {
        FloatVector v0 = FloatVector.zero(F128), v1 = FloatVector.zero(F128), v2 = FloatVector.zero(F128), v3 = FloatVector.zero(F128);
        for (int j = 0; j < n; j++) {
            FloatVector qj = FloatVector.broadcast(F128, q[qo + j]);
            ByteVector b = ByteVector.fromArray(B128, c, base + j * 16);
            v0 = qj.fma((FloatVector) b.convert(VectorOperators.B2F, 0), v0);
            v1 = qj.fma((FloatVector) b.convert(VectorOperators.B2F, 1), v1);
            v2 = qj.fma((FloatVector) b.convert(VectorOperators.B2F, 2), v2);
            v3 = qj.fma((FloatVector) b.convert(VectorOperators.B2F, 3), v3);
        }
        v0.intoArray(out, 0);
        v1.intoArray(out, 4);
        v2.intoArray(out, 8);
        v3.intoArray(out, 12);
    }
}
