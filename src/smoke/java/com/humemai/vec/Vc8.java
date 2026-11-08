package com.humemai.vec;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Experiment round 5 (branch native-perf-experiment only): the four VKernels kernels that use convertShape (dotI8, dotF8, dotF8x4, block16),
 * x86-64, 256-bit float species, all species static final. No kernel here contains a mask (VKernels.dotI8, dotF8, dotF8x4 and block16 have none).
 * Variants per kernel:
 * cur = the VKernels form with the species fixed (convertShape from a 64-bit or 128-bit byte species, hsum by slice and lane for the float kernels);
 * tx = textbook (one species per element type, minimal code, reduceLanes, scalar tail), with a different widening step where cur has more than one
 * (dotI8: byte straight to int with B2I; block16: two dimensions per step, four accumulators);
 * cv = no convertShape anywhere: one 256-bit byte load and ByteVector.convert(B2x, part) (32 bytes = 4 float vectors, or 2 shorts then 4 int vectors).
 * The cv loops step by 32 bytes, so up to 31 trailing elements run in the scalar tail.
 */
final class Vc8 {
    private Vc8() {}

    static final VectorSpecies<Float> F256 = FloatVector.SPECIES_256;
    static final VectorSpecies<Integer> I256 = IntVector.SPECIES_256;
    static final VectorSpecies<Short> S256 = ShortVector.SPECIES_256;
    static final VectorSpecies<Byte> B64 = ByteVector.SPECIES_64;
    static final VectorSpecies<Byte> B128 = ByteVector.SPECIES_128;
    static final VectorSpecies<Byte> B256 = ByteVector.SPECIES_256;
    static final int L = 8;

    /** same as VKernels.hsum: log2 steps of shift-and-add, result in lane 0 */
    static float hsum(FloatVector v) {
        for (int h = L / 2; h >= 1; h >>= 1) v = v.add(v.slice(h));
        return v.lane(0);
    }

    // ---- dotI8: int8 x int8 -> int32 ----

    static int dotI8cur(byte[] a, int ao, byte[] b, int bo, int n) {
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

    static int dotI8tx(byte[] a, int ao, byte[] b, int bo, int n) {
        IntVector acc = IntVector.zero(I256);
        int i = 0;
        int end = n - (n % 8);
        for (; i < end; i += 8) {
            IntVector av = (IntVector) ByteVector.fromArray(B64, a, ao + i).convertShape(VectorOperators.B2I, I256, 0);
            IntVector bv = (IntVector) ByteVector.fromArray(B64, b, bo + i).convertShape(VectorOperators.B2I, I256, 0);
            acc = acc.add(av.mul(bv));
        }
        int s = acc.reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) s += a[ao + i] * b[bo + i];
        return s;
    }

    static int dotI8cv(byte[] a, int ao, byte[] b, int bo, int n) {
        IntVector acc = IntVector.zero(I256);
        int i = 0;
        int end = n - (n % 32);
        for (; i < end; i += 32) {
            ByteVector av = ByteVector.fromArray(B256, a, ao + i);
            ByteVector bv = ByteVector.fromArray(B256, b, bo + i);
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
        FloatVector a0 = FloatVector.zero(F256);
        int i = 0;
        int end = n - (n % L);
        for (; i < end; i += L) {
            FloatVector cv = (FloatVector) ByteVector.fromArray(B64, c, co + i).convertShape(VectorOperators.B2F, F256, 0);
            a0 = FloatVector.fromArray(F256, q, qo + i).fma(cv, a0);
        }
        float s = hsum(a0);
        for (; i < n; i++) s += q[qo + i] * c[co + i];
        return s;
    }

    static float dotF8tx(float[] q, int qo, byte[] c, int co, int n) {
        FloatVector a0 = FloatVector.zero(F256);
        int i = 0;
        int end = n - (n % 8);
        for (; i < end; i += 8) {
            FloatVector cv = (FloatVector) ByteVector.fromArray(B64, c, co + i).convertShape(VectorOperators.B2F, F256, 0);
            a0 = FloatVector.fromArray(F256, q, qo + i).fma(cv, a0);
        }
        float s = a0.reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) s += q[qo + i] * c[co + i];
        return s;
    }

    static float dotF8cv(float[] q, int qo, byte[] c, int co, int n) {
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

    // ---- dotF8x4: same with four accumulators ----

    static float dotF8x4cur(float[] q, int qo, byte[] c, int co, int n) {
        FloatVector a0 = FloatVector.zero(F256), a1 = FloatVector.zero(F256), a2 = FloatVector.zero(F256), a3 = FloatVector.zero(F256);
        int i = 0;
        int end4 = n - (n % (4 * L));
        for (; i < end4; i += 4 * L) {
            FloatVector c0 = (FloatVector) ByteVector.fromArray(B64, c, co + i).convertShape(VectorOperators.B2F, F256, 0);
            FloatVector c1 = (FloatVector) ByteVector.fromArray(B64, c, co + i + L).convertShape(VectorOperators.B2F, F256, 0);
            FloatVector c2 = (FloatVector) ByteVector.fromArray(B64, c, co + i + 2 * L).convertShape(VectorOperators.B2F, F256, 0);
            FloatVector c3 = (FloatVector) ByteVector.fromArray(B64, c, co + i + 3 * L).convertShape(VectorOperators.B2F, F256, 0);
            a0 = FloatVector.fromArray(F256, q, qo + i).fma(c0, a0);
            a1 = FloatVector.fromArray(F256, q, qo + i + L).fma(c1, a1);
            a2 = FloatVector.fromArray(F256, q, qo + i + 2 * L).fma(c2, a2);
            a3 = FloatVector.fromArray(F256, q, qo + i + 3 * L).fma(c3, a3);
        }
        int end1 = n - (n % L);
        for (; i < end1; i += L) {
            FloatVector cv = (FloatVector) ByteVector.fromArray(B64, c, co + i).convertShape(VectorOperators.B2F, F256, 0);
            a0 = FloatVector.fromArray(F256, q, qo + i).fma(cv, a0);
        }
        float s = hsum(a0.add(a1).add(a2.add(a3)));
        for (; i < n; i++) s += q[qo + i] * c[co + i];
        return s;
    }

    static float dotF8x4tx(float[] q, int qo, byte[] c, int co, int n) {
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

    static float dotF8x4cv(float[] q, int qo, byte[] c, int co, int n) {
        FloatVector a0 = FloatVector.zero(F256), a1 = FloatVector.zero(F256), a2 = FloatVector.zero(F256), a3 = FloatVector.zero(F256);
        int i = 0;
        int end = n - (n % 32);
        for (; i < end; i += 32) {
            ByteVector cb = ByteVector.fromArray(B256, c, co + i);
            a0 = FloatVector.fromArray(F256, q, qo + i).fma((FloatVector) cb.convert(VectorOperators.B2F, 0), a0);
            a1 = FloatVector.fromArray(F256, q, qo + i + 8).fma((FloatVector) cb.convert(VectorOperators.B2F, 1), a1);
            a2 = FloatVector.fromArray(F256, q, qo + i + 16).fma((FloatVector) cb.convert(VectorOperators.B2F, 2), a2);
            a3 = FloatVector.fromArray(F256, q, qo + i + 24).fma((FloatVector) cb.convert(VectorOperators.B2F, 3), a3);
        }
        float s = a0.add(a1).add(a2.add(a3)).reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) s += q[qo + i] * c[co + i];
        return s;
    }

    // ---- block16: dimension-major block of 16 vectors, element (dimension j, lane) at base + j * 16 + lane ----

    static void block16cur(float[] q, int qo, byte[] c, int base, int n, float[] out) {
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

    static void block16tx(float[] q, int qo, byte[] c, int base, int n, float[] out) {
        FloatVector lo0 = FloatVector.zero(F256), hi0 = FloatVector.zero(F256), lo1 = FloatVector.zero(F256), hi1 = FloatVector.zero(F256);
        int j = 0;
        int end = n - (n % 2);
        for (; j < end; j += 2) {
            FloatVector q0 = FloatVector.broadcast(F256, q[qo + j]);
            FloatVector q1 = FloatVector.broadcast(F256, q[qo + j + 1]);
            ByteVector b0 = ByteVector.fromArray(B128, c, base + j * 16);
            ByteVector b1 = ByteVector.fromArray(B128, c, base + j * 16 + 16);
            lo0 = q0.fma((FloatVector) b0.convertShape(VectorOperators.B2F, F256, 0), lo0);
            hi0 = q0.fma((FloatVector) b0.convertShape(VectorOperators.B2F, F256, 1), hi0);
            lo1 = q1.fma((FloatVector) b1.convertShape(VectorOperators.B2F, F256, 0), lo1);
            hi1 = q1.fma((FloatVector) b1.convertShape(VectorOperators.B2F, F256, 1), hi1);
        }
        if (j < n) {
            FloatVector qj = FloatVector.broadcast(F256, q[qo + j]);
            ByteVector b = ByteVector.fromArray(B128, c, base + j * 16);
            lo0 = qj.fma((FloatVector) b.convertShape(VectorOperators.B2F, F256, 0), lo0);
            hi0 = qj.fma((FloatVector) b.convertShape(VectorOperators.B2F, F256, 1), hi0);
        }
        lo0.add(lo1).intoArray(out, 0);
        hi0.add(hi1).intoArray(out, 8);
    }

    static void block16cv(float[] q, int qo, byte[] c, int base, int n, float[] out) {
        FloatVector lo0 = FloatVector.zero(F256), hi0 = FloatVector.zero(F256), lo1 = FloatVector.zero(F256), hi1 = FloatVector.zero(F256);
        int j = 0;
        int end = n - (n % 2);
        for (; j < end; j += 2) {
            FloatVector q0 = FloatVector.broadcast(F256, q[qo + j]);
            FloatVector q1 = FloatVector.broadcast(F256, q[qo + j + 1]);
            ByteVector b = ByteVector.fromArray(B256, c, base + j * 16);
            lo0 = q0.fma((FloatVector) b.convert(VectorOperators.B2F, 0), lo0);
            hi0 = q0.fma((FloatVector) b.convert(VectorOperators.B2F, 1), hi0);
            lo1 = q1.fma((FloatVector) b.convert(VectorOperators.B2F, 2), lo1);
            hi1 = q1.fma((FloatVector) b.convert(VectorOperators.B2F, 3), hi1);
        }
        lo0.add(lo1).intoArray(out, 0);
        hi0.add(hi1).intoArray(out, 8);
        if (j < n) {
            float qj = q[qo + j];
            for (int l = 0; l < 16; l++) out[l] += qj * c[base + j * 16 + l];
        }
    }
}
