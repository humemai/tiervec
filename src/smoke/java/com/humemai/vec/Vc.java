package com.humemai.vec;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/** Experiment round 2 (branch native-perf-experiment only; generated): textbook Vector API kernels with static final species, independent of VKernels. */
final class Vc {
    private Vc() {}

    static final VectorSpecies<Float> S256 = FloatVector.SPECIES_256;
    static final VectorSpecies<Float> S512 = FloatVector.SPECIES_512;

    static float dot256(float[] a, int ao, float[] b, int bo, int n) {
        FloatVector a0 = FloatVector.zero(S256), a1 = FloatVector.zero(S256);
        int i = 0;
        int end2 = n - (n % 16);
        for (; i < end2; i += 16) {
            a0 = FloatVector.fromArray(S256, a, ao + i).fma(FloatVector.fromArray(S256, b, bo + i), a0);
            a1 = FloatVector.fromArray(S256, a, ao + i + 8).fma(FloatVector.fromArray(S256, b, bo + i + 8), a1);
        }
        int end1 = n - (n % 8);
        for (; i < end1; i += 8) {
            a0 = FloatVector.fromArray(S256, a, ao + i).fma(FloatVector.fromArray(S256, b, bo + i), a0);
        }
        float s = a0.add(a1).reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) s += a[ao + i] * b[bo + i];
        return s;
    }

    static float l2sq256(float[] a, int ao, float[] b, int bo, int n) {
        FloatVector a0 = FloatVector.zero(S256), a1 = FloatVector.zero(S256);
        int i = 0;
        int end2 = n - (n % 16);
        for (; i < end2; i += 16) {
            FloatVector d0 = FloatVector.fromArray(S256, a, ao + i).sub(FloatVector.fromArray(S256, b, bo + i));
            FloatVector d1 = FloatVector.fromArray(S256, a, ao + i + 8).sub(FloatVector.fromArray(S256, b, bo + i + 8));
            a0 = d0.fma(d0, a0);
            a1 = d1.fma(d1, a1);
        }
        int end1 = n - (n % 8);
        for (; i < end1; i += 8) {
            FloatVector d0 = FloatVector.fromArray(S256, a, ao + i).sub(FloatVector.fromArray(S256, b, bo + i));
            a0 = d0.fma(d0, a0);
        }
        float s = a0.add(a1).reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) {
            float d = a[ao + i] - b[bo + i];
            s += d * d;
        }
        return s;
    }

    static float dot512(float[] a, int ao, float[] b, int bo, int n) {
        FloatVector a0 = FloatVector.zero(S512), a1 = FloatVector.zero(S512);
        int i = 0;
        int end2 = n - (n % 32);
        for (; i < end2; i += 32) {
            a0 = FloatVector.fromArray(S512, a, ao + i).fma(FloatVector.fromArray(S512, b, bo + i), a0);
            a1 = FloatVector.fromArray(S512, a, ao + i + 16).fma(FloatVector.fromArray(S512, b, bo + i + 16), a1);
        }
        int end1 = n - (n % 16);
        for (; i < end1; i += 16) {
            a0 = FloatVector.fromArray(S512, a, ao + i).fma(FloatVector.fromArray(S512, b, bo + i), a0);
        }
        float s = a0.add(a1).reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) s += a[ao + i] * b[bo + i];
        return s;
    }

    static float l2sq512(float[] a, int ao, float[] b, int bo, int n) {
        FloatVector a0 = FloatVector.zero(S512), a1 = FloatVector.zero(S512);
        int i = 0;
        int end2 = n - (n % 32);
        for (; i < end2; i += 32) {
            FloatVector d0 = FloatVector.fromArray(S512, a, ao + i).sub(FloatVector.fromArray(S512, b, bo + i));
            FloatVector d1 = FloatVector.fromArray(S512, a, ao + i + 16).sub(FloatVector.fromArray(S512, b, bo + i + 16));
            a0 = d0.fma(d0, a0);
            a1 = d1.fma(d1, a1);
        }
        int end1 = n - (n % 16);
        for (; i < end1; i += 16) {
            FloatVector d0 = FloatVector.fromArray(S512, a, ao + i).sub(FloatVector.fromArray(S512, b, bo + i));
            a0 = d0.fma(d0, a0);
        }
        float s = a0.add(a1).reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) {
            float d = a[ao + i] - b[bo + i];
            s += d * d;
        }
        return s;
    }

    /** same loops as the 256-bit kernels, but the horizontal sum is the slice-based one of VKernels.hsum, unrolled by hand for 8 lanes */
    static float hsum8(FloatVector v) {
        v = v.add(v.slice(4));
        v = v.add(v.slice(2));
        v = v.add(v.slice(1));
        return v.lane(0);
    }

    static float dotH256(float[] a, int ao, float[] b, int bo, int n) {
        FloatVector a0 = FloatVector.zero(S256), a1 = FloatVector.zero(S256);
        int i = 0;
        int end2 = n - (n % 16);
        for (; i < end2; i += 16) {
            a0 = FloatVector.fromArray(S256, a, ao + i).fma(FloatVector.fromArray(S256, b, bo + i), a0);
            a1 = FloatVector.fromArray(S256, a, ao + i + 8).fma(FloatVector.fromArray(S256, b, bo + i + 8), a1);
        }
        int end1 = n - (n % 8);
        for (; i < end1; i += 8) {
            a0 = FloatVector.fromArray(S256, a, ao + i).fma(FloatVector.fromArray(S256, b, bo + i), a0);
        }
        float s = hsum8(a0.add(a1));
        for (; i < n; i++) s += a[ao + i] * b[bo + i];
        return s;
    }

    static float l2sqH256(float[] a, int ao, float[] b, int bo, int n) {
        FloatVector a0 = FloatVector.zero(S256), a1 = FloatVector.zero(S256);
        int i = 0;
        int end2 = n - (n % 16);
        for (; i < end2; i += 16) {
            FloatVector d0 = FloatVector.fromArray(S256, a, ao + i).sub(FloatVector.fromArray(S256, b, bo + i));
            FloatVector d1 = FloatVector.fromArray(S256, a, ao + i + 8).sub(FloatVector.fromArray(S256, b, bo + i + 8));
            a0 = d0.fma(d0, a0);
            a1 = d1.fma(d1, a1);
        }
        int end1 = n - (n % 8);
        for (; i < end1; i += 8) {
            FloatVector d0 = FloatVector.fromArray(S256, a, ao + i).sub(FloatVector.fromArray(S256, b, bo + i));
            a0 = d0.fma(d0, a0);
        }
        float s = hsum8(a0.add(a1));
        for (; i < n; i++) {
            float d = a[ao + i] - b[bo + i];
            s += d * d;
        }
        return s;
    }

    // ---- round 4: fixed-order horizontal sums (L, A) with a scalar tail, and reduceLanes with a masked tail (M); all on the 256-bit species ----

    /** round 4: fixed-order horizontal sum, explicit pairwise tree over v.lane(i) */
    static float hsumL256(FloatVector v) {
        return (((v.lane(0) + v.lane(1)) + (v.lane(2) + v.lane(3))) + ((v.lane(4) + v.lane(5)) + (v.lane(6) + v.lane(7))));
    }

    /** round 4: fixed-order horizontal sum, v.intoArray into a per-call float[] (escape analysis may remove it), then the same pairwise tree in scalar code */
    static float hsumA256(FloatVector v) {
        float[] t = new float[8];
        v.intoArray(t, 0);
        return (((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7])));
    }

    static float dotL256(float[] a, int ao, float[] b, int bo, int n) {
        FloatVector a0 = FloatVector.zero(S256), a1 = FloatVector.zero(S256);
        int i = 0;
        int end2 = n - (n % (2 * 8));
        for (; i < end2; i += 2 * 8) {
            a0 = FloatVector.fromArray(S256, a, ao + i).fma(FloatVector.fromArray(S256, b, bo + i), a0);
            a1 = FloatVector.fromArray(S256, a, ao + i + 8).fma(FloatVector.fromArray(S256, b, bo + i + 8), a1);
        }
        int end1 = n - (n % 8);
        for (; i < end1; i += 8) {
            a0 = FloatVector.fromArray(S256, a, ao + i).fma(FloatVector.fromArray(S256, b, bo + i), a0);
        }
        float s = hsumL256(a0.add(a1));
        for (; i < n; i++) s += a[ao + i] * b[bo + i];
        return s;
    }

    static float l2sqL256(float[] a, int ao, float[] b, int bo, int n) {
        FloatVector a0 = FloatVector.zero(S256), a1 = FloatVector.zero(S256);
        int i = 0;
        int end2 = n - (n % (2 * 8));
        for (; i < end2; i += 2 * 8) {
            FloatVector d0 = FloatVector.fromArray(S256, a, ao + i).sub(FloatVector.fromArray(S256, b, bo + i));
            FloatVector d1 = FloatVector.fromArray(S256, a, ao + i + 8).sub(FloatVector.fromArray(S256, b, bo + i + 8));
            a0 = d0.fma(d0, a0);
            a1 = d1.fma(d1, a1);
        }
        int end1 = n - (n % 8);
        for (; i < end1; i += 8) {
            FloatVector d0 = FloatVector.fromArray(S256, a, ao + i).sub(FloatVector.fromArray(S256, b, bo + i));
            a0 = d0.fma(d0, a0);
        }
        float s = hsumL256(a0.add(a1));
        for (; i < n; i++) {
            float d = a[ao + i] - b[bo + i];
            s += d * d;
        }
        return s;
    }

    static float dotA256(float[] a, int ao, float[] b, int bo, int n) {
        FloatVector a0 = FloatVector.zero(S256), a1 = FloatVector.zero(S256);
        int i = 0;
        int end2 = n - (n % (2 * 8));
        for (; i < end2; i += 2 * 8) {
            a0 = FloatVector.fromArray(S256, a, ao + i).fma(FloatVector.fromArray(S256, b, bo + i), a0);
            a1 = FloatVector.fromArray(S256, a, ao + i + 8).fma(FloatVector.fromArray(S256, b, bo + i + 8), a1);
        }
        int end1 = n - (n % 8);
        for (; i < end1; i += 8) {
            a0 = FloatVector.fromArray(S256, a, ao + i).fma(FloatVector.fromArray(S256, b, bo + i), a0);
        }
        float s = hsumA256(a0.add(a1));
        for (; i < n; i++) s += a[ao + i] * b[bo + i];
        return s;
    }

    static float l2sqA256(float[] a, int ao, float[] b, int bo, int n) {
        FloatVector a0 = FloatVector.zero(S256), a1 = FloatVector.zero(S256);
        int i = 0;
        int end2 = n - (n % (2 * 8));
        for (; i < end2; i += 2 * 8) {
            FloatVector d0 = FloatVector.fromArray(S256, a, ao + i).sub(FloatVector.fromArray(S256, b, bo + i));
            FloatVector d1 = FloatVector.fromArray(S256, a, ao + i + 8).sub(FloatVector.fromArray(S256, b, bo + i + 8));
            a0 = d0.fma(d0, a0);
            a1 = d1.fma(d1, a1);
        }
        int end1 = n - (n % 8);
        for (; i < end1; i += 8) {
            FloatVector d0 = FloatVector.fromArray(S256, a, ao + i).sub(FloatVector.fromArray(S256, b, bo + i));
            a0 = d0.fma(d0, a0);
        }
        float s = hsumA256(a0.add(a1));
        for (; i < n; i++) {
            float d = a[ao + i] - b[bo + i];
            s += d * d;
        }
        return s;
    }

    static float dotM256(float[] a, int ao, float[] b, int bo, int n) {
        FloatVector a0 = FloatVector.zero(S256), a1 = FloatVector.zero(S256);
        int i = 0;
        int end2 = n - (n % (2 * 8));
        for (; i < end2; i += 2 * 8) {
            a0 = FloatVector.fromArray(S256, a, ao + i).fma(FloatVector.fromArray(S256, b, bo + i), a0);
            a1 = FloatVector.fromArray(S256, a, ao + i + 8).fma(FloatVector.fromArray(S256, b, bo + i + 8), a1);
        }
        int end1 = n - (n % 8);
        for (; i < end1; i += 8) {
            a0 = FloatVector.fromArray(S256, a, ao + i).fma(FloatVector.fromArray(S256, b, bo + i), a0);
        }
        if (i < n) {
            VectorMask<Float> m = S256.indexInRange(i, n);
            a0 = FloatVector.fromArray(S256, a, ao + i, m).fma(FloatVector.fromArray(S256, b, bo + i, m), a0);
        }
        return a0.add(a1).reduceLanes(VectorOperators.ADD);
    }

    static float l2sqM256(float[] a, int ao, float[] b, int bo, int n) {
        FloatVector a0 = FloatVector.zero(S256), a1 = FloatVector.zero(S256);
        int i = 0;
        int end2 = n - (n % (2 * 8));
        for (; i < end2; i += 2 * 8) {
            FloatVector d0 = FloatVector.fromArray(S256, a, ao + i).sub(FloatVector.fromArray(S256, b, bo + i));
            FloatVector d1 = FloatVector.fromArray(S256, a, ao + i + 8).sub(FloatVector.fromArray(S256, b, bo + i + 8));
            a0 = d0.fma(d0, a0);
            a1 = d1.fma(d1, a1);
        }
        int end1 = n - (n % 8);
        for (; i < end1; i += 8) {
            FloatVector d0 = FloatVector.fromArray(S256, a, ao + i).sub(FloatVector.fromArray(S256, b, bo + i));
            a0 = d0.fma(d0, a0);
        }
        if (i < n) {
            VectorMask<Float> m = S256.indexInRange(i, n);
            FloatVector d0 = FloatVector.fromArray(S256, a, ao + i, m).sub(FloatVector.fromArray(S256, b, bo + i, m));
            a0 = d0.fma(d0, a0);
        }
        return a0.add(a1).reduceLanes(VectorOperators.ADD);
    }
}
