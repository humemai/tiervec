package com.humemai.vec;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/** Experiment round 3 (branch native-perf-experiment only; generated): textbook Vector API kernels for SPECIES_64, 128, 256, 512 and PREFERRED, all static final. */
final class VcA {
    private VcA() {}

    static final VectorSpecies<Float> S64 = FloatVector.SPECIES_64;
    static final int L64 = 2;
    static final VectorSpecies<Float> S128 = FloatVector.SPECIES_128;
    static final int L128 = 4;
    static final VectorSpecies<Float> S256 = FloatVector.SPECIES_256;
    static final int L256 = 8;
    static final VectorSpecies<Float> S512 = FloatVector.SPECIES_512;
    static final int L512 = 16;
    static final VectorSpecies<Float> Sp = FloatVector.SPECIES_PREFERRED;
    static final int Lp = Sp.length();

    static float dot64(float[] a, int ao, float[] b, int bo, int n) {
        FloatVector a0 = FloatVector.zero(S64), a1 = FloatVector.zero(S64);
        int i = 0;
        int end2 = n - (n % (2 * L64));
        for (; i < end2; i += 2 * L64) {
            a0 = FloatVector.fromArray(S64, a, ao + i).fma(FloatVector.fromArray(S64, b, bo + i), a0);
            a1 = FloatVector.fromArray(S64, a, ao + i + L64).fma(FloatVector.fromArray(S64, b, bo + i + L64), a1);
        }
        int end1 = n - (n % L64);
        for (; i < end1; i += L64) {
            a0 = FloatVector.fromArray(S64, a, ao + i).fma(FloatVector.fromArray(S64, b, bo + i), a0);
        }
        float s = a0.add(a1).reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) s += a[ao + i] * b[bo + i];
        return s;
    }

    static float l2sq64(float[] a, int ao, float[] b, int bo, int n) {
        FloatVector a0 = FloatVector.zero(S64), a1 = FloatVector.zero(S64);
        int i = 0;
        int end2 = n - (n % (2 * L64));
        for (; i < end2; i += 2 * L64) {
            FloatVector d0 = FloatVector.fromArray(S64, a, ao + i).sub(FloatVector.fromArray(S64, b, bo + i));
            FloatVector d1 = FloatVector.fromArray(S64, a, ao + i + L64).sub(FloatVector.fromArray(S64, b, bo + i + L64));
            a0 = d0.fma(d0, a0);
            a1 = d1.fma(d1, a1);
        }
        int end1 = n - (n % L64);
        for (; i < end1; i += L64) {
            FloatVector d0 = FloatVector.fromArray(S64, a, ao + i).sub(FloatVector.fromArray(S64, b, bo + i));
            a0 = d0.fma(d0, a0);
        }
        float s = a0.add(a1).reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) {
            float d = a[ao + i] - b[bo + i];
            s += d * d;
        }
        return s;
    }

    static float dot128(float[] a, int ao, float[] b, int bo, int n) {
        FloatVector a0 = FloatVector.zero(S128), a1 = FloatVector.zero(S128);
        int i = 0;
        int end2 = n - (n % (2 * L128));
        for (; i < end2; i += 2 * L128) {
            a0 = FloatVector.fromArray(S128, a, ao + i).fma(FloatVector.fromArray(S128, b, bo + i), a0);
            a1 = FloatVector.fromArray(S128, a, ao + i + L128).fma(FloatVector.fromArray(S128, b, bo + i + L128), a1);
        }
        int end1 = n - (n % L128);
        for (; i < end1; i += L128) {
            a0 = FloatVector.fromArray(S128, a, ao + i).fma(FloatVector.fromArray(S128, b, bo + i), a0);
        }
        float s = a0.add(a1).reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) s += a[ao + i] * b[bo + i];
        return s;
    }

    static float l2sq128(float[] a, int ao, float[] b, int bo, int n) {
        FloatVector a0 = FloatVector.zero(S128), a1 = FloatVector.zero(S128);
        int i = 0;
        int end2 = n - (n % (2 * L128));
        for (; i < end2; i += 2 * L128) {
            FloatVector d0 = FloatVector.fromArray(S128, a, ao + i).sub(FloatVector.fromArray(S128, b, bo + i));
            FloatVector d1 = FloatVector.fromArray(S128, a, ao + i + L128).sub(FloatVector.fromArray(S128, b, bo + i + L128));
            a0 = d0.fma(d0, a0);
            a1 = d1.fma(d1, a1);
        }
        int end1 = n - (n % L128);
        for (; i < end1; i += L128) {
            FloatVector d0 = FloatVector.fromArray(S128, a, ao + i).sub(FloatVector.fromArray(S128, b, bo + i));
            a0 = d0.fma(d0, a0);
        }
        float s = a0.add(a1).reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) {
            float d = a[ao + i] - b[bo + i];
            s += d * d;
        }
        return s;
    }

    static float dot256(float[] a, int ao, float[] b, int bo, int n) {
        FloatVector a0 = FloatVector.zero(S256), a1 = FloatVector.zero(S256);
        int i = 0;
        int end2 = n - (n % (2 * L256));
        for (; i < end2; i += 2 * L256) {
            a0 = FloatVector.fromArray(S256, a, ao + i).fma(FloatVector.fromArray(S256, b, bo + i), a0);
            a1 = FloatVector.fromArray(S256, a, ao + i + L256).fma(FloatVector.fromArray(S256, b, bo + i + L256), a1);
        }
        int end1 = n - (n % L256);
        for (; i < end1; i += L256) {
            a0 = FloatVector.fromArray(S256, a, ao + i).fma(FloatVector.fromArray(S256, b, bo + i), a0);
        }
        float s = a0.add(a1).reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) s += a[ao + i] * b[bo + i];
        return s;
    }

    static float l2sq256(float[] a, int ao, float[] b, int bo, int n) {
        FloatVector a0 = FloatVector.zero(S256), a1 = FloatVector.zero(S256);
        int i = 0;
        int end2 = n - (n % (2 * L256));
        for (; i < end2; i += 2 * L256) {
            FloatVector d0 = FloatVector.fromArray(S256, a, ao + i).sub(FloatVector.fromArray(S256, b, bo + i));
            FloatVector d1 = FloatVector.fromArray(S256, a, ao + i + L256).sub(FloatVector.fromArray(S256, b, bo + i + L256));
            a0 = d0.fma(d0, a0);
            a1 = d1.fma(d1, a1);
        }
        int end1 = n - (n % L256);
        for (; i < end1; i += L256) {
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
        int end2 = n - (n % (2 * L512));
        for (; i < end2; i += 2 * L512) {
            a0 = FloatVector.fromArray(S512, a, ao + i).fma(FloatVector.fromArray(S512, b, bo + i), a0);
            a1 = FloatVector.fromArray(S512, a, ao + i + L512).fma(FloatVector.fromArray(S512, b, bo + i + L512), a1);
        }
        int end1 = n - (n % L512);
        for (; i < end1; i += L512) {
            a0 = FloatVector.fromArray(S512, a, ao + i).fma(FloatVector.fromArray(S512, b, bo + i), a0);
        }
        float s = a0.add(a1).reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) s += a[ao + i] * b[bo + i];
        return s;
    }

    static float l2sq512(float[] a, int ao, float[] b, int bo, int n) {
        FloatVector a0 = FloatVector.zero(S512), a1 = FloatVector.zero(S512);
        int i = 0;
        int end2 = n - (n % (2 * L512));
        for (; i < end2; i += 2 * L512) {
            FloatVector d0 = FloatVector.fromArray(S512, a, ao + i).sub(FloatVector.fromArray(S512, b, bo + i));
            FloatVector d1 = FloatVector.fromArray(S512, a, ao + i + L512).sub(FloatVector.fromArray(S512, b, bo + i + L512));
            a0 = d0.fma(d0, a0);
            a1 = d1.fma(d1, a1);
        }
        int end1 = n - (n % L512);
        for (; i < end1; i += L512) {
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

    static float dotp(float[] a, int ao, float[] b, int bo, int n) {
        FloatVector a0 = FloatVector.zero(Sp), a1 = FloatVector.zero(Sp);
        int i = 0;
        int end2 = n - (n % (2 * Lp));
        for (; i < end2; i += 2 * Lp) {
            a0 = FloatVector.fromArray(Sp, a, ao + i).fma(FloatVector.fromArray(Sp, b, bo + i), a0);
            a1 = FloatVector.fromArray(Sp, a, ao + i + Lp).fma(FloatVector.fromArray(Sp, b, bo + i + Lp), a1);
        }
        int end1 = n - (n % Lp);
        for (; i < end1; i += Lp) {
            a0 = FloatVector.fromArray(Sp, a, ao + i).fma(FloatVector.fromArray(Sp, b, bo + i), a0);
        }
        float s = a0.add(a1).reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) s += a[ao + i] * b[bo + i];
        return s;
    }

    static float l2sqp(float[] a, int ao, float[] b, int bo, int n) {
        FloatVector a0 = FloatVector.zero(Sp), a1 = FloatVector.zero(Sp);
        int i = 0;
        int end2 = n - (n % (2 * Lp));
        for (; i < end2; i += 2 * Lp) {
            FloatVector d0 = FloatVector.fromArray(Sp, a, ao + i).sub(FloatVector.fromArray(Sp, b, bo + i));
            FloatVector d1 = FloatVector.fromArray(Sp, a, ao + i + Lp).sub(FloatVector.fromArray(Sp, b, bo + i + Lp));
            a0 = d0.fma(d0, a0);
            a1 = d1.fma(d1, a1);
        }
        int end1 = n - (n % Lp);
        for (; i < end1; i += Lp) {
            FloatVector d0 = FloatVector.fromArray(Sp, a, ao + i).sub(FloatVector.fromArray(Sp, b, bo + i));
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
