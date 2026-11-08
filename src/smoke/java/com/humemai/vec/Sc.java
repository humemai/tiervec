package com.humemai.vec;

/** Experiment round 2 (branch native-perf-experiment only; generated): plain scalar loops with 1, 4 and 8 independent accumulators. */
final class Sc {
    private Sc() {}

    static float dot1(float[] a, int ao, float[] b, int bo, int n) {
        float s0 = 0;
        int i = 0;
        for (; i < n; i++) s0 += a[ao + i] * b[bo + i];
        return s0;
    }

    static float l2sq1(float[] a, int ao, float[] b, int bo, int n) {
        float s0 = 0;
        int i = 0;
        for (; i < n; i++) {
            float d = a[ao + i] - b[bo + i];
            s0 += d * d;
        }
        return s0;
    }
    static float dot4(float[] a, int ao, float[] b, int bo, int n) {
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int i = 0;
        for (; i + 4 <= n; i += 4) {
            s0 += a[ao + i + 0] * b[bo + i + 0];
            s1 += a[ao + i + 1] * b[bo + i + 1];
            s2 += a[ao + i + 2] * b[bo + i + 2];
            s3 += a[ao + i + 3] * b[bo + i + 3];
        }
        for (; i < n; i++) s0 += a[ao + i] * b[bo + i];
        return (s0 + s1) + (s2 + s3);
    }

    static float l2sq4(float[] a, int ao, float[] b, int bo, int n) {
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int i = 0;
        for (; i + 4 <= n; i += 4) {
            float d0 = a[ao + i + 0] - b[bo + i + 0];
            float d1 = a[ao + i + 1] - b[bo + i + 1];
            float d2 = a[ao + i + 2] - b[bo + i + 2];
            float d3 = a[ao + i + 3] - b[bo + i + 3];
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
    static float dot8(float[] a, int ao, float[] b, int bo, int n) {
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            s0 += a[ao + i + 0] * b[bo + i + 0];
            s1 += a[ao + i + 1] * b[bo + i + 1];
            s2 += a[ao + i + 2] * b[bo + i + 2];
            s3 += a[ao + i + 3] * b[bo + i + 3];
            s4 += a[ao + i + 4] * b[bo + i + 4];
            s5 += a[ao + i + 5] * b[bo + i + 5];
            s6 += a[ao + i + 6] * b[bo + i + 6];
            s7 += a[ao + i + 7] * b[bo + i + 7];
        }
        for (; i < n; i++) s0 += a[ao + i] * b[bo + i];
        return ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
    }

    static float l2sq8(float[] a, int ao, float[] b, int bo, int n) {
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            float d0 = a[ao + i + 0] - b[bo + i + 0];
            float d1 = a[ao + i + 1] - b[bo + i + 1];
            float d2 = a[ao + i + 2] - b[bo + i + 2];
            float d3 = a[ao + i + 3] - b[bo + i + 3];
            float d4 = a[ao + i + 4] - b[bo + i + 4];
            float d5 = a[ao + i + 5] - b[bo + i + 5];
            float d6 = a[ao + i + 6] - b[bo + i + 6];
            float d7 = a[ao + i + 7] - b[bo + i + 7];
            s0 += d0 * d0;
            s1 += d1 * d1;
            s2 += d2 * d2;
            s3 += d3 * d3;
            s4 += d4 * d4;
            s5 += d5 * d5;
            s6 += d6 * d6;
            s7 += d7 * d7;
        }
        for (; i < n; i++) {
            float d = a[ao + i] - b[bo + i];
            s0 += d * d;
        }
        return ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
    }
}
