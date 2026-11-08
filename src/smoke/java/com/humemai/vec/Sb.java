package com.humemai.vec;

/**
 * Experiment round 5 (branch native-perf-experiment only): plain scalar loops for the byte kernels, the baseline. dotI81 and blk1 are
 * Kernels.dotI8Scalar and Kernels.block16Scalar, dotF84 is Kernels.dotF8Scalar. The digit is the number of independent accumulators
 * (dot kernels) or the number of dimensions folded per pass over the 16 lanes (block kernel).
 */
final class Sb {
    private Sb() {}

    static int dotI81(byte[] a, int ao, byte[] b, int bo, int n) {
        int s = 0;
        for (int i = 0; i < n; i++) s += a[ao + i] * b[bo + i];
        return s;
    }

    static int dotI84(byte[] a, int ao, byte[] b, int bo, int n) {
        int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
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

    static int dotI88(byte[] a, int ao, byte[] b, int bo, int n) {
        int s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            s0 += a[ao + i] * b[bo + i];
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

    static float dotF81(float[] q, int qo, byte[] c, int co, int n) {
        float s = 0;
        for (int i = 0; i < n; i++) s += q[qo + i] * c[co + i];
        return s;
    }

    static float dotF84(float[] q, int qo, byte[] c, int co, int n) {
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

    static float dotF88(float[] q, int qo, byte[] c, int co, int n) {
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            s0 += q[qo + i] * c[co + i];
            s1 += q[qo + i + 1] * c[co + i + 1];
            s2 += q[qo + i + 2] * c[co + i + 2];
            s3 += q[qo + i + 3] * c[co + i + 3];
            s4 += q[qo + i + 4] * c[co + i + 4];
            s5 += q[qo + i + 5] * c[co + i + 5];
            s6 += q[qo + i + 6] * c[co + i + 6];
            s7 += q[qo + i + 7] * c[co + i + 7];
        }
        for (; i < n; i++) s0 += q[qo + i] * c[co + i];
        return ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
    }

    static void blk1(float[] q, int qo, byte[] c, int base, int n, float[] out) {
        for (int lane = 0; lane < 16; lane++) out[lane] = 0f;
        for (int j = 0; j < n; j++) {
            float qj = q[qo + j];
            int b = base + j * 16;
            for (int lane = 0; lane < 16; lane++) out[lane] += qj * c[b + lane];
        }
    }

    static void blk4(float[] q, int qo, byte[] c, int base, int n, float[] out) {
        for (int lane = 0; lane < 16; lane++) out[lane] = 0f;
        int j = 0;
        for (; j + 4 <= n; j += 4) {
            float q0 = q[qo + j], q1 = q[qo + j + 1], q2 = q[qo + j + 2], q3 = q[qo + j + 3];
            int b = base + j * 16;
            for (int lane = 0; lane < 16; lane++) {
                out[lane] += (q0 * c[b + lane] + q1 * c[b + 16 + lane]) + (q2 * c[b + 32 + lane] + q3 * c[b + 48 + lane]);
            }
        }
        for (; j < n; j++) {
            float qj = q[qo + j];
            int b = base + j * 16;
            for (int lane = 0; lane < 16; lane++) out[lane] += qj * c[b + lane];
        }
    }

    static void blk8(float[] q, int qo, byte[] c, int base, int n, float[] out) {
        for (int lane = 0; lane < 16; lane++) out[lane] = 0f;
        int j = 0;
        for (; j + 8 <= n; j += 8) {
            float q0 = q[qo + j], q1 = q[qo + j + 1], q2 = q[qo + j + 2], q3 = q[qo + j + 3];
            float q4 = q[qo + j + 4], q5 = q[qo + j + 5], q6 = q[qo + j + 6], q7 = q[qo + j + 7];
            int b = base + j * 16;
            for (int lane = 0; lane < 16; lane++) {
                float t0 = (q0 * c[b + lane] + q1 * c[b + 16 + lane]) + (q2 * c[b + 32 + lane] + q3 * c[b + 48 + lane]);
                float t1 = (q4 * c[b + 64 + lane] + q5 * c[b + 80 + lane]) + (q6 * c[b + 96 + lane] + q7 * c[b + 112 + lane]);
                out[lane] += t0 + t1;
            }
        }
        for (; j < n; j++) {
            float qj = q[qo + j];
            int b = base + j * 16;
            for (int lane = 0; lane < 16; lane++) out[lane] += qj * c[b + lane];
        }
    }
}
