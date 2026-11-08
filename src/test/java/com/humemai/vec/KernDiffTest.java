package com.humemai.vec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.IntStream;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Differential test of the graph kernels: the Vector API version ({@link VKern}) against the scalar twin ({@link Kernels#dotScalar},
 * {@link Kernels#l2sqScalar}, {@link Kernels#dotF8Scalar}), for dot, squared L2 and the SQ8 dot (a float query against signed int8 codes).
 * Dimensions include 0, 1, every size around the lane width (so the masked-free tails of both versions run) and sizes that are not a
 * multiple of it. Runs in the default surefire execution, where the Vector API kernels are active.
 *
 * The routing of the int8, SQ8 and block kernels on a JVM with 128-bit float lanes (aarch64) is tested in {@link JvmQuantRoutingTest}.
 *
 * The kernels of a native image ({@link NKernels}: fp32 dot and squared L2, int8 dot, the SQ8 dots, block16) are tested here on the JVM against the same
 * scalar twins: every form of each kernel (the one {@link NKernels#WIDE} selects and both explicit forms, so the 128-bit forms of aarch64 run on an x86
 * host too), with the lengths of the native-image tests, exact for integer data and within {@link #N_REL_TOL} plus {@link #N_ABS_FLOOR} for random data.
 *
 * Documented tolerance, {@link #REL_TOL}: two float results agree when they differ by at most 1e-5 times the sum of the absolute
 * terms of the sum (so a dot product that cancels to nearly zero is not held to a relative error of its own value). A float sum of n
 * terms in any order is within n * 2^-24 of the exact one in that unit, about 1e-4 at n = 1536 in the worst case; the random data here
 * is far from the worst case (a typical difference is 1e-7). Exactly representable data (small integers) must agree bit for bit, which
 * tests the index arithmetic of the loops and tails without any rounding.
 */
class KernDiffTest {
    static final double REL_TOL = 1e-5;
    static final int AO = 3, BO = 5, CO = 7;

    @BeforeAll
    static void setUp() {
        assertTrue(Kernels.VECTOR, "the Vector API kernels must be the ones under test (is --add-modules jdk.incubator.vector set?)");
    }

    static int[] dims() {
        Set<Integer> s = new TreeSet<>(List.of(0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 257, 768, 1000, 1536));
        int l = VKern.L;
        for (int d : new int[] {l - 1, l, l + 1, 2 * l - 1, 2 * l, 2 * l + 1, 4 * l + 3}) s.add(d);
        return s.stream().mapToInt(Integer::intValue).toArray();
    }

    static float[] rnd(Random r, int n) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) v[i] = r.nextFloat() * 2f - 1f;
        return v;
    }

    static byte[] codes(Random r, int n) {
        byte[] c = new byte[n];
        for (int i = 0; i < n; i++) c[i] = (byte) (r.nextInt(255) - 127);
        return c;
    }

    static void within(double x, double y, double scale, String what) {
        assertTrue(Math.abs(x - y) <= REL_TOL * scale, what + ": " + x + " against " + y + ", scale " + scale + ", tolerance " + REL_TOL * scale);
    }

    @Test
    void integerDataAgreesBitForBit() {
        Random r = new Random(11);
        for (int n : dims()) {
            // values outside the window are large, so a read past the window or before its offset changes the sum
            float[] a = new float[AO + n + 2], b = new float[BO + n + 2], q = new float[n];
            byte[] c = new byte[CO + n + 2];
            Arrays.fill(a, 1000f);
            Arrays.fill(b, 1000f);
            Arrays.fill(c, (byte) 127);
            long dot = 0, l2 = 0, dq = 0;
            for (int i = 0; i < n; i++) {
                int x = r.nextInt(17) - 8, y = r.nextInt(17) - 8, z = r.nextInt(256) - 128, w = r.nextInt(17) - 8;
                a[AO + i] = x;
                b[BO + i] = y;
                c[CO + i] = (byte) z;
                q[i] = w;
                dot += (long) x * y;
                l2 += (long) (x - y) * (x - y);
                dq += (long) w * z;
            }
            String m = " n=" + n;
            assertEquals((float) dot, VKern.dot(a, AO, b, BO, n), 0f, "vector dot" + m);
            assertEquals((float) dot, Kernels.dotScalar(a, AO, b, BO, n), 0f, "scalar dot" + m);
            assertEquals((float) l2, VKern.l2(a, AO, b, BO, n), 0f, "vector l2" + m);
            assertEquals((float) l2, Kernels.l2sqScalar(a, AO, b, BO, n), 0f, "scalar l2" + m);
            if (VKern.Q8_OK) assertEquals((float) dq, VKern.dotQ8(q, c, CO, n), 0f, "vector dotQ8" + m);
            assertEquals((float) dq, Kernels.dotF8Scalar(q, 0, c, CO, n), 0f, "scalar dotQ8" + m);
        }
    }

    @Test
    void randomDataAgreesWithinTolerance() {
        Random r = new Random(12);
        for (int n : dims()) {
            for (int rep = 0; rep < 20; rep++) {
                float[] a = rnd(r, n), b = rnd(r, n), q = rnd(r, n);
                byte[] c = codes(r, n);
                double dot = 0, dotAbs = 0, l2 = 0, dq = 0, dqAbs = 0;
                for (int i = 0; i < n; i++) {
                    dot += (double) a[i] * b[i];
                    dotAbs += Math.abs((double) a[i] * b[i]);
                    double d = (double) a[i] - b[i];
                    l2 += d * d;
                    dq += (double) q[i] * c[i];
                    dqAbs += Math.abs((double) q[i] * c[i]);
                }
                String m = " n=" + n;
                float vd = VKern.dot(a, 0, b, 0, n), sd = Kernels.dotScalar(a, 0, b, 0, n);
                within(vd, sd, dotAbs, "dot, vector against scalar" + m);
                within(vd, dot, dotAbs, "dot, vector against exact" + m);
                within(sd, dot, dotAbs, "dot, scalar against exact" + m);
                float vl = VKern.l2(a, 0, b, 0, n), sl = Kernels.l2sqScalar(a, 0, b, 0, n);
                within(vl, sl, l2, "l2, vector against scalar" + m);
                within(vl, l2, l2, "l2, vector against exact" + m);
                within(sl, l2, l2, "l2, scalar against exact" + m);
                float sq = Kernels.dotF8Scalar(q, 0, c, 0, n);
                within(sq, dq, dqAbs, "dotQ8, scalar against exact" + m);
                if (VKern.Q8_OK) {
                    float vq = VKern.dotQ8(q, c, 0, n);
                    within(vq, sq, dqAbs, "dotQ8, vector against scalar" + m);
                    within(vq, dq, dqAbs, "dotQ8, vector against exact" + m);
                }
            }
        }
    }

    @Test
    void zeroVectorsGiveZero() {
        Random r = new Random(13);
        for (int n : dims()) {
            float[] z = new float[n], x = rnd(r, n);
            byte[] cz = new byte[n], cx = codes(r, n);
            double xx = 0;
            for (float f : x) xx += (double) f * f;
            String m = " n=" + n;
            assertEquals(0f, VKern.dot(z, 0, x, 0, n), 0f, "vector dot, zero first" + m);
            assertEquals(0f, VKern.dot(x, 0, z, 0, n), 0f, "vector dot, zero second" + m);
            assertEquals(0f, Kernels.dotScalar(z, 0, x, 0, n), 0f, "scalar dot, zero first" + m);
            assertEquals(0f, Kernels.dotScalar(x, 0, z, 0, n), 0f, "scalar dot, zero second" + m);
            assertEquals(0f, VKern.l2(z, 0, z, 0, n), 0f, "vector l2 of two zero vectors" + m);
            assertEquals(0f, Kernels.l2sqScalar(z, 0, z, 0, n), 0f, "scalar l2 of two zero vectors" + m);
            within(VKern.l2(z, 0, x, 0, n), xx, xx, "vector l2 of a zero vector and x" + m);
            within(Kernels.l2sqScalar(z, 0, x, 0, n), xx, xx, "scalar l2 of a zero vector and x" + m);
            assertEquals(0f, Kernels.dotF8Scalar(z, 0, cx, 0, n), 0f, "scalar dotQ8, zero query" + m);
            assertEquals(0f, Kernels.dotF8Scalar(x, 0, cz, 0, n), 0f, "scalar dotQ8, zero codes" + m);
            if (VKern.Q8_OK) {
                assertEquals(0f, VKern.dotQ8(z, cx, 0, n), 0f, "vector dotQ8, zero query" + m);
                assertEquals(0f, VKern.dotQ8(x, cz, 0, n), 0f, "vector dotQ8, zero codes" + m);
            }
        }
    }

    // ---- NKernels: the fp32 kernels of a native image, tested here on the JVM against the scalar twins ----

    /** relative tolerance of NKernels against the scalar twin and the exact value, times the sum of the absolute terms, plus an absolute floor */
    static final double N_REL_TOL = 1e-4, N_ABS_FLOOR = 1e-6;

    static int[] nDims() {
        Set<Integer> s = new TreeSet<>(List.of(0, 1, 7, 8, 9, 100, 128, 770));
        int l = NKernels.L;
        for (int d : new int[] {l - 1, l, l + 1, 2 * l - 1, 2 * l, 2 * l + 1, 4 * l + 3}) s.add(d);
        return s.stream().mapToInt(Integer::intValue).toArray();
    }

    static void withinN(double x, double y, double scale, String what) {
        double tol = N_REL_TOL * scale + N_ABS_FLOOR;
        assertTrue(Math.abs(x - y) <= tol, what + ": " + x + " against " + y + ", scale " + scale + ", tolerance " + tol);
    }

    @Test
    void nKernelsSpeciesIsChosenFromTheArchitecture() {
        assertTrue(NKernels.S != null, "NKernels has no species: " + NKernels.ERROR);
        String arch = System.getProperty("os.arch", "");
        boolean arm = arch.equals("aarch64") || arch.equals("arm64");
        assertEquals(arm ? 4 : 8, NKernels.L, "lanes for os.arch " + arch);
        assertEquals(!arm, NKernels.WIDE, "wide form for os.arch " + arch);
        assertEquals(NKernels.S.length(), NKernels.L, "lanes of the species");
    }

    @Test
    void nKernelsIntegerDataAgreesBitForBit() {
        Random r = new Random(21);
        for (int n : nDims()) {
            // values outside the window are large, so a read past the window or before its offset changes the sum
            float[] a = new float[AO + n + 2], b = new float[BO + n + 2];
            Arrays.fill(a, 1000f);
            Arrays.fill(b, 1000f);
            long dot = 0, l2 = 0;
            for (int i = 0; i < n; i++) {
                int x = r.nextInt(17) - 8, y = r.nextInt(17) - 8;
                a[AO + i] = x;
                b[BO + i] = y;
                dot += (long) x * y;
                l2 += (long) (x - y) * (x - y);
            }
            String m = " n=" + n;
            assertEquals((float) dot, NKernels.dot(a, AO, b, BO, n), 0f, "NKernels dot" + m);
            assertEquals((float) l2, NKernels.l2sq(a, AO, b, BO, n), 0f, "NKernels l2sq" + m);
        }
    }

    @Test
    void nKernelsRandomDataAgreesWithinTolerance() {
        Random r = new Random(22);
        for (int n : nDims()) {
            for (int rep = 0; rep < 20; rep++) {
                float[] a = rnd(r, n), b = rnd(r, n);
                double dot = 0, dotAbs = 0, l2 = 0;
                for (int i = 0; i < n; i++) {
                    dot += (double) a[i] * b[i];
                    dotAbs += Math.abs((double) a[i] * b[i]);
                    double d = (double) a[i] - b[i];
                    l2 += d * d;
                }
                String m = " n=" + n;
                float nd = NKernels.dot(a, 0, b, 0, n), sd = Kernels.dotScalar(a, 0, b, 0, n);
                withinN(nd, sd, dotAbs, "dot, NKernels against scalar" + m);
                withinN(nd, dot, dotAbs, "dot, NKernels against exact" + m);
                float nl = NKernels.l2sq(a, 0, b, 0, n), sl = Kernels.l2sqScalar(a, 0, b, 0, n);
                withinN(nl, sl, l2, "l2sq, NKernels against scalar" + m);
                withinN(nl, l2, l2, "l2sq, NKernels against exact" + m);
                // a call on equal data at another offset gives the same bits
                float[] a2 = new float[n + 3], b2 = new float[n + 5];
                System.arraycopy(a, 0, a2, 3, n);
                System.arraycopy(b, 0, b2, 5, n);
                assertEquals(Float.floatToIntBits(nd), Float.floatToIntBits(NKernels.dot(a2, 3, b2, 5, n)), "dot at offsets" + m);
                assertEquals(Float.floatToIntBits(nl), Float.floatToIntBits(NKernels.l2sq(a2, 3, b2, 5, n)), "l2sq at offsets" + m);
            }
        }
    }

    @Test
    void nKernelsZeroVectorsGiveZero() {
        Random r = new Random(23);
        for (int n : nDims()) {
            float[] z = new float[n], x = rnd(r, n);
            double xx = 0;
            for (float f : x) xx += (double) f * f;
            String m = " n=" + n;
            assertEquals(0f, NKernels.dot(z, 0, x, 0, n), 0f, "NKernels dot, zero first" + m);
            assertEquals(0f, NKernels.l2sq(z, 0, z, 0, n), 0f, "NKernels l2sq of two zero vectors" + m);
            withinN(NKernels.l2sq(z, 0, x, 0, n), xx, xx, "NKernels l2sq of a zero vector and x" + m);
        }
    }

    @Test
    void inAJvmTheKernelsDoNotUseNKernels() {
        // on a JVM Kernels and Kern use VKernels and VKern, bit for bit
        assertTrue(!Kernels.IN_IMAGE && Kernels.JVM_VECTOR, "this test runs on a JVM with the Vector API");
        Random r = new Random(24);
        float[] a = rnd(r, 770), b = rnd(r, 770);
        assertEquals(Float.floatToIntBits(VKernels.dot(a, 0, b, 0, 770)), Float.floatToIntBits(Kernels.dot(a, 0, b, 0, 770)), "Kernels.dot");
        assertEquals(Float.floatToIntBits(VKernels.l2sq(a, 0, b, 0, 770)), Float.floatToIntBits(Kernels.l2sq(a, 0, b, 0, 770)), "Kernels.l2sq");
    }

    // ---- NKernels: the int8, SQ8 and block kernels of a native image (issue #48), tested here on the JVM against the scalar twins ----

    interface I8Kernel { int apply(byte[] a, int ao, byte[] b, int bo, int n); }

    interface F8Kernel { float apply(float[] q, int qo, byte[] c, int co, int n); }

    interface BlockKernel { void apply(float[] q, int qo, byte[] c, int base, int n, float[] out); }

    /** the lengths of the native-image tests: 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 128 and 770, and 0 */
    static final int[] Q_DIMS = {0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 128, 770};
    static final int[] BLOCK_DIMS = {0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 33, 100, 128, 770};

    static Map<String, I8Kernel> i8Forms() {
        Map<String, I8Kernel> m = new LinkedHashMap<>();
        m.put("NKernels.dotI8", NKernels::dotI8);
        m.put("NKernels.dotI8W", NKernels::dotI8W);
        m.put("NKernels.dotI8N", NKernels::dotI8N);
        return m;
    }

    static Map<String, F8Kernel> f8Forms() {
        Map<String, F8Kernel> m = new LinkedHashMap<>();
        m.put("NKernels.dotF8", NKernels::dotF8);
        m.put("NKernels.dotF8W", NKernels::dotF8W);
        m.put("NKernels.dotF8N", NKernels::dotF8N);
        m.put("NKernels.dotF8x4", NKernels::dotF8x4);
        m.put("NKernels.dotF8x4W", NKernels::dotF8x4W);
        m.put("NKernels.dotF8x4N", NKernels::dotF8x4N);
        return m;
    }

    static Map<String, BlockKernel> blockForms() {
        Map<String, BlockKernel> m = new LinkedHashMap<>();
        m.put("NKernels.block16", NKernels::block16);
        m.put("NKernels.block16W", NKernels::block16W);
        m.put("NKernels.block16N", NKernels::block16N);
        return m;
    }

    @Test
    void nKernelsDotI8IsExact() {
        Random r = new Random(31);
        for (int n : Q_DIMS) {
            // bytes outside the window are 127, so a read past the window or before its offset changes the sum
            byte[] a = new byte[AO + n + 2], b = new byte[BO + n + 2];
            Arrays.fill(a, (byte) 127);
            Arrays.fill(b, (byte) 127);
            for (int i = 0; i < n; i++) {
                a[AO + i] = (byte) (r.nextInt(255) - 127);
                b[BO + i] = (byte) (r.nextInt(255) - 127);
            }
            int ref = Kernels.dotI8Scalar(a, AO, b, BO, n);
            for (Map.Entry<String, I8Kernel> e : i8Forms().entrySet()) assertEquals(ref, e.getValue().apply(a, AO, b, BO, n), e.getKey() + " n=" + n);
            // the extremes: every product is -127 * 127, which fits a short
            for (int i = 0; i < n; i++) {
                a[AO + i] = (byte) 127;
                b[BO + i] = (byte) -127;
            }
            for (Map.Entry<String, I8Kernel> e : i8Forms().entrySet()) assertEquals(-127 * 127 * n, e.getValue().apply(a, AO, b, BO, n), e.getKey() + " extreme n=" + n);
        }
    }

    @Test
    void nKernelsSq8IntegerDataAgreesBitForBit() {
        Random r = new Random(32);
        for (int n : Q_DIMS) {
            float[] q = new float[AO + n + 2];
            byte[] c = new byte[CO + n + 2];
            Arrays.fill(q, 1000f);
            Arrays.fill(c, (byte) 127);
            long dq = 0;
            for (int i = 0; i < n; i++) {
                int w = r.nextInt(17) - 8, z = r.nextInt(256) - 128;
                q[AO + i] = w;
                c[CO + i] = (byte) z;
                dq += (long) w * z;
            }
            for (Map.Entry<String, F8Kernel> e : f8Forms().entrySet()) assertEquals((float) dq, e.getValue().apply(q, AO, c, CO, n), 0f, e.getKey() + " n=" + n);
            assertEquals((float) dq, Kernels.dotF8Scalar(q, AO, c, CO, n), 0f, "scalar twin n=" + n);
        }
    }

    @Test
    void nKernelsSq8RandomDataAgreesWithinTolerance() {
        Random r = new Random(33);
        for (int n : Q_DIMS) {
            for (int rep = 0; rep < 20; rep++) {
                float[] q = rnd(r, n);
                byte[] c = codes(r, n);
                double dq = 0, dqAbs = 0;
                for (int i = 0; i < n; i++) {
                    dq += (double) q[i] * c[i];
                    dqAbs += Math.abs((double) q[i] * c[i]);
                }
                float sq = Kernels.dotF8Scalar(q, 0, c, 0, n);
                float[] q2 = new float[n + 3];
                byte[] c2 = new byte[n + 7];
                System.arraycopy(q, 0, q2, 3, n);
                System.arraycopy(c, 0, c2, 7, n);
                for (Map.Entry<String, F8Kernel> e : f8Forms().entrySet()) {
                    String m = e.getKey() + " n=" + n;
                    float v = e.getValue().apply(q, 0, c, 0, n);
                    withinN(v, sq, dqAbs, m + ", against scalar");
                    withinN(v, dq, dqAbs, m + ", against exact");
                    // a call on equal data at other offsets gives the same bits
                    assertEquals(Float.floatToIntBits(v), Float.floatToIntBits(e.getValue().apply(q2, 3, c2, 7, n)), m + " at offsets");
                }
            }
        }
    }

    @Test
    void nKernelsBlock16IntegerAndRandomData() {
        Random r = new Random(34);
        int qo = 3, base = 5;
        for (int n : BLOCK_DIMS) {
            for (int rep = 0; rep < 10; rep++) {
                boolean ints = rep < 3;
                // outside the window q is 1000 and the codes are 127, so a read past the window or before its offset changes a sum
                float[] q = new float[qo + n + 2];
                byte[] c = new byte[base + n * 16 + 3];
                Arrays.fill(q, 1000f);
                Arrays.fill(c, (byte) 127);
                for (int j = 0; j < n; j++) q[qo + j] = ints ? r.nextInt(17) - 8 : r.nextFloat() * 2f - 1f;
                for (int i = 0; i < n * 16; i++) c[base + i] = (byte) (r.nextInt(255) - 127);
                float[] ref = new float[16];
                Kernels.block16Scalar(q, qo, c, base, n, ref);
                for (Map.Entry<String, BlockKernel> e : blockForms().entrySet()) {
                    float[] out = new float[16];
                    Arrays.fill(out, 12345f); // every lane must be written
                    e.getValue().apply(q, qo, c, base, n, out);
                    for (int lane = 0; lane < 16; lane++) {
                        double ex = 0, exAbs = 0;
                        for (int j = 0; j < n; j++) {
                            ex += (double) q[qo + j] * c[base + j * 16 + lane];
                            exAbs += Math.abs((double) q[qo + j] * c[base + j * 16 + lane]);
                        }
                        String m = e.getKey() + " n=" + n + " lane " + lane + (ints ? " (integer data)" : "");
                        if (ints) {
                            assertEquals((float) ex, out[lane], 0f, m);
                        } else {
                            withinN(out[lane], ref[lane], exAbs, m + ", against scalar");
                            withinN(out[lane], ex, exAbs, m + ", against exact");
                        }
                    }
                }
            }
        }
    }

    @Test
    void nKernelsDispatchFollowsTheSpecies() {
        Random r = new Random(35);
        int n = 770;
        float[] q = rnd(r, n);
        byte[] c = codes(r, n), d = codes(r, n), blk = codes(r, n * 16);
        assertEquals(NKernels.WIDE ? NKernels.dotI8W(c, 0, d, 0, n) : NKernels.dotI8N(c, 0, d, 0, n), NKernels.dotI8(c, 0, d, 0, n), "dotI8");
        assertEquals(bits(NKernels.WIDE ? NKernels.dotF8W(q, 0, c, 0, n) : NKernels.dotF8N(q, 0, c, 0, n)), bits(NKernels.dotF8(q, 0, c, 0, n)), "dotF8");
        assertEquals(bits(NKernels.WIDE ? NKernels.dotF8x4W(q, 0, c, 0, n) : NKernels.dotF8x4N(q, 0, c, 0, n)), bits(NKernels.dotF8x4(q, 0, c, 0, n)), "dotF8x4");
        float[] o1 = new float[16], o2 = new float[16];
        if (NKernels.WIDE) NKernels.block16W(q, 0, blk, 0, n, o1);
        else NKernels.block16N(q, 0, blk, 0, n, o1);
        NKernels.block16(q, 0, blk, 0, n, o2);
        assertEquals(Arrays.toString(o1), Arrays.toString(o2), "block16");
    }

    @Test
    void inAJvmTheQuantKernelsDoNotUseNKernels() {
        // on a JVM with 8 or more float lanes Kernels uses VKernels for the int8, SQ8 and block kernels (the JVM path is unchanged), bit for bit;
        // with fewer lanes (NEON on aarch64) it uses the 128-bit forms of NKernels or the scalar twins, see JvmQuantRoutingTest
        assertTrue(!Kernels.IN_IMAGE && Kernels.JVM_VECTOR && !Kernels.NATIVE_VECTOR, "this test runs on a JVM with the Vector API");
        Assumptions.assumeTrue(Kernels.JVM_QUANT_VK, "the 128-bit float lanes of this host route these kernels away from VKernels (JvmQuantRoutingTest)");
        Random r = new Random(36);
        int n = 770;
        float[] q = rnd(r, n);
        byte[] c = codes(r, n), d = codes(r, n), blk = codes(r, n * 16);
        assertEquals(VKernels.dotI8(c, 0, d, 0, n), Kernels.dotI8(c, 0, d, 0, n), "Kernels.dotI8");
        if (VKernels.F8_OK) {
            assertEquals(bits(VKernels.dotF8(q, 0, c, 0, n)), bits(Kernels.dotF8(q, 0, c, 0, n)), "Kernels.dotF8");
            assertEquals(bits(VKernels.dotF8x4(q, 0, c, 0, n)), bits(Kernels.dotF8x4(q, 0, c, 0, n)), "Kernels.dotF8x4");
        }
        if (VKernels.BLOCK_OK) {
            float[] o1 = new float[16], o2 = new float[16];
            VKernels.block16(q, 0, blk, 0, n, o1);
            Kernels.block16(q, 0, blk, 0, n, o2);
            assertEquals(Arrays.toString(o1), Arrays.toString(o2), "Kernels.block16");
        }
    }

    static int bits(float f) {
        return Float.floatToIntBits(f);
    }

    /** the k best by the score (smaller is better), ties by position */
    static int[] top(float[] score, int k) {
        Integer[] idx = IntStream.range(0, score.length).boxed().toArray(Integer[]::new);
        Arrays.sort(idx, Comparator.comparingDouble((Integer i) -> score[i]).thenComparingInt(i -> i));
        int[] t = new int[k];
        for (int i = 0; i < k; i++) t[i] = idx[i];
        return t;
    }

    @Test
    void top10SetsOfAFixedDatasetAreEqual() {
        int n = 3000, dim = 33, nq = 50, k = 10;
        Random r = new Random(14);
        float[][] data = new float[n][];
        for (int i = 0; i < n; i++) data[i] = rnd(r, dim);
        for (boolean l2 : new boolean[] {true, false}) {
            int flips = 0;
            for (int qi = 0; qi < nq; qi++) {
                float[] q = rnd(r, dim);
                float[] vs = new float[n], ss = new float[n];
                for (int i = 0; i < n; i++) {
                    vs[i] = l2 ? VKern.l2(q, 0, data[i], 0, dim) : -VKern.dot(q, 0, data[i], 0, dim);
                    ss[i] = l2 ? Kernels.l2sqScalar(q, 0, data[i], 0, dim) : -Kernels.dotScalar(q, 0, data[i], 0, dim);
                }
                int[] tv = top(vs, k), ts = top(ss, k);
                Set<Integer> sv = new HashSet<>(), sc = new HashSet<>();
                for (int i = 0; i < k; i++) {
                    sv.add(tv[i]);
                    sc.add(ts[i]);
                }
                if (sv.equals(sc)) continue;
                // a tie may flip: every id in the difference must score within the tolerance of the 10th best score
                flips++;
                float kth = ss[ts[k - 1]];
                Set<Integer> diff = new HashSet<>(sv);
                diff.addAll(sc);
                Set<Integer> both = new HashSet<>(sv);
                both.retainAll(sc);
                diff.removeAll(both);
                for (int id : diff) {
                    assertTrue(Math.abs(ss[id] - kth) <= REL_TOL * (Math.abs(kth) + 1), (l2 ? "l2" : "dot") + " query " + qi + ": id " + id + " differs and does not tie with the 10th score " + kth);
                }
            }
            System.out.println("  top-10 sets, " + (l2 ? "l2" : "dot") + ": " + (nq - flips) + " of " + nq + " equal, " + flips + " flipped on a tie");
        }
    }

    @Test
    void kernUsesTheVectorKernelsInThisMode() {
        Random r = new Random(15);
        int n = 129;
        float[] a = rnd(r, n), b = rnd(r, n);
        byte[] c = codes(r, n);
        assertEquals(VKern.L, Kern.lanes(), "lanes");
        assertEquals(Float.floatToIntBits(VKern.dot(a, 0, b, 0, n)), Float.floatToIntBits(Kern.dot(a, 0, b, 0, n)), "Kern.dot");
        assertEquals(Float.floatToIntBits(VKern.l2(a, 0, b, 0, n)), Float.floatToIntBits(Kern.l2(a, 0, b, 0, n)), "Kern.l2");
        if (VKern.Q8_OK) assertEquals(Float.floatToIntBits(VKern.dotQ8(a, c, 0, n)), Float.floatToIntBits(Kern.dotQ8(a, c, 0, n)), "Kern.dotQ8");
    }
}
