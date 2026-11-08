package com.humemai.vec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Routing of the int8 dot, the SQ8 dots and block16 on a JVM with the Vector API (issue #52). Where the preferred float species has fewer than 8 lanes (NEON on aarch64)
 * {@link VKernels} has no fast form of these four (its dotI8 ran about 140 times slower than the scalar twin there), so {@link Kernels} uses the 128-bit forms of
 * {@link NKernels} on aarch64 and the scalar twins on any other such host; with 8 or more lanes it uses {@link VKernels} as before. The routing tests assert the decision
 * itself (which method a call reaches, by bit equality), never a time, so they hold on any host; a test that needs a particular host is skipped on the others. The
 * agreement tests hold in every mode: integer data exactly, random data within the tolerance of {@link KernDiffTest}.
 */
class JvmQuantRoutingTest {
    static final int[] DIMS = {1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 128, 770};
    static final int AO = 3, BO = 5;

    static boolean arm() {
        String arch = System.getProperty("os.arch", "");
        return arch.equals("aarch64") || arch.equals("arm64");
    }

    @BeforeAll
    static void setUp() {
        assertTrue(Kernels.VECTOR && Kernels.JVM_VECTOR, "this test runs on a JVM with the Vector API (is --add-modules jdk.incubator.vector set?)");
    }

    @Test
    void theDecisionFollowsTheLanesAndTheArchitecture() {
        int fl = VKernels.FL;
        int flags = (Kernels.JVM_QUANT_VK ? 1 : 0) + (Kernels.JVM_NARROW ? 1 : 0) + (Kernels.JVM_QUANT_SCALAR ? 1 : 0);
        assertEquals(1, flags, "exactly one of JVM_QUANT_VK, JVM_NARROW and JVM_QUANT_SCALAR, FL " + fl);
        if (fl >= 8) {
            assertTrue(Kernels.JVM_QUANT_VK, "8 or more float lanes keep VKernels");
            assertNull(Kernels.quantReason, "no reason when VKernels is used");
        } else if (arm()) {
            // the narrow forms need NKernels to have initialized; if it could not, the scalar twins are used and the reason says why
            if (NKernels.S != null && !NKernels.WIDE) assertTrue(Kernels.JVM_NARROW, "aarch64 with fewer than 8 lanes uses the 128-bit forms of NKernels, reason " + Kernels.quantReason);
            else assertTrue(Kernels.JVM_QUANT_SCALAR, "NKernels is unusable, so the scalar twins are used");
            assertNotNull(Kernels.quantReason, "a reason is given");
        } else {
            assertTrue(Kernels.JVM_QUANT_SCALAR, "fewer than 8 lanes off aarch64 uses the scalar twins");
            assertNotNull(Kernels.quantReason, "a reason is given");
        }
        assertEquals(Kernels.QUANT == Kernels.Q_NARROW, Kernels.JVM_NARROW, "QUANT and JVM_NARROW");
    }

    @Test
    void anAarch64JvmWithNeonUsesTheNarrowForms() {
        Assumptions.assumeTrue(arm() && VKernels.FL < 8, "needs an aarch64 JVM with a 128-bit preferred species");
        assertTrue(NKernels.S != null && !NKernels.WIDE, "NKernels initializes on an aarch64 JVM and picks the narrow forms: " + NKernels.ERROR);
        assertTrue(Kernels.JVM_NARROW && !Kernels.JVM_QUANT_VK && !Kernels.JVM_QUANT_SCALAR, "narrow routing, reason " + Kernels.quantReason);
        Random r = new Random(41);
        for (int n : DIMS) {
            float[] q = KernDiffTest.rnd(r, n);
            byte[] c = KernDiffTest.codes(r, n + AO), d = KernDiffTest.codes(r, n + BO), blk = KernDiffTest.codes(r, n * 16 + 5);
            String m = " n=" + n;
            assertEquals(NKernels.dotI8N(c, AO, d, BO, n), Kernels.dotI8(c, AO, d, BO, n), "dotI8" + m);
            assertEquals(KernDiffTest.bits(NKernels.dotF8N(q, 0, c, AO, n)), KernDiffTest.bits(Kernels.dotF8(q, 0, c, AO, n)), "dotF8" + m);
            assertEquals(KernDiffTest.bits(NKernels.dotF8x4N(q, 0, c, AO, n)), KernDiffTest.bits(Kernels.dotF8x4(q, 0, c, AO, n)), "dotF8x4" + m);
            assertEquals(KernDiffTest.bits(NKernels.dotF8N(q, 0, c, AO, n)), KernDiffTest.bits(Kern.dotQ8(q, c, AO, n)), "Kern.dotQ8" + m);
            float[] o1 = new float[16], o2 = new float[16];
            NKernels.block16N(q, 0, blk, 5, n, o1);
            Kernels.block16(q, 0, blk, 5, n, o2);
            assertEquals(Arrays.toString(o1), Arrays.toString(o2), "block16" + m);
        }
    }

    @Test
    void aJvmWithWideLanesStillUsesVKernels() {
        Assumptions.assumeTrue(VKernels.FL >= 8, "needs a preferred species of 256 bits or more");
        assertTrue(Kernels.JVM_QUANT_VK && !Kernels.JVM_NARROW && !Kernels.JVM_QUANT_SCALAR, "VKernels routing");
        Random r = new Random(42);
        for (int n : DIMS) {
            float[] q = KernDiffTest.rnd(r, n);
            byte[] c = KernDiffTest.codes(r, n + AO), d = KernDiffTest.codes(r, n + BO), blk = KernDiffTest.codes(r, n * 16 + 5);
            String m = " n=" + n;
            assertEquals(VKernels.dotI8(c, AO, d, BO, n), Kernels.dotI8(c, AO, d, BO, n), "dotI8" + m);
            assertEquals(KernDiffTest.bits(VKernels.dotF8(q, 0, c, AO, n)), KernDiffTest.bits(Kernels.dotF8(q, 0, c, AO, n)), "dotF8" + m);
            assertEquals(KernDiffTest.bits(VKernels.dotF8x4(q, 0, c, AO, n)), KernDiffTest.bits(Kernels.dotF8x4(q, 0, c, AO, n)), "dotF8x4" + m);
            assertEquals(KernDiffTest.bits(VKern.dotQ8(q, c, AO, n)), KernDiffTest.bits(Kern.dotQ8(q, c, AO, n)), "Kern.dotQ8" + m);
            if (VKernels.BLOCK_OK) {
                float[] o1 = new float[16], o2 = new float[16];
                VKernels.block16(q, 0, blk, 5, n, o1);
                Kernels.block16(q, 0, blk, 5, n, o2);
                assertEquals(Arrays.toString(o1), Arrays.toString(o2), "block16" + m);
            }
        }
    }

    @Test
    void theFp32KernelsStayOnVKernelsOnEveryJvm() {
        Random r = new Random(43);
        for (int n : DIMS) {
            float[] a = KernDiffTest.rnd(r, n), b = KernDiffTest.rnd(r, n);
            assertEquals(KernDiffTest.bits(VKernels.dot(a, 0, b, 0, n)), KernDiffTest.bits(Kernels.dot(a, 0, b, 0, n)), "dot n=" + n);
            assertEquals(KernDiffTest.bits(VKernels.l2sq(a, 0, b, 0, n)), KernDiffTest.bits(Kernels.l2sq(a, 0, b, 0, n)), "l2sq n=" + n);
            assertEquals(KernDiffTest.bits(VKern.dot(a, 0, b, 0, n)), KernDiffTest.bits(Kern.dot(a, 0, b, 0, n)), "Kern.dot n=" + n);
            assertEquals(KernDiffTest.bits(VKern.l2(a, 0, b, 0, n)), KernDiffTest.bits(Kern.l2(a, 0, b, 0, n)), "Kern.l2 n=" + n);
        }
    }

    @Test
    void routedIntegerKernelsAgreeWithTheScalarTwinsExactly() {
        Random r = new Random(44);
        for (int n : DIMS) {
            // values outside the window are extreme, so a read past the window or before its offset changes the sum
            byte[] a = new byte[AO + n + 2], b = new byte[BO + n + 2];
            Arrays.fill(a, (byte) 127);
            Arrays.fill(b, (byte) 127);
            float[] q = new float[n];
            float[] qw = new float[AO + n + 2];
            Arrays.fill(qw, 1000f);
            for (int i = 0; i < n; i++) {
                a[AO + i] = (byte) (r.nextInt(255) - 127);
                b[BO + i] = (byte) (r.nextInt(255) - 127);
                q[i] = r.nextInt(17) - 8;
                qw[AO + i] = q[i];
            }
            String m = " n=" + n;
            assertEquals(Kernels.dotI8Scalar(a, AO, b, BO, n), Kernels.dotI8(a, AO, b, BO, n), "dotI8" + m);
            // integer products and sums are exact in float here, so every order of summation gives the same bits
            float s = Kernels.dotF8Scalar(q, 0, a, AO, n);
            assertEquals(s, Kernels.dotF8(q, 0, a, AO, n), 0f, "dotF8" + m);
            assertEquals(s, Kernels.dotF8x4(q, 0, a, AO, n), 0f, "dotF8x4" + m);
            assertEquals(s, Kern.dotQ8(q, a, AO, n), 0f, "Kern.dotQ8" + m);
            // the extremes: every product is -127 * 127, which fits a short
            for (int i = 0; i < n; i++) {
                a[AO + i] = (byte) 127;
                b[BO + i] = (byte) -127;
            }
            assertEquals(-127 * 127 * n, Kernels.dotI8(a, AO, b, BO, n), "dotI8 extreme" + m);
            byte[] bb = new byte[5 + n * 16 + 3];
            Arrays.fill(bb, (byte) 127);
            for (int i = 0; i < n * 16; i++) bb[5 + i] = (byte) (r.nextInt(256) - 128);
            float[] ref = new float[16], out = new float[16];
            Arrays.fill(out, 12345f); // every lane must be written
            Kernels.block16Scalar(q, 0, bb, 5, n, ref);
            Kernels.block16(q, 0, bb, 5, n, out);
            assertEquals(Arrays.toString(ref), Arrays.toString(out), "block16" + m);
        }
    }

    @Test
    void routedFloatKernelsAgreeWithTheScalarTwinsWithinTolerance() {
        Random r = new Random(45);
        for (int n : DIMS) {
            for (int rep = 0; rep < 20; rep++) {
                float[] q = KernDiffTest.rnd(r, n);
                byte[] c = KernDiffTest.codes(r, n);
                double dq = 0, dqAbs = 0;
                for (int i = 0; i < n; i++) {
                    dq += (double) q[i] * c[i];
                    dqAbs += Math.abs((double) q[i] * c[i]);
                }
                float sq = Kernels.dotF8Scalar(q, 0, c, 0, n);
                String m = " n=" + n;
                for (float v : new float[] {Kernels.dotF8(q, 0, c, 0, n), Kernels.dotF8x4(q, 0, c, 0, n), Kern.dotQ8(q, c, 0, n)}) {
                    KernDiffTest.withinN(v, sq, dqAbs, "SQ8 dot against scalar" + m);
                    KernDiffTest.withinN(v, dq, dqAbs, "SQ8 dot against exact" + m);
                }
                byte[] blk = KernDiffTest.codes(r, n * 16);
                float[] ref = new float[16], out = new float[16];
                Kernels.block16Scalar(q, 0, blk, 0, n, ref);
                Kernels.block16(q, 0, blk, 0, n, out);
                for (int lane = 0; lane < 16; lane++) {
                    double ex = 0, exAbs = 0;
                    for (int j = 0; j < n; j++) {
                        ex += (double) q[j] * blk[j * 16 + lane];
                        exAbs += Math.abs((double) q[j] * blk[j * 16 + lane]);
                    }
                    KernDiffTest.withinN(out[lane], ref[lane], exAbs, "block16 lane " + lane + " against scalar" + m);
                    KernDiffTest.withinN(out[lane], ex, exAbs, "block16 lane " + lane + " against exact" + m);
                }
            }
        }
    }
}
