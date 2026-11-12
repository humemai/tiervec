package ai.humem.tiervec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * The graph kernels of a JVM ({@link VKern}) return the same bits for the same input in every JIT tier. A segment that is written and read back
 * is then served with the same score bits as the in-memory graph it came from ({@link SegmentForgedFileTest} compares them with
 * {@code assertArrayEquals}, run 43165203453 failed on aarch64 with a score one ulp apart).
 *
 * The test states the order of the lane sum as scalar code, {@link VKernels#hsum}: lane j gets lane j + h for h = L/2 down to 1, and lane 0 is the sum.
 * {@code reduceLanes(ADD)} leaves the order to the implementation: the interpreter adds the lanes one after the other, and C1 and C2 add them in the
 * order of the instruction. The model is compared with the kernel before and after a warm-up that takes the kernel through every tier, and for lengths that
 * run the main loop, the single-vector loop and the scalar tail. This class runs in a JVM of its own (one JVM per test class, see the surefire
 * configuration) and the test that runs first has not warmed the kernels up, so its first calls run in the interpreter.
 */
class VKernFixedOrderTest {
    private static final int[] DIMS = {4, 5, 8, 9, 16, 17, 33, 100, 128, 770};

    @BeforeAll
    static void setUp() {
        assertTrue(Kernels.VECTOR, "the Vector API kernels must be the ones under test (is --add-modules jdk.incubator.vector set?)");
    }

    /** order 0: the lanes one after the other (the Java fallback of reduceLanes); order 1: {@link VKernels#hsum}, written as scalar code */
    static float laneSum(float[] lanes, int order) {
        if (order == 0) {
            float s = lanes[0];
            for (int j = 1; j < lanes.length; j++) s += lanes[j];
            return s;
        }
        float[] w = lanes.clone();
        for (int h = w.length / 2; h >= 1; h >>= 1) for (int j = 0; j < h; j++) w[j] = w[j] + w[j + h];
        return w[0];
    }

    /** VKern.dot (l2 false) or VKern.l2 (l2 true) as scalar code: two accumulators of L lanes with fma, one more single-vector loop, the lane sum, the scalar tail */
    static float model(float[] a, float[] b, int n, boolean l2, int order) {
        int lanes = VKern.L;
        float[] s0 = new float[lanes], s1 = new float[lanes];
        int i = 0;
        for (; i + 2 * lanes <= n; i += 2 * lanes) {
            for (int j = 0; j < lanes; j++) {
                float d0 = l2 ? a[i + j] - b[i + j] : a[i + j], x0 = l2 ? d0 : b[i + j];
                float d1 = l2 ? a[i + lanes + j] - b[i + lanes + j] : a[i + lanes + j], x1 = l2 ? d1 : b[i + lanes + j];
                s0[j] = Math.fma(d0, x0, s0[j]);
                s1[j] = Math.fma(d1, x1, s1[j]);
            }
        }
        for (; i + lanes <= n; i += lanes) {
            for (int j = 0; j < lanes; j++) {
                float d0 = l2 ? a[i + j] - b[i + j] : a[i + j], x0 = l2 ? d0 : b[i + j];
                s0[j] = Math.fma(d0, x0, s0[j]);
            }
        }
        float[] sum = new float[lanes];
        for (int j = 0; j < lanes; j++) sum[j] = s0[j] + s1[j];
        float s = laneSum(sum, order);
        for (; i < n; i++) {
            float d = l2 ? a[i] - b[i] : a[i], x = l2 ? d : b[i];
            s += d * x;
        }
        return s;
    }

    /** VKern.dotQ8 as scalar code (only when Q8_OK) */
    static float modelQ8(float[] q, byte[] c, int n, int order) {
        int lanes = VKern.L;
        float[] s0 = new float[lanes];
        int i = 0;
        for (; i + lanes <= n; i += lanes) for (int j = 0; j < lanes; j++) s0[j] = Math.fma(q[i + j], (float) c[i + j], s0[j]);
        float s = laneSum(s0, order);
        for (; i < n; i++) s += q[i] * c[i];
        return s;
    }

    private static int bits(float x) {
        return Float.floatToRawIntBits(x);
    }

    private static final class Data {
        final float[] a, b, q;
        final byte[] c;

        Data(int n, long seed) {
            Random r = new Random(seed);
            a = new float[n];
            b = new float[n];
            q = new float[n];
            c = new byte[n];
            for (int i = 0; i < n; i++) {
                a[i] = r.nextFloat() - 0.5f;
                b[i] = r.nextFloat() - 0.5f;
                q[i] = r.nextFloat() - 0.5f;
                c[i] = (byte) (r.nextInt(255) - 127);
            }
        }
    }

    /** the kernels against the model of the fixed order; returns the number of cases where the other order, the lanes one after the other, gives other bits */
    private static int checkAll(Data[] data, String when) {
        int distinguishes = 0;
        for (Data d : data) {
            int n = d.a.length;
            assertEquals(bits(model(d.a, d.b, n, false, 1)), bits(VKern.dot(d.a, 0, d.b, 0, n)), when + ": dot n=" + n);
            assertEquals(bits(model(d.a, d.b, n, true, 1)), bits(VKern.l2(d.a, 0, d.b, 0, n)), when + ": l2 n=" + n);
            if (VKern.Q8_OK) assertEquals(bits(modelQ8(d.q, d.c, n, 1)), bits(VKern.dotQ8(d.q, d.c, 0, n)), when + ": dotQ8 n=" + n);
            if (bits(model(d.a, d.b, n, false, 0)) != bits(model(d.a, d.b, n, false, 1))) distinguishes++;
            if (bits(model(d.a, d.b, n, true, 0)) != bits(model(d.a, d.b, n, true, 1))) distinguishes++;
        }
        return distinguishes;
    }

    @Test
    void theLaneSumHasTheSameOrderBeforeAndAfterTheKernelsAreCompiled() {
        Data[] data = new Data[DIMS.length * 4];
        for (int k = 0; k < data.length; k++) data[k] = new Data(DIMS[k % DIMS.length], 500 + k);
        int cold = checkAll(data, "cold");
        // the warm-up takes dot, l2 and dotQ8 through the interpreter, C1 and C2 (OSR included), each on every length
        float sink = 0;
        for (int rep = 0; rep < 400_000; rep++) {
            Data d = data[rep % data.length];
            int n = d.a.length;
            sink += VKern.dot(d.a, 0, d.b, 0, n) + VKern.l2(d.a, 0, d.b, 0, n);
            if (VKern.Q8_OK) sink += VKern.dotQ8(d.q, d.c, 0, n);
        }
        assertTrue(Float.isFinite(sink), "the warm-up result is used");
        checkAll(data, "warm");
        // a control for the model: the data must tell the fixed order from the other one, or the comparisons above would hold for any order
        assertTrue(cold > 10, "the data separates the two orders in " + cold + " of " + 2 * data.length + " cases (lanes " + VKern.L + ")");
    }
}
