package ai.humem.tiervec;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

/**
 * The Vector API version of the graph kernels. Only {@link Kern} calls this class, and only when {@link Kernels#VECTOR} is true, so a
 * scalar-only run (native image, no jdk.incubator.vector module, -Dai.humem.tiervec.scalar=true) never loads it. No Vector object appears in a
 * field type or a method signature of this class: the signatures use plain arrays, so a caller does not link against the incubator module.
 * Plain float arrays with offsets; two accumulators; scalar tail.
 * dotQ8: dot of a float query with SIGNED int8 codes (the codes are converted to float lanes).
 */
final class VKern {
    private VKern() {}

    static final VectorSpecies<Float> F = FloatVector.SPECIES_PREFERRED;
    static final int L = F.length();
    /** a bytes species with the same lane count as F needs at least 64 bits, i.e. L >= 8; below that dotQ8 is not available here and Kern uses the scalar twin */
    static final boolean Q8_OK = L >= 8;
    static final VectorSpecies<Byte> BS = Q8_OK ? VectorSpecies.of(byte.class, VectorShape.forBitSize(L * 8)) : ByteVector.SPECIES_64;

    static float dot(float[] a, int ao, float[] b, int bo, int n) {
        FloatVector s0 = FloatVector.zero(F), s1 = FloatVector.zero(F);
        int i = 0;
        for (; i + 2 * L <= n; i += 2 * L) {
            s0 = FloatVector.fromArray(F, a, ao + i).fma(FloatVector.fromArray(F, b, bo + i), s0);
            s1 = FloatVector.fromArray(F, a, ao + i + L).fma(FloatVector.fromArray(F, b, bo + i + L), s1);
        }
        for (; i + L <= n; i += L) s0 = FloatVector.fromArray(F, a, ao + i).fma(FloatVector.fromArray(F, b, bo + i), s0);
        float s = s0.add(s1).reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) s += a[ao + i] * b[bo + i];
        return s;
    }

    /** squared L2 distance */
    static float l2(float[] a, int ao, float[] b, int bo, int n) {
        FloatVector s0 = FloatVector.zero(F), s1 = FloatVector.zero(F);
        int i = 0;
        for (; i + 2 * L <= n; i += 2 * L) {
            FloatVector d0 = FloatVector.fromArray(F, a, ao + i).sub(FloatVector.fromArray(F, b, bo + i));
            FloatVector d1 = FloatVector.fromArray(F, a, ao + i + L).sub(FloatVector.fromArray(F, b, bo + i + L));
            s0 = d0.fma(d0, s0);
            s1 = d1.fma(d1, s1);
        }
        for (; i + L <= n; i += L) {
            FloatVector d0 = FloatVector.fromArray(F, a, ao + i).sub(FloatVector.fromArray(F, b, bo + i));
            s0 = d0.fma(d0, s0);
        }
        float s = s0.add(s1).reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) { float d = a[ao + i] - b[bo + i]; s += d * d; }
        return s;
    }

    /** sum over i of q[i] * codes[co + i] (codes are signed bytes); only when Q8_OK */
    static float dotQ8(float[] q, byte[] codes, int co, int n) {
        FloatVector s0 = FloatVector.zero(F);
        int i = 0;
        for (; i + L <= n; i += L) {
            FloatVector c = (FloatVector) ByteVector.fromArray(BS, codes, co + i).convertShape(VectorOperators.B2F, F, 0);
            s0 = FloatVector.fromArray(F, q, i).fma(c, s0);
        }
        float s = s0.reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) s += q[i] * codes[co + i];
        return s;
    }
}
