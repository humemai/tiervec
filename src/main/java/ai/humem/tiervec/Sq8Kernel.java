package ai.humem.tiervec;

/**
 * How an SQ8 scan scores a vector. All variants store the same per-vector symmetric int8 codes (scale = max|x| / 127), so recall is
 * the same up to the query quantization of I8. BLOCK also changes the code layout (dimension-major blocks of 16 vectors).
 */
public enum Sq8Kernel {
    /** query quantized to int8 too; int8 x int8 through shorts and ints (the first version) */
    I8,
    /** float query against the codes widened to float, one accumulator */
    F8,
    /** as F8 with four independent accumulators */
    F8X4,
    /** codes stored in dimension-major blocks of 16 vectors; one query value multiplies 16 vectors, no horizontal sum */
    BLOCK;

    /** used when the constructor does not name one (set from the micro report) */
    public static final Sq8Kernel DEFAULT = F8X4;
}
