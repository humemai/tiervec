package com.humemai.vec;

/** Code kind kept next to the fp32 vectors of a {@link FlatIndex}. The fp32 copy is always kept for the exact rerank. */
public enum Quant {
    /** no codes: every search is an exact fp32 scan */
    NONE,
    /** symmetric int8 with one scale per vector (see FlatIndex); scan the codes, rerank the best candidates in fp32 */
    SQ8,
    /** one sign bit per dimension (against a per-dimension threshold), Hamming scan, fp32 rerank */
    BINARY
}
