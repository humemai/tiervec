package com.humemai.vec;

/**
 * The similarity of an index. A {@link Hit} score is higher for a better match under every metric.
 */
public enum Metric {
    /** the cosine of the angle between two vectors, the score as is */
    COSINE,
    /** the dot product, the score as is */
    DOT,
    /** the Euclidean distance, the score is the negative squared distance */
    L2
}
