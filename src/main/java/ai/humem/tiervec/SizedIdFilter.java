package ai.humem.tiervec;

import java.util.function.LongPredicate;

/**
 * An allowed-id filter that also says how many ids it allows. A host that has the set of the ids at hand (the result of a predicate in its
 * query language) passes one wherever a search takes a {@link LongPredicate}.
 *
 * <p>{@link TieredIndex}, {@link GraphIndex} and {@link SealedGraph} read the count once or a few times per search. When it is at most the beam
 * width of the search ({@code ef}, never below {@code k}), they skip the graph walk and scan the allowed ids exactly, which costs one call of
 * {@link #test} per live id and returns the exact top {@code k}. With a larger or an unknown count the search is the one of a plain predicate.
 * The threshold has not been set from measurements.
 *
 * <p>The count is a hint and picks the plan only. {@link #test} alone decides which ids may be returned, so a wrong count changes the cost of a
 * search and never lets an id through that the filter rejects. A count of 0 scans too: a filter that allows ids in spite of it still gets them.
 */
public interface SizedIdFilter extends LongPredicate {

    /** the value of {@link #allowedCount} when the number is not known */
    long UNKNOWN = -1;

    /** @return how many ids this filter allows, or a negative number ({@link #UNKNOWN}) when that is not known */
    long allowedCount();
}
