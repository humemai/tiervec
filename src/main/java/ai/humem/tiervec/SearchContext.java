package ai.humem.tiervec;

/**
 * What the caller of a search says about how long it may run. A host passes one to
 * {@link TieredIndex#search(float[], int, java.util.function.LongPredicate, SearchContext)} (or to one of the other overloads that take one) to stop a search at
 * a statement timeout, after a client disconnect, or for any other reason that the host knows and the library does not.
 *
 * <p>The cooperation is by polling. The library calls {@link #expired()} on the thread that runs the search: once when the search starts, before it reads each
 * tail and each segment, at least once every 256 candidates inside a tail scan or a segment search, and after each pause while it waits for a batch that is
 * being applied. A candidate is a vector that a scan or a graph walk looks at, whether or not the allow predicate accepts it. When the method returns
 * {@code true}, the search throws {@link SearchCancelledException} and returns no result.
 *
 * <p>The method must be cheap, must not block, and must not call into the index. An exception that it throws reaches the caller of the search unchanged. The
 * library does not look at the interrupt status of the thread; a context can.
 *
 * <p>A deadline and a cancel flag in one context:
 *
 * <pre>{@code
 * long deadline = System.nanoTime() + 50_000_000L;
 * Hit[] hits = index.search(query, 10, null, () -> cancelled.get() || System.nanoTime() - deadline >= 0);
 * }</pre>
 */
@FunctionalInterface
public interface SearchContext {
    /**
     * Asked by the search at the points that the class comment lists.
     *
     * @return true when the search has to stop
     */
    boolean expired();
}
