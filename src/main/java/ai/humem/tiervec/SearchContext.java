package ai.humem.tiervec;

/**
 * What the caller of a search says about how long it may run. A host passes one to
 * {@link TieredIndex#search(float[], int, java.util.function.LongPredicate, SearchContext)} (or to one of the other overloads that take one) to stop a search at
 * a statement timeout, after a client disconnect, or for any other reason that the host knows and the library does not.
 *
 * <p>The cooperation is by polling. The library calls {@link #expired()} on the thread that runs the search: once when the search starts, before it reads each
 * tail and each segment, and at least once every 1024 candidates inside a tail scan or a segment search. A candidate is a vector that a scan or a graph
 * walk looks at, whether or not the allow predicate accepts it. When the method returns {@code true}, the search throws
 * {@link SearchCancelledException} and returns no result.
 *
 * <p>A caller that prefers some answer over none at a deadline returns {@code true} from {@link #partialWanted()}. The search still throws, and the
 * exception carries the hits of the sources that were read to the end ({@link SearchCancelledException#partialHits()}):
 *
 * <pre>{@code
 * SearchContext context = new SearchContext() {
 *     public boolean expired()       { return System.nanoTime() - deadline >= 0; }
 *     public boolean partialWanted() { return true; }
 * };
 * try {
 *     use(index.search(query, 10, null, context));   // the complete answer
 * } catch (SearchCancelledException e) {
 *     useIncomplete(e.partialHits());                // at most 10 hits, best first
 * }
 * }</pre>
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

    /**
     * Asked once, when a search stops because {@link #expired()} returned true: whether the caller wants the hits that the search can still answer
     * for. A source is a tail or a segment. The partial result holds the best hits, at most {@code k}, of the sources that were read to the end,
     * with their exact scores; the source in progress is left out, so a search that stops inside the only large segment often has hits of the tail
     * alone, or none. Like a complete result it answers from the state of the index when the search started: no id deleted before then, and no
     * vector replaced before then.
     *
     * @return false (the default): the exception carries no hits
     */
    default boolean partialWanted() {
        return false;
    }
}
