package ai.humem.tiervec;

/**
 * A search stopped because its {@link SearchContext} said so. The search has no result. A context that asked for it
 * ({@link SearchContext#partialWanted()}) finds the hits of the sources that were read to the end in {@link #partialHits()}. The search changed nothing in
 * the index, and the same thread can search again at once.
 */
public final class SearchCancelledException extends TiervecException {
    private static final long serialVersionUID = 1L;

    private transient Hit[] partial; // set by the search that throws this, when its context asked for the partial result

    /**
     * Creates the exception.
     *
     * @param message what stopped
     */
    public SearchCancelledException(String message) {
        super(message);
    }

    /**
     * The partial result of the stopped search, see {@link SearchContext#partialWanted()}. It is not complete: a vector of a source that was not read to
     * the end can be better than every hit here.
     *
     * @return a fresh array of at most {@code k} hits, best first; empty when the context did not ask, and when no source had been read to the end
     */
    public Hit[] partialHits() {
        return partial == null ? new Hit[0] : partial.clone();
    }

    void keep(Hit[] hits) {
        partial = hits;
    }
}
