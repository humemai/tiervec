package ai.humem.tiervec;

/**
 * The polls of one search: it counts the candidates that the scans and graph walks look at, and asks the {@link SearchContext} again before more than
 * {@link #INTERVAL} of them have been looked at since the last poll. A search with a context has one, used by its own thread only. A search without a context
 * has none, and the loops test for null.
 */
final class Poll {
    /** the most candidates between two polls. The Javadoc of {@link SearchContext} and docs/PROVIDER-CONTRACT.md state this number. */
    static final int INTERVAL = 256;

    private final SearchContext context;
    private int left;

    Poll(SearchContext context) {
        this.context = context;
    }

    /** asks the context now and starts a new interval */
    void now() {
        if (context.expired()) throw new SearchCancelledException("the search context expired");
        left = INTERVAL;
    }

    /** called before the next {@code n} candidates are looked at; {@code n} is at most {@link #INTERVAL} */
    void before(int n) {
        if (n > left) now();
        left -= n;
    }
}
