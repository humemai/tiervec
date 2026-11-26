package ai.humem.tiervec;

/**
 * A search stopped because its {@link SearchContext} said so. The search has no result, not a partial one. It changed nothing in the index, and the same thread
 * can search again at once.
 */
public final class SearchCancelledException extends TiervecException {
    private static final long serialVersionUID = 1L;

    /**
     * Creates the exception.
     *
     * @param message what stopped
     */
    public SearchCancelledException(String message) {
        super(message);
    }
}
