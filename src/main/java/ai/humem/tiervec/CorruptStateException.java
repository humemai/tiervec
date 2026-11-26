package ai.humem.tiervec;

/**
 * Persisted state is damaged, so it cannot be served: the host throws the derived state away and builds the index from its records with
 * {@link TieredIndex#rebuildFrom}. {@link TieredIndex#open(java.nio.file.Path, TieredIndex.Config)} throws it. The cause is the
 * {@link SegmentFormat.CorruptException} that names the damage. A directory of a newer format ({@link SegmentFormat.UnsupportedException}),
 * files of another index definition ({@link SegmentFormat.MismatchException}) and a missing manifest are not damage and stay checked
 * exceptions.
 */
public final class CorruptStateException extends TiervecException {
    private static final long serialVersionUID = 1L;

    /**
     * Creates the exception.
     *
     * @param message what is damaged
     * @param cause what found it, or null
     */
    public CorruptStateException(String message, Throwable cause) {
        super(message, cause);
    }
}
