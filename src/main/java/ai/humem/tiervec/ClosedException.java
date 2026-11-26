package ai.humem.tiervec;

/**
 * A call reached an index that is closed. After {@link TieredIndex#close()} the calls that read or change vectors or files throw it:
 * {@code add}, {@code delete}, {@code apply}, {@code search}, {@code rebuildFrom}, {@code sealNow}, {@code mergeNow}, {@code checkpoint}
 * and {@code reconfigure}. A search that was already running when the index was closed finishes with a correct result or throws it. The
 * reports ({@code size}, {@code stats}, {@code metrics}, {@code health}, {@code config}) keep working.
 */
public final class ClosedException extends TiervecException {
    private static final long serialVersionUID = 1L;

    /**
     * Creates the exception.
     *
     * @param message what was called
     */
    public ClosedException(String message) {
        super(message);
    }
}
