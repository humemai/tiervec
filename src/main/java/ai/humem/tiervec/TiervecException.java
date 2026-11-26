package ai.humem.tiervec;

/**
 * The base of the exceptions that tiervec throws for the state of an index: {@link CorruptStateException}, {@link SearchCancelledException},
 * {@link ClosedException} and {@link BuildRefusedException}. All are unchecked. A programming error (a wrong dimension, a bad configuration,
 * a call in the wrong state) stays an {@link IllegalArgumentException} or an {@link IllegalStateException}, and a failure of the file system
 * stays an {@link java.io.IOException}. Only the library defines subclasses.
 */
public abstract class TiervecException extends RuntimeException {
    private static final long serialVersionUID = 1L;

    TiervecException(String message) {
        super(message);
    }

    TiervecException(String message, Throwable cause) {
        super(message, cause);
    }
}
