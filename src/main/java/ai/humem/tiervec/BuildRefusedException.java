package ai.humem.tiervec;

/**
 * The host refused a build that a call had to run on its own thread, so the call did nothing. A build that the library runs in the
 * background does not throw it: it is deferred and asked for again. This version never refuses a build, because it takes no build permit
 * from the host yet (<a href="https://github.com/humemai/tiervec/issues/13">issue 13</a>); the class is here so that a host can name the
 * whole exception family.
 */
public final class BuildRefusedException extends TiervecException {
    private static final long serialVersionUID = 1L;

    /**
     * Creates the exception.
     *
     * @param message what was refused
     */
    public BuildRefusedException(String message) {
        super(message);
    }
}
