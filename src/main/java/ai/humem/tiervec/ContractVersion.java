package ai.humem.tiervec;

/**
 * The version of the contract between a host and this library (docs/PROVIDER-CONTRACT.md), which is not the version of the jar. A host
 * writes down the version it was built against and asks {@link #accepts} about the library it runs with:
 *
 * <pre>{@code
 * if (!new ContractVersion(1, 0).accepts(TieredIndex.contractVersion())) throw ...;
 * }</pre>
 *
 * <p>The major part changes when the contract changes in a way that an older host cannot follow. The minor part changes when something is
 * added; what a library can do is in {@link TieredIndex#capabilities()}.
 *
 * @param major the major part
 * @param minor the minor part
 */
public record ContractVersion(int major, int minor) {
    /** rejects a negative part */
    public ContractVersion {
        if (major < 0 || minor < 0) throw new IllegalArgumentException("negative contract version " + major + "." + minor);
    }

    /**
     * Whether a host that was built against this version can use a library of {@code library}: a larger major is refused, a larger minor is
     * accepted.
     *
     * @param library the version of the library, from {@link TieredIndex#contractVersion()}
     * @return false when the library's major part is above this one's
     */
    public boolean accepts(ContractVersion library) {
        return library.major <= major;
    }

    @Override
    public String toString() {
        return major + "." + minor;
    }
}
