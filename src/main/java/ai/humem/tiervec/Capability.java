package ai.humem.tiervec;

/**
 * What this library can do, as the set that {@link TieredIndex#capabilities()} returns. A host checks the set when an index definition is
 * created, so that it refuses a feature that does not exist at that point and not at query time. A later minor version of the contract may
 * add constants; a host ignores the ones it does not know.
 *
 * <p>What the tier does not offer, in any version that does not say so here: grouped search (a top {@code k} per group), scores that come
 * from compressed codes without an exact fp32 rescoring pass (returned scores are exact fp32 scores), a retract of an applied write other
 * than the opposite write, transactions or uncommitted rows, a persisted tail ({@code checkpoint} needs an empty one), memory-mapped
 * segment readers, and more than one process on one index directory.
 */
public enum Capability {
    /** {@link TieredIndex#apply} makes the adds and deletes of one batch visible to searches as a unit */
    ATOMIC_BATCH,
    /** the host applies a batch when the changes of its transaction stand; the library has no retract and needs none then */
    DEFERRED_APPLY,
    /** a search stops at a deadline or a cancel flag that the host supplies ({@link SearchContext}) */
    SEARCH_CONTEXT,
    /** {@link TieredIndex#checkpoint} and {@link TieredIndex#open}: segments and their hidden rows, not the tail */
    PERSISTENCE,
    /** a rebuild from a snapshot of the records while writes go on ({@link TieredIndex#rebuildFrom(java.util.function.Supplier, long)}) */
    SNAPSHOT_REBUILD,
    /** the host gives the threads, the build permits and the memory budget ({@link HostEnv}) */
    HOST_ENVIRONMENT,
    /** a filter that says how many ids it allows ({@link SizedIdFilter}) is scanned exactly, without a graph walk, when it allows few */
    SELECTIVE_PLAN
}
