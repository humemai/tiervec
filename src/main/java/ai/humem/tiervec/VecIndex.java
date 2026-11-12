package ai.humem.tiervec;

import java.util.function.LongPredicate;

/**
 * The interface that every index of this library implements: {@link FlatIndex}, {@link IvfIndex}, {@link GraphIndex}, {@link SealedGraph}
 * and {@link TieredIndex}.
 * Thread-safety contract: add, delete and search may be called concurrently from many threads; a search never throws because of a concurrent
 * add or delete; after add(id, v) returned, a search started later by ANY thread can return id (read-your-writes); after delete(id) returned,
 * a search started later never returns id. Ids are unique: adding an existing id replaces its vector.
 */
public interface VecIndex extends AutoCloseable {
    int dim();
    Metric metric();
    void add(long id, float[] v);
    /** @return true when the id existed */
    boolean delete(long id);
    /** Top k by score, best first. allow == null means no filter. May return fewer than k hits only when fewer than k live vectors qualify. */
    Hit[] search(float[] q, int k, LongPredicate allow);
    /** number of live vectors */
    int size();
    /** approximate bytes held by this index (heap arrays plus off-heap), for the memory report */
    long bytesHeld();
    @Override default void close() {}
}
