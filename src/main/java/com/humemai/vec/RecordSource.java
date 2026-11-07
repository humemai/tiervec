package com.humemai.vec;

/**
 * What a host offers to the library so that the library can reconcile its persisted state with the host's records after a restart
 * (see {@code docs/ID-EPOCH-AND-REPLAY.md}, steps 2 to 7, and {@code TieredIndex.open(Path, int, Metric, TieredIndex.Config, RecordSource)}).
 * The host's records are the truth. Every method is called on the thread that calls {@code open}, after the host finished its own recovery.
 */
public interface RecordSource {

    /** The epoch of the host's current numbering of ids. The library compares epochs for equality only. {@link Manifest#NONE} is not a valid epoch. */
    long epoch();

    /** The highest id the host itself found after its own recovery, or {@link Manifest#NONE} when it holds no record. Held ids above it are dropped. */
    long recoveredMax();

    /** Whether the host holds a live record with this id. Called once for every id that a persisted segment holds. */
    boolean isLive(long id);

    /** Every live record with an id above {@code id}, in increasing id order. Called with the high-water id of the persisted state. */
    Iterable<TieredIndex.IdVector> liveAbove(long id);

    /** Every live record, for the cold build when the persisted state cannot be used. */
    Iterable<TieredIndex.IdVector> allLive();
}
