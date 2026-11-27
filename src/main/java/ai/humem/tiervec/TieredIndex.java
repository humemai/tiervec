package ai.humem.tiervec;

import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicLongArray;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.LockSupport;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Consumer;
import java.util.function.IntPredicate;
import java.util.function.LongConsumer;
import java.util.function.LongPredicate;
import java.util.function.LongSupplier;
import java.util.function.Supplier;
import java.util.stream.LongStream;

/**
 * Tiered vector index for agent memory: a bounded exact TAIL in front of immutable graph SEGMENTS (an LSM-style layout: writes go to
 * the tail, a seal turns the tail into a segment, merges keep the number of segments small).
 *
 * <p>Source of truth. The vectors in the records are the truth; the segments (graphs) and the marks on their hidden rows are DERIVED
 * data. After a crash the engine would re-read its records and call {@link #rebuildFrom(Iterable)}: everything (tail, segments, marks)
 * is reconstructed from raw vectors and nothing else. In the same way a segment or the whole tier can always be thrown away and rebuilt
 * from the tail plus the records, so the code never has to make the derived state durable. Porting to LSM_VECTOR: the tail is the
 * engine's delta list (its exact scan stays), a segment is one sealed graph file written beside the records, the hidden rows follow
 * from the record versions, and the seal/merge worker is the engine's background rebuild with a bounded queue.
 *
 * <p>Writes. {@link #add} puts the vector into the active tail (a {@link FlatIndex}, exact fp32 or SQ8) and a raw copy next to it
 * (the raw copy is what a seal and a rebuild read). It is visible to the next search by any thread before add returns.
 * {@link #delete} removes it from the tails and hides every segment copy at once.
 *
 * <p>Search = exact scan of the tail(s) merged with the approximate search of every segment, minus hidden ids, best score first,
 * the allow-list applied in both. A search reads one version of the tier when it starts and answers from that version only: it holds every
 * write that had returned by then and none that returns later, however long the search runs, and it never waits for a write. Duplicate ids
 * are resolved newest source first: active tail, frozen tail, then segments. {@link Freshness#SEALED_ONLY} skips the tails: it may lag
 * (a vector added or replaced since the last seal is not in it, a replaced id is absent until its new version is sealed), and it never
 * returns an id deleted before it started, a vector replaced before then, or an id that was never added. Below {@code smallLimit} live
 * vectors no graph is ever built and every search is an exact scan, which removes the rebuild on the query path of the engine today.
 *
 * <p>Deadline and cancellation. The search overloads that take a {@link SearchContext} poll it at the start, before each tail and each segment, at a bounded
 * interval inside them, and throw {@link SearchCancelledException} when it says stop.
 *
 * <p>Seal. When the active tail reaches {@code tailLimit} vectors (or its first vector is older than {@code tailAgeMs}) and the
 * index holds at least {@code smallLimit} vectors, the worker thread (1) freezes the tail under the write lock (a new empty tail
 * takes over writes, the view now names a frozen tail and an active tail), (2) builds a GraphIndex from the frozen contents and
 * seals it, outside any lock, while searches and writes continue (searches scan both tails), (3) publishes the segment by one
 * volatile write of an immutable view that drops the frozen tail. A vector is always in the frozen tail, the segment, or both for an
 * instant, and the dedupe resolves the overlap, so it is never missing. A search that started before the swap finishes on the old
 * view, which still holds everything it needs.
 *
 * <p>Delete and replace. Versions come from one counter under the write lock. A replace or a delete marks every visible copy of the id
 * in the segments as hidden, with the version at which it was hidden; a mark is set once and never removed. A copy in the frozen tail
 * that a seal is building into a segment gets its mark when the seal publishes the segment. Searches pass over marked rows inside the
 * beam, so a deleted vector never appears, with no wait for a merge. Each segment counts its marked rows ({@code dead}); a segment with
 * none is searched with the plain allow-list.
 *
 * <p>Merge. When the segment count exceeds {@code maxSegments}, or one segment is {@code deadFraction} dead, the worker rebuilds the
 * smallest segments (or the dead one) into ONE segment from their valid vectors only, throttled to {@code mergeRatePerSec} inserts
 * per second so the cores that serve queries are not starved, copies the marks that were set while it ran onto the new segment, and
 * swaps it in atomically. A search that still runs on an older view keeps the segments that view names, with their marks.
 *
 * <p>Size-ratio trigger ({@code mergeTriggerRatio} X, default 0 = off: merges are then picked by the count rule alone). The segment with the most live rows is the BIG one. When the live rows of all the other
 * segments together exceed X times the live rows of the big one, ALL segments are merged into one (the small ones go into the big one). Otherwise, when the count exceeds {@code maxSegments} and
 * maxSegments is at least 2, the count rule picks among the SMALL segments only (the big one is never a candidate), so the small segments merge among themselves and the big one is rewritten only by the
 * ratio rule; with maxSegments 1 the count rule includes the big one (a merge at every seal). With a large maxSegments the small segments pile up until the ratio fires. A merge into the big segment
 * rewrites (1 + X) * rows(big) rows per X * rows(big) inserts, about (1 + X) / X vectors per insert, on top of the small-segment merges.
 *
 * <p>Bounds. One worker thread, one task at a time (seal or merge, a lock named work), a cap on segments, no allocation proportional
 * to n on the query path (per query: the hit lists and a hash table of size about twice the hits). Back pressure: when the active tail
 * holds {@code hardTailFactor * max(tailLimit, smallLimit)} vectors, writers wait (bounded, 10 s) for the worker.
 *
 * <p>Versions. Every added row and every delete takes the next number of one counter, under the write lock. After a write, or after all
 * the writes of a batch, the writer publishes a new view with one volatile write: the segment list, the tails with the slots that they had
 * then, and the version. Rows added later lie beyond those slot counts, and a row hidden later keeps a mark above the version, so what a
 * view shows never changes. No version is tracked or released: a view lives while a search holds it, and the garbage collector frees it
 * and what only it names (a frozen tail after its seal, the inputs of a merge).
 *
 * <p>Batches. {@link #apply} applies the adds and then the deletes of one host transaction under one hold of the write lock and publishes
 * them as one version, so a search sees all of the batch or none of it, and a batch never delays a search. {@link #awaitCapacity} is the
 * separate back pressure pre-check: {@code apply} never waits for the worker.
 *
 * <p>Thread safety: add and delete take the write lock {@code mu} (short: one flat add, a few binary searches), and apply takes it once
 * for the whole batch; searches take no lock, wait for nothing, and write nothing that other searches read (each does one volatile read of
 * the view). {@link #rebuildFrom} and {@link #sealNow} and {@link #mergeNow} take the work lock.
 *
 * <p>Persistence (first cut). A tier can be written to a directory with {@link #checkpoint} and read back with {@link #open}: segment files,
 * a tombstone file (the rows that a delete or a replace has hidden), a tail file (the rows that no segment holds yet) and a manifest per
 * generation, see docs/SEGMENT-FORMAT.md. Only a checkpoint writes: a write between two checkpoints is in memory only. The records stay the
 * truth; when {@code open} cannot read the directory, the host rebuilds from its records as above.
 *
 * <p>Epoch and replay (docs/ID-EPOCH-AND-REPLAY.md). The host names its numbering of ids with an epoch. A rebuild with an epoch
 * ({@link #rebuildFrom(Supplier, long)}) keeps the writes that arrive during the build and applies them after the swap, skipping adds at or below the
 * snapshot position. A checkpoint stores the epoch and the largest id that the segments or the tail hold (the high-water id) in the manifest. {@link #open(Path, int,
 * Metric, Config, RecordSource)} discards persisted state of another epoch, drops held ids above the id the host recovered (durably), hides ids that
 * the host no longer holds, and replays the host's live records above the high-water id.
 *
 * <p>Host contract (docs/PROVIDER-CONTRACT.md). {@link Config#toMap}, {@link Config#fromMap} and {@link Config#fingerprint} give the text form
 * of the settings, {@link #reconfigure} changes the search-time knobs of an open index, {@link #contractVersion} and {@link #capabilities} say
 * what exists, {@link #metrics}, {@link #bindMetrics} and {@link #health} report, and the exceptions for the state of an index extend
 * {@link TiervecException}. After {@link #close} the calls that read or change vectors or files throw {@link ClosedException}.
 */
public final class TieredIndex implements VecIndex {

    /**
     * How the background maintenance is doing. OK: the last seal and the last merge, if any, worked. DEGRADED: a seal or a merge has failed
     * 3 times in a row; it is retried, {@link TieredIndex#lastError} holds the failure, and writes and searches go on (the tail grows while seals
     * fail). FAILED: the maintenance cannot run any more, so nothing is sealed or merged until the index is rebuilt in a new object.
     */
    public enum Health { OK, DEGRADED, FAILED }

    /**
     * The fingerprint of an index definition, see {@link Config#fingerprint}.
     * @param text the canonical text, {@code key=value} parts joined by {@code ;}, that names every data-affecting setting
     * @param sha256 the SHA-256 of that text in UTF-8, 64 hex digits
     */
    public record Fingerprint(String text, String sha256) {}

    /** FRESH (default): tails included, read-your-writes. SEALED_ONLY: segments only, may lag. */
    public enum Freshness { FRESH, SEALED_ONLY }

    /**
     * How a merge that takes the largest segment builds its result. REBUILD (default): a new graph over the valid vectors of all inputs.
     * INSERT: a copy of the graph of the largest segment, without its hidden rows, into which the valid vectors of the other inputs are
     * inserted; the cost of the graph work then follows what changed and not the size of the largest segment. A merge rebuilds in both modes
     * when it does not take the largest segment, when it takes one segment only, or when the hidden share of the largest segment is above
     * {@code deadFraction}. The graph of repeated INSERT merges is not the graph that a rebuild makes, and its cost has not been measured.
     */
    public enum MergeMode { REBUILD, INSERT }

    /**
     * Where a segment that {@code open} reads from a directory keeps its fp32 vectors. HEAP (default): copied into a heap array. MAPPED: left
     * in the segment file, which is mapped read-only; each distance copies its row into a scratch array of the searching thread first. The
     * graph, the ids and the SQ8 codes are heap arrays with both, and the checks of {@code open} are the same, the checksum of every section
     * included. The hits and scores are the same.
     *
     * <p>MAPPED saves heap and costs time: distances on mapped rows were measured slower than on the heap, and so were searches that walk fp32
     * vectors in most runs (docs/BENCHMARKS.md, "The cost of a row copy"). It covers only what {@code open} reads: a segment that a seal, a merge, or a rebuild builds keeps its vectors on
     * the heap until a later {@code open} reads it from a file. {@code bytesHeld} leaves the mapped vectors out; the operating system's page
     * cache holds the pages that are in use. A mapping lasts until the garbage collector releases it, also after {@code close}, and the
     * library never unmaps: until then Windows can refuse to delete a retired segment file (which then stays for a later checkpoint or
     * open), and a deleted file can keep its disk space. Nothing may change the files while they are mapped: damage after the open is not detected, and a
     * file that is cut short or cannot be read can end a search with an {@code InternalError} or end the JVM. Details:
     * docs/SEGMENT-FORMAT.md, "Mapped vectors".
     */
    public enum VectorStorage { HEAP, MAPPED }

    /**
     * Result of a search with its meta (what the facets report as exact/patched/lag).
     * @param exact true when the answer equals brute force: the tail scan was exact, and no segment was searched or every segment was
     *              scanned because of the count of a {@link SizedIdFilter}
     * @param tailScanned live tail vectors scanned (0 for SEALED_ONLY)
     * @param segments number of segments searched
     */
    public record SearchResult(Hit[] hits, boolean exact, int tailScanned, int segments) {}

    /** one raw vector with its id, the unit {@link #rebuildFrom} reads (the engine would stream its records) */
    public record IdVector(long id, float[] v) {}

    /** the records a rebuild reads and the highest id they cover (see {@link #rebuildFrom(Supplier, long)}) */
    public record Snapshot(long position, Iterable<IdVector> records) {}

    /**
     * The changes of one committed host transaction: adds first (in list order, a later add of an id replaces an earlier one), then deletes.
     * An update is an add of a new id plus a delete of the old id. The lists are copied at construction.
     */
    public record Batch(List<IdVector> adds, List<Long> deletes) {
        public Batch {
            adds = List.copyOf(adds);
            deletes = List.copyOf(deletes);
        }
    }

    /**
     * Parameters. tailLimit: seal when the active tail has this many vectors (default 2000). tailAgeMs: also seal when the oldest tail
     * vector is older than this (0 = never by age). smallLimit: below this many live vectors never build a graph (default 1000).
     * maxSegments: merge when exceeded (default 8). mergeFan: smallest segments merged at once (default 4). deadFraction: compact a
     * segment when this share of it is hidden (default 0.3). mergeRatePerSec: merge builder insert rate cap, 0 = unlimited.
     * hardTailFactor: back pressure threshold. tailQuant: NONE (exact fp32) or SQ8. graph: segment graph parameters (efSearch is the
     * search beam). buildThreads: threads of one seal or merge build. vectorStorage (default HEAP): where a segment that {@code open} reads
     * from a directory keeps its fp32 vectors, see {@link VectorStorage}. splitBeam (default false = the
     * full beam in every segment): when true and more than one segment is searched, segment s gets the beam max(k, ceil(ef * rows(s) / rows of
     * all searched segments)) instead of ef. tailQuant is the tail kernel: NONE scans fp32, SQ8 scans int8 codes (FlatIndex default kernel
     * F8X4) and reranks the best 5 * k candidates in fp32. mergeTriggerRatio (default 0 = off): the size-ratio merge trigger. When it is above 0 and
     * the live rows of all segments except the largest exceed mergeTriggerRatio times the live rows of the largest, the worker merges ALL segments into one (the small ones go into the big one); otherwise
     * the count rule (maxSegments, mergeFan) applies to the small segments only (the largest is not a candidate) when maxSegments is at least 2, and to all segments when it is 1 (merge at every seal).
     * mergeMode (default REBUILD): how a merge that takes the largest segment builds its result, see {@link MergeMode}.
     */
    public record Config(int tailLimit, long tailAgeMs, int smallLimit, int maxSegments, int mergeFan, double deadFraction,
                         int mergeRatePerSec, int hardTailFactor, Quant tailQuant, GraphIndex.Params graph, int buildThreads, boolean splitBeam, double mergeTriggerRatio,
                         MergeMode mergeMode, VectorStorage vectorStorage) {
        /** the 14-component form without vectorStorage: it is HEAP */
        public Config(int tailLimit, long tailAgeMs, int smallLimit, int maxSegments, int mergeFan, double deadFraction,
                      int mergeRatePerSec, int hardTailFactor, Quant tailQuant, GraphIndex.Params graph, int buildThreads, boolean splitBeam, double mergeTriggerRatio,
                      MergeMode mergeMode) {
            this(tailLimit, tailAgeMs, smallLimit, maxSegments, mergeFan, deadFraction, mergeRatePerSec, hardTailFactor, tailQuant, graph, buildThreads, splitBeam, mergeTriggerRatio, mergeMode, VectorStorage.HEAP);
        }

        /** the 13-component form without mergeMode: it is REBUILD */
        public Config(int tailLimit, long tailAgeMs, int smallLimit, int maxSegments, int mergeFan, double deadFraction,
                      int mergeRatePerSec, int hardTailFactor, Quant tailQuant, GraphIndex.Params graph, int buildThreads, boolean splitBeam, double mergeTriggerRatio) {
            this(tailLimit, tailAgeMs, smallLimit, maxSegments, mergeFan, deadFraction, mergeRatePerSec, hardTailFactor, tailQuant, graph, buildThreads, splitBeam, mergeTriggerRatio, MergeMode.REBUILD);
        }

        /** the 12-component form without mergeTriggerRatio: it is 0 (off) */
        public Config(int tailLimit, long tailAgeMs, int smallLimit, int maxSegments, int mergeFan, double deadFraction,
                      int mergeRatePerSec, int hardTailFactor, Quant tailQuant, GraphIndex.Params graph, int buildThreads, boolean splitBeam) {
            this(tailLimit, tailAgeMs, smallLimit, maxSegments, mergeFan, deadFraction, mergeRatePerSec, hardTailFactor, tailQuant, graph, buildThreads, splitBeam, 0.0);
        }

        /** the 11-component form without splitBeam and mergeTriggerRatio: splitBeam is false and mergeTriggerRatio is 0 (off) */
        public Config(int tailLimit, long tailAgeMs, int smallLimit, int maxSegments, int mergeFan, double deadFraction,
                      int mergeRatePerSec, int hardTailFactor, Quant tailQuant, GraphIndex.Params graph, int buildThreads) {
            this(tailLimit, tailAgeMs, smallLimit, maxSegments, mergeFan, deadFraction, mergeRatePerSec, hardTailFactor, tailQuant, graph, buildThreads, false, 0.0);
        }

        public Config {
            Objects.requireNonNull(mergeMode, "mergeMode");
            Objects.requireNonNull(vectorStorage, "vectorStorage");
            if (tailLimit < 1 || smallLimit < 0 || maxSegments < 1 || mergeFan < 2 || hardTailFactor < 2 || buildThreads < 1 || !(mergeTriggerRatio >= 0.0) || Double.isInfinite(mergeTriggerRatio))
                throw new IllegalArgumentException("bad config");
            if (tailQuant == Quant.BINARY) throw new IllegalArgumentException("tail quant must be NONE or SQ8");
        }

        public static Config defaults() {
            return new Config(2000, 60_000L, 1000, 8, 4, 0.3, 50_000, 4, Quant.NONE, GraphIndex.Params.defaults(),
                    Math.max(1, Math.min(4, Runtime.getRuntime().availableProcessors() - 1)), false, 0.0);
        }

        public Config withTail(int tailLimit, int smallLimit) {
            return new Config(tailLimit, tailAgeMs, smallLimit, maxSegments, mergeFan, deadFraction, mergeRatePerSec, hardTailFactor, tailQuant, graph, buildThreads, splitBeam, mergeTriggerRatio, mergeMode, vectorStorage);
        }

        public Config withMaxSegments(int m) {
            return new Config(tailLimit, tailAgeMs, smallLimit, m, mergeFan, deadFraction, mergeRatePerSec, hardTailFactor, tailQuant, graph, buildThreads, splitBeam, mergeTriggerRatio, mergeMode, vectorStorage);
        }

        public Config withTailQuant(Quant q) {
            return new Config(tailLimit, tailAgeMs, smallLimit, maxSegments, mergeFan, deadFraction, mergeRatePerSec, hardTailFactor, q, graph, buildThreads, splitBeam, mergeTriggerRatio, mergeMode, vectorStorage);
        }

        public Config withGraph(GraphIndex.Params p) {
            return new Config(tailLimit, tailAgeMs, smallLimit, maxSegments, mergeFan, deadFraction, mergeRatePerSec, hardTailFactor, tailQuant, p, buildThreads, splitBeam, mergeTriggerRatio, mergeMode, vectorStorage);
        }

        public Config withBuildThreads(int t) {
            return new Config(tailLimit, tailAgeMs, smallLimit, maxSegments, mergeFan, deadFraction, mergeRatePerSec, hardTailFactor, tailQuant, graph, t, splitBeam, mergeTriggerRatio, mergeMode, vectorStorage);
        }

        public Config withMergeRate(int perSec) {
            return new Config(tailLimit, tailAgeMs, smallLimit, maxSegments, mergeFan, deadFraction, perSec, hardTailFactor, tailQuant, graph, buildThreads, splitBeam, mergeTriggerRatio, mergeMode, vectorStorage);
        }

        public Config withMergeFan(int f) {
            return new Config(tailLimit, tailAgeMs, smallLimit, maxSegments, f, deadFraction, mergeRatePerSec, hardTailFactor, tailQuant, graph, buildThreads, splitBeam, mergeTriggerRatio, mergeMode, vectorStorage);
        }

        public Config withSplitBeam(boolean sb) {
            return new Config(tailLimit, tailAgeMs, smallLimit, maxSegments, mergeFan, deadFraction, mergeRatePerSec, hardTailFactor, tailQuant, graph, buildThreads, sb, mergeTriggerRatio, mergeMode, vectorStorage);
        }

        public Config withMergeTriggerRatio(double r) {
            return new Config(tailLimit, tailAgeMs, smallLimit, maxSegments, mergeFan, deadFraction, mergeRatePerSec, hardTailFactor, tailQuant, graph, buildThreads, splitBeam, r, mergeMode, vectorStorage);
        }

        /** this configuration with another way to build a merge that takes the largest segment (see {@link MergeMode}) */
        public Config withMergeMode(MergeMode mode) {
            return new Config(tailLimit, tailAgeMs, smallLimit, maxSegments, mergeFan, deadFraction, mergeRatePerSec, hardTailFactor, tailQuant, graph, buildThreads, splitBeam, mergeTriggerRatio, mode, vectorStorage);
        }

        /** this configuration with another place for the vectors of the segments that {@code open} reads (see {@link VectorStorage}) */
        public Config withVectorStorage(VectorStorage storage) {
            return new Config(tailLimit, tailAgeMs, smallLimit, maxSegments, mergeFan, deadFraction, mergeRatePerSec, hardTailFactor, tailQuant, graph, buildThreads, splitBeam, mergeTriggerRatio, mergeMode, storage);
        }

        /**
         * This configuration as text, for a host that stores its index definitions as text: one entry per component and per graph parameter
         * ({@code graph.M} and so on), all of them present, in a fixed order. The keys are listed in docs/DESIGN.md, "Tier configuration".
         * {@link #fromMap} of the result equals this configuration. The dimension and the metric are not part of a configuration.
         *
         * @return an unmodifiable map
         */
        public Map<String, String> toMap() {
            return ConfigMap.write(this);
        }

        /**
         * The configuration of a map, for the moment an index is created: a key that the map does not have takes the default of this version
         * of the library, and an unknown key or a value that cannot be used is refused.
         *
         * @param map keys and values as {@link #toMap} writes them; an empty map gives the defaults
         * @return the configuration
         * @throws IllegalArgumentException naming the key, when a key is unknown or a value cannot be used
         */
        public static Config fromMap(Map<String, String> map) {
            return ConfigMap.read(map, null);
        }

        /**
         * {@link #fromMap} for the moment a stored definition is loaded: an unknown key, which a newer version of the library may have written, is
         * reported and ignored, so that a downgrade can open the index. A value that cannot be used is refused as {@code fromMap} refuses it.
         *
         * @param map the stored keys and values
         * @param warning receives one message per ignored key, which names the key
         * @return the configuration
         * @throws IllegalArgumentException naming the key, when a value of a known key cannot be used
         */
        public static Config fromStoredMap(Map<String, String> map, Consumer<String> warning) {
            return ConfigMap.read(map, Objects.requireNonNull(warning, "warning"));
        }

        /**
         * The fingerprint of an index with this configuration. It covers what decides the data of the graphs and their files: the dimension, the
         * metric, the id width, {@code graph.M}, {@code graph.efConstruction}, {@code graph.alpha}, {@code graph.sq8} and the layer 0 layout. It
         * does not cover the search-time knobs, the tail settings (a persisted tail holds the vectors as the host passed them, so they change
         * nothing in a directory), the seal and merge policy, or {@code graph.seed}.
         * A host stores it with its index definition and compares it before it opens the persisted state: {@link TieredIndex#open} itself checks
         * the dimension, the metric, {@code M}, {@code efConstruction} and {@code sq8}, and the files do not record {@code alpha}.
         *
         * @param dim the dimension of the index
         * @param metric the metric of the index
         * @return the fingerprint
         */
        public Fingerprint fingerprint(int dim, Metric metric) {
            return ConfigMap.fingerprint(this, dim, metric);
        }
    }

    /** counters for the report. floorEntries counts the rows of the segments that a delete or a replace has hidden and that no merge has
     *  left out yet. mergedVectors counts the rows that merges put into a graph: with {@link MergeMode#INSERT} the inserted rows only, not those of a copied graph.
     *  segmentSizes: rows of each segment, oldest first, hidden copies included; its length is segments. adds (calls of add), mergeWriteAmp (mergedVectors / adds, 0 when adds is 0),
     *  totalWriteAmp ((sealedVectors + mergedVectors) / adds), maxSealMillis and maxMergeMillis (the longest single seal or merge).
     *  The task log, one slot per finished seal or merge in finish order: taskKinds (1 seal, 2 merge), taskStartNs and taskEndNs (System.nanoTime() values; end is the instant after the swap),
     *  taskRows (rows built); taskDropped counts tasks that did not fit the 4096 slots.
     *  bigMerges and bigMergedVectors count the merges that rewrote the segment with the most live rows at pick time (a forced mergeNow included) and the rows they built; both are parts of merges and mergedVectors */
    public record Stats(long seals, long merges, int tailSize, int segments, long bytesHeld, long sealMillis, long mergeMillis,
                        int floorEntries, long failures, int live, long backpressureWaits, long sealedVectors, long mergedVectors,
                        int[] segmentSizes, long adds, double mergeWriteAmp, double totalWriteAmp, long maxSealMillis, long maxMergeMillis,
                        int[] taskKinds, long[] taskStartNs, long[] taskEndNs, int[] taskRows, long taskDropped, long bigMerges, long bigMergedVectors) {}

    /** one accepted write, kept for the rebuild that is running: the adds with the tier's own copies of their vectors, then the deletes */
    private record Change(long[] addIds, float[][] addVecs, long[] deletes) {}

    private static final Hit[] NONE = new Hit[0];
    private static final int FLAT_RERANK = 5; // the default of FlatIndex for SQ8
    private static final long RESTORED = -1; // the mark of a row that open reads as hidden: below every version, so hidden at all of them

    // ---- state ----

    private static final class Rec {
        final float[] v;
        final long ver;

        Rec(float[] v, long ver) {
            this.v = v;
            this.ver = ver;
        }
    }

    /** an id whose copies below the version {@code ver} are hidden */
    private record Mark(long id, long ver) {}

    /**
     * a bounded tail: the flat index scanned by searches and the raw copies a seal reads. Mutated only under mu, and every change carries its
     * version, so that the flat index can be searched as of an earlier one ({@link FlatIndex#searchAt}).
     */
    private static final class Tail {
        final FlatIndex flat;
        final ConcurrentHashMap<Long, Rec> raw = new ConcurrentHashMap<>();
        volatile long firstNs;
        /** the rows that were hidden while this tail was frozen, which a seal marks on the segment it built from the tail; guarded by mu */
        final List<Mark> late = new ArrayList<>();

        Tail(int dim, Metric metric, Quant q) {
            flat = new FlatIndex(dim, metric, q);
        }

        void put(long id, Rec r) {
            if (raw.isEmpty()) firstNs = System.nanoTime();
            flat.add(id, r.v, r.ver);
            raw.put(id, r);
        }

        void remove(long id, long ver) {
            raw.remove(id);
            flat.delete(id, ver);
        }

        /** removes a row of a frozen tail, and remembers it for the seal that is building a segment from this tail */
        void hide(long id, long ver) {
            remove(id, ver);
            late.add(new Mark(id, ver));
        }
    }

    private static final class Segment {
        private static final VarHandle HIDDEN_AT = MethodHandles.arrayElementVarHandle(long[].class);

        final SealedGraph g;
        final int n;
        final long[] ids;  // sorted
        final long[] vers; // aligned with ids
        volatile int dead; // rows that are hidden; written under mu, after the mark of the row
        /** hiddenAt[row]: the version at which the row was hidden, 0 while it is visible; null until the first row is hidden. A mark is set once. */
        private volatile long[] hiddenAt;
        private int[] order; // the hidden rows in the order in which they were hidden, the first dead of them; guarded by mu
        private volatile long firstHidden = Long.MAX_VALUE, lastHidden = Long.MIN_VALUE; // the smallest and the largest mark; written under mu
        volatile long fileId = -1; // id of the segment file this segment was written to or loaded from (seg-<id>.tvs), -1 when it has none; written under the work lock

        Segment(SealedGraph g, long[] idsIn, long[] versIn) {
            this.g = g;
            this.n = idsIn.length;
            Integer[] ord = new Integer[n];
            for (int i = 0; i < n; i++) ord[i] = i;
            Arrays.sort(ord, (a, b) -> Long.compare(idsIn[a], idsIn[b]));
            ids = new long[n];
            vers = new long[n];
            for (int i = 0; i < n; i++) {
                ids[i] = idsIn[ord[i]];
                vers[i] = versIn[ord[i]];
            }
        }

        /** a segment read from a file: the id order comes with the graph (no sort) and versByRow is aligned with the graph's rows */
        Segment(SealedGraph g, long[] versByRow) {
            this.g = g;
            this.n = g.n;
            this.ids = g.sortedIds;
            this.vers = new long[n];
            for (int i = 0; i < n; i++) vers[i] = versByRow[g.sortedRows[i]];
        }

        /** the version of each row of the graph, in row order (the order a segment file stores them in) */
        long[] versByRow() {
            long[] r = new long[n];
            for (int row = 0; row < n; row++) r[row] = vers[Arrays.binarySearch(ids, g.ids[row])];
            return r;
        }

        int indexOf(long id) {
            int i = Arrays.binarySearch(ids, id);
            return i >= 0 ? i : -1;
        }

        /** the row of an id in the graph, or -1 */
        int rowOf(long id) {
            int i = indexOf(id);
            return i >= 0 ? g.sortedRows[i] : -1;
        }

        boolean isHidden(int row) {
            long[] h = hiddenAt;
            return h != null && (long) HIDDEN_AT.getOpaque(h, row) != 0L;
        }

        /**
         * The rows that a search as of version {@code at} may return, or null when it may return every row. A row is visible at a version
         * when it has no mark, or a mark above that version. The caller holds a view of that version, so it sees every mark up to it; a
         * later mark, which it may or may not see, changes nothing.
         */
        IntPredicate rowsAt(long at) {
            if (dead == 0 || firstHidden > at) return null;
            long[] h = hiddenAt;
            return row -> {
                long mark = (long) HIDDEN_AT.getOpaque(h, row);
                return mark == 0L || mark > at;
            };
        }

        /** whether every row was hidden at version {@code at} */
        boolean goneAt(long at) {
            return dead >= n && lastHidden <= at;
        }

        /** under mu, or before the segment is shared: hides a row at a version, which is not 0; a row that is hidden already keeps its mark */
        void hide(int row, long ver) {
            long[] h = hiddenAt;
            if (h == null) hiddenAt = h = new long[n];
            if (h[row] != 0L) return;
            HIDDEN_AT.setOpaque(h, row, ver);
            if (ver < firstHidden) firstHidden = ver;
            if (ver > lastHidden) lastHidden = ver;
            int d = dead;
            if (order == null) order = new int[Math.min(n, 8)];
            else if (d == order.length) order = Arrays.copyOf(order, (int) Math.min(n, 2L * d));
            order[d] = row;
            dead = d + 1;
        }

        /** under mu: hides the copy of an id that is older than {@code ver}, when this segment holds one */
        void hideOlder(long id, long ver) {
            int i = indexOf(id);
            if (i >= 0 && vers[i] < ver) hide(g.sortedRows[i], ver);
        }

        /** under mu: the row that was hidden as number {@code i}, 0 &lt;= i &lt; dead */
        int hiddenRow(int i) {
            return order[i];
        }

        /** under mu: the version at which a hidden row was hidden */
        long hiddenVersion(int row) {
            return hiddenAt[row];
        }

        long bytes() {
            long b = g.bytesHeld() + 16L * n + 64;
            if (hiddenAt != null) b += 16 + 8L * n;
            int[] o = order; // read without the lock: a number for a report
            if (o != null) b += 16 + 4L * o.length;
            return b;
        }
    }

    /**
     * Immutable: one version of the tier. Every write publishes a new one with one volatile write, and a search reads one and answers from
     * it alone. The segments and the tails are shared between versions and go on changing; what they gain after {@code at} is not part of
     * this version: rows past the slot counts of the tails, and marks above {@code at}. Nothing that this version holds is taken away:
     * slots are never reused, marks are never removed, and a seal or a merge builds new segments and leaves the old ones as they are.
     */
    private static final class View {
        final Segment[] segs;
        final Tail frozen; // being sealed, or null
        final Tail active;
        final long at; // the version: every write up to it is in this view, and no later one
        final int activeSlots, activeRows, frozenSlots, frozenRows; // the slots written and the live rows of the tails at this version

        /** under mu, or before the tier is shared: these parts as they are now, which hold every write up to the version {@code at} */
        View(Segment[] segs, Tail frozen, Tail active, long at) {
            this.segs = segs;
            this.frozen = frozen;
            this.active = active;
            this.at = at;
            this.activeSlots = active.flat.slotsUsed();
            this.activeRows = active.raw.size();
            this.frozenSlots = frozen == null ? 0 : frozen.flat.slotsUsed();
            this.frozenRows = frozen == null ? 0 : frozen.raw.size();
        }
    }

    /** row order of a build: by arrival (version). Arrival order keeps rows that were written together next to each other in memory, which
     *  is what a graph built in one pass over the dataset has; a hash order costs cache misses on every query. */
    private static int[] orderByVersion(long[] vers, int n) {
        int[] ord = new int[n];
        if (n < (1 << 24)) {
            long[] key = new long[n];
            for (int i = 0; i < n; i++) key[i] = (vers[i] << 24) | i;
            Arrays.sort(key);
            for (int i = 0; i < n; i++) ord[i] = (int) (key[i] & 0xFFFFFF);
        } else {
            Integer[] o = new Integer[n];
            for (int i = 0; i < n; i++) o[i] = i;
            Arrays.sort(o, (a, b) -> Long.compare(vers[a], vers[b]));
            for (int i = 0; i < n; i++) ord[i] = o[i];
        }
        return ord;
    }

    private static long[] permute(long[] a, int[] ord) {
        long[] r = new long[ord.length];
        for (int i = 0; i < ord.length; i++) r[i] = a[ord[i]];
        return r;
    }

    private interface VecSource {
        float[] vec(int i);
    }

    private final int dim;
    private final Metric metric;
    private volatile Config cfg; // replaced only by reconfigure, under mu, and only in its search-time knobs
    private final HostEnv env;
    private final boolean hosted; // the host runs the tasks and grants the builds; there is no worker thread
    /** hosted only: a maintenance task is with the executor or running */
    private final AtomicBoolean queued = new AtomicBoolean();
    private volatile boolean rejected; // hosted only: the executor refused the last task
    /** per {@link HostEnv.BuildKind}: the System.nanoTime() before which a refused or failed background build is not tried again, or {@link #NOW} */
    private final AtomicLongArray retryAt = new AtomicLongArray(new long[] {NOW, NOW, NOW});
    private final long[] backoff = new long[3];
    private final AtomicLong deferrals = new AtomicLong();
    private volatile long ageAskedFor = NOW; // hosted only: Tail.firstNs of the tail for which the host's timer was last asked to look again
    private static final long NOW = Long.MIN_VALUE, FIRST_BACKOFF = 50_000_000L, MAX_BACKOFF = 5_000_000_000L;
    private static final int MIN_TAIL = 64; // a memory budget does not shrink the seal threshold below this
    private final ReentrantLock mu = new ReentrantLock();
    /**
     * One seal, merge, rebuild or checkpoint at a time. Fair: the worker takes it again right after each task while more is due, and
     * under steady writes a lock that lets a newcomer pass a waiting thread can keep a caller of {@code sealNow}, {@code mergeNow} or
     * {@code checkpoint} out for as long as the writes go on (issue #144).
     */
    private final ReentrantLock work = new ReentrantLock(true);
    private volatile View view;
    private volatile double filterScanBelow;
    /**
     * Filtered searches: a segment whose allowed share is estimated at or below this fraction is scanned exactly instead of walked first
     * (see FilterPlan and {@link GraphIndex#filterScanBelow(double)}). Default 0 = off, the walk with its fallback after the visit budget.
     * The tails are always scanned exactly. Takes effect for the next search.
     */
    public void filterScanBelow(double fraction) { FilterPlan.check(fraction); filterScanBelow = fraction; }
    public double filterScanBelow() { return filterScanBelow; }
    private long counter; // the versions: every row and every delete takes the next one; guarded by mu
    private final AtomicInteger live = new AtomicInteger();
    private final Semaphore wake = new Semaphore(0);
    private volatile boolean closed;
    private volatile int activity; // 0 idle, 1 sealing, 2 merging
    private volatile Throwable lastError;
    private Path persistedDir;           // the directory of the last checkpoint or open, guarded by mu
    private DirectoryLock dirLock;       // the claim on the directory that this tier uses, null before it uses one and after close; guarded by mu
    private Manifest persistedManifest;  // the generation that directory holds, guarded by mu
    private volatile Throwable fallbackCause; // why open fell back to a rebuild, null when the segments were read from files
    /** the host's id epoch: set by a rebuild with an epoch, by open with a record source, and read back from a manifest; {@link Manifest#NONE} when none was ever given */
    private volatile long epoch = Manifest.NONE;
    /** the changes that arrived while a snapshot rebuild runs, in arrival order, or null when none runs; guarded by mu (see {@link #rebuildFrom(Supplier, long)}) */
    private ArrayList<Change> capture;
    private volatile Runnable hookBeforePublish;
    private volatile Runnable hookAfterPublish;
    private volatile Runnable hookInBatch;
    private final AtomicLong seals = new AtomicLong(), merges = new AtomicLong(), sealNanos = new AtomicLong(), mergeNanos = new AtomicLong();
    private final AtomicLong failures = new AtomicLong(), backpressure = new AtomicLong(), sealedVectors = new AtomicLong(), mergedVectors = new AtomicLong();
    private final AtomicLong maxSealNanos = new AtomicLong(), maxMergeNanos = new AtomicLong();
    private final AtomicLong bigMerges = new AtomicLong(), bigMergedVectors = new AtomicLong(); // merges that rewrote the largest segment, and the rows they built
    private volatile long addsCount; // add() calls; incremented under mu
    private final Thread worker;
    /** background seals and merges that failed in a row, per kind; a success of that kind resets it (see {@link Health}) */
    private volatile int sealFailStreak, mergeFailStreak;
    private static final int DEGRADED_AFTER = 3;
    /** the counters of a bound {@link MetricSink}, in the order of {@link #COUNTERS}, or null */
    private volatile LongConsumer[] sinkCounters;
    private static final String[] COUNTERS = {"tiervec.seals", "tiervec.merges", "tiervec.failures", "tiervec.backpressureWaits",
            "tiervec.sealedVectors", "tiervec.mergedVectors", "tiervec.bigMerges", "tiervec.bigMergedVectors", "tiervec.deferredBuilds"};
    private static final int C_SEALS = 0, C_MERGES = 1, C_FAILURES = 2, C_BACKPRESSURE = 3, C_SEALED = 4, C_MERGED = 5, C_BIG = 6, C_BIG_MERGED = 7,
            C_DEFERRED = 8;
    private static final ContractVersion CONTRACT = new ContractVersion(1, 0);

    // task log: the seal and merge intervals as System.nanoTime() stamps, so a benchmark can compare them with its own timestamps. doSeal and doMerge always run
    // under the work lock (the worker loop, sealNow, mergeNow), so there is one writer; stats() reads the slots below taskCount, which is written after the slot (volatile).
    private static final int TASK_LOG_CAP = 4096;
    private final int[] taskKind = new int[TASK_LOG_CAP];
    private final int[] taskRows = new int[TASK_LOG_CAP];
    private final long[] taskStart = new long[TASK_LOG_CAP];
    private final long[] taskEnd = new long[TASK_LOG_CAP];
    private volatile int taskCount;
    private final AtomicLong taskDropped = new AtomicLong();
    /** seal builds are throttled to this many inserts per second (0 = unthrottled); a benchmark knob set by setSealRatePerSec, not a Config component */
    private volatile int sealRate;

    /** one finished task (kind 1 seal, 2 merge) into the log; the caller holds the work lock */
    private void logTask(int kind, long startNs, long endNs, int rows) {
        int i = taskCount;
        if (i >= TASK_LOG_CAP) {
            taskDropped.incrementAndGet();
            return;
        }
        taskKind[i] = kind;
        taskStart[i] = startNs;
        taskEnd[i] = endNs;
        taskRows[i] = rows;
        taskCount = i + 1;
    }

    public int sealRatePerSec() { return sealRate; }
    public void setSealRatePerSec(int perSec) { sealRate = Math.max(0, perSec); }

    public TieredIndex(int dim, Metric metric) {
        this(dim, metric, Config.defaults());
    }

    public TieredIndex(int dim, Metric metric, Config cfg) {
        this(dim, metric, cfg, HostEnv.standalone());
    }

    /**
     * A tier that runs on the threads, the build permits and the memory budget of a host, see {@link HostEnv}. With
     * {@link HostEnv#standalone()} it is the tier that {@link #TieredIndex(int, Metric, Config)} makes.
     */
    public TieredIndex(int dim, Metric metric, Config cfg, HostEnv env) {
        if (dim < 1) throw new IllegalArgumentException("dim");
        this.dim = dim;
        this.metric = metric;
        this.cfg = cfg;
        this.env = Objects.requireNonNull(env, "env");
        this.hosted = !(env instanceof StandaloneEnv);
        this.view = new View(new Segment[0], null, new Tail(dim, metric, cfg.tailQuant()), 0L);
        if (hosted) {
            this.worker = null;
        } else {
            this.worker = new Thread(this::workerLoop, "tiered-worker");
            this.worker.setDaemon(true);
            this.worker.start();
        }
        MetricSink sink = env.metrics();
        if (sink != null) bindMetrics(sink);
    }

    @Override public int dim() { return dim; }
    @Override public Metric metric() { return metric; }
    @Override public int size() { return live.get(); }
    public Config config() { return cfg; }

    /**
     * Changes search-time knobs of this index while it runs: {@code graph.efSearch}, the beam of a search that names none, and {@code splitBeam}.
     * They take effect for the searches that start after the call, and {@link #config} shows them. Every other key is refused: a data-affecting
     * key describes graphs that exist already, and the settings of the tail and of the maintenance are fixed for the life of the object.
     * Either all changes are made or none. {@link #filterScanBelow(double)} is the other knob that changes at run time.
     *
     * @param changes keys and values as {@link Config#toMap} writes them; a key whose value does not change is accepted
     * @throws IllegalArgumentException naming the key, when a key is unknown, cannot change or has a value that cannot be used
     * @throws ClosedException when the index is closed
     */
    public void reconfigure(Map<String, String> changes) {
        mu.lock();
        try {
            checkOpen();
            Map<String, String> now = cfg.toMap();
            Config next = cfg;
            for (Map.Entry<String, String> e : new TreeMap<>(changes).entrySet()) {
                String k = e.getKey();
                if (!now.containsKey(k)) throw new IllegalArgumentException("unknown configuration key '" + k + "'");
                if (!now.get(k).equals(e.getValue())) next = ConfigMap.reconfigured(next, k, e.getValue());
            }
            cfg = next; // one write, after every change was accepted
        } finally {
            mu.unlock();
        }
    }

    /** the version of the contract between a host and this library, see {@link ContractVersion} */
    public static ContractVersion contractVersion() {
        return CONTRACT;
    }

    /** what this library can do, an unmodifiable set; see {@link Capability} for what each means and for what the tier does not offer */
    public static Set<Capability> capabilities() {
        return Collections.unmodifiableSet(EnumSet.allOf(Capability.class));
    }

    /** how the background maintenance is doing, see {@link Health}; a closed index keeps the state it had apart from FAILED */
    public Health health() {
        if (!closed && (hosted ? rejected : !worker.isAlive())) return Health.FAILED;
        return sealFailStreak >= DEGRADED_AFTER || mergeFailStreak >= DEGRADED_AFTER ? Health.DEGRADED : Health.OK;
    }

    /**
     * The statistics as names and numbers, for a host that forwards them to its own metrics: counters since the object was made, and gauges.
     * The names start with {@code tiervec.} and are listed in docs/DESIGN.md, "Statistics". A later version may add names. A write
     * amplification is in thousandths, and {@code tiervec.health} is 0 for OK, 1 for DEGRADED and 2 for FAILED.
     *
     * @return an unmodifiable map in a fixed order
     */
    public Map<String, Long> metrics() {
        Map<String, Long> m = new LinkedHashMap<>();
        long[] counted = {seals.get(), merges.get(), failures.get(), backpressure.get(), sealedVectors.get(), mergedVectors.get(), bigMerges.get(),
                bigMergedVectors.get(), deferrals.get()};
        for (int i = 0; i < counted.length; i++) m.put(COUNTERS[i], counted[i]);
        for (Map.Entry<String, LongSupplier> g : gauges().entrySet()) m.put(g.getKey(), g.getValue().getAsLong());
        return Collections.unmodifiableMap(m);
    }

    /**
     * Pushes the statistics to a sink: it receives every gauge once, to read when it wants, and from now on each increment of the counters
     * (the names of {@link #metrics} that the sink is asked a counter for). Increments from before this call are not repeated. A second call
     * replaces the sink for the counters.
     *
     * @param sink the receiver
     */
    public void bindMetrics(MetricSink sink) {
        Objects.requireNonNull(sink, "sink");
        LongConsumer[] c = new LongConsumer[COUNTERS.length];
        for (int i = 0; i < c.length; i++) c[i] = Objects.requireNonNull(sink.counter(COUNTERS[i]), COUNTERS[i]);
        for (Map.Entry<String, LongSupplier> g : gauges().entrySet()) sink.gauge(g.getKey(), g.getValue());
        sinkCounters = c;
    }

    private Map<String, LongSupplier> gauges() {
        Map<String, LongSupplier> g = new LinkedHashMap<>();
        g.put("tiervec.adds", () -> addsCount);
        g.put("tiervec.live", live::get);
        g.put("tiervec.tailSize", this::pendingTail);
        g.put("tiervec.segments", () -> view.segs.length);
        g.put("tiervec.bytesHeld", this::bytesHeld);
        g.put("tiervec.floorEntries", () -> hiddenRows(view.segs));
        g.put("tiervec.sealMillis", () -> sealNanos.get() / 1_000_000L);
        g.put("tiervec.mergeMillis", () -> mergeNanos.get() / 1_000_000L);
        g.put("tiervec.maxSealMillis", () -> maxSealNanos.get() / 1_000_000L);
        g.put("tiervec.maxMergeMillis", () -> maxMergeNanos.get() / 1_000_000L);
        g.put("tiervec.mergeWriteAmpThousandths", () -> thousandths(mergedVectors.get()));
        g.put("tiervec.totalWriteAmpThousandths", () -> thousandths(sealedVectors.get() + mergedVectors.get()));
        g.put("tiervec.health", () -> health().ordinal());
        return g;
    }

    private long thousandths(long rows) {
        long a = addsCount;
        return a > 0 ? rows * 1000L / a : 0L;
    }

    /** adds to a counter and passes the increment to the bound sink; a sink that throws does not disturb the index */
    private void count(AtomicLong counter, int index, long n) {
        counter.addAndGet(n);
        LongConsumer[] c = sinkCounters;
        if (c != null && n > 0) {
            try {
                c[index].accept(n);
            } catch (RuntimeException hostMetrics) {
                // the number is still in metrics() and stats()
            }
        }
    }

    private void checkOpen() {
        if (closed) throw new ClosedException("the index is closed");
    }

    /** 0 idle, 1 a seal is running, 2 a merge is running (for per-phase latency reports) */
    public int activity() { return activity; }
    public Throwable lastError() { return lastError; }
    /** whether the worker thread of a standalone tier runs; false for a tier with a host environment, which has none */
    public boolean workerAlive() { return worker != null && worker.isAlive(); }
    /** test hook: ends the maintenance worker while the tier stays open, which {@link #health} reports as FAILED */
    void interruptWorker() { worker.interrupt(); }

    // ---- the host environment ----

    /**
     * For a tier with a host environment: submits a maintenance task to the executor when a seal or a merge is due (a seal by age included)
     * and no refused or failed build is still waiting for its retry. The tier calls it itself after every write; a host calls it from its
     * timer when {@link HostEnv#schedule} has none. A standalone tier ignores the call.
     */
    public void maintain() {
        if (!hosted || closed) return;
        askForAgeSeal();
        if (due()) submit();
    }

    /** asks the host's timer, once per tail, to look again when the active tail is older than {@code tailAgeMs} */
    private void askForAgeSeal() {
        long age = cfg.tailAgeMs();
        Tail t = view.active;
        if (age <= 0 || t.raw.isEmpty()) return;
        long first = t.firstNs;
        if (first == ageAskedFor) return;
        ageAskedFor = first; // the value, not the tail: a sealed tail must not stay reachable
        long left = age * 1_000_000L - (System.nanoTime() - first);
        if (left > 0) env.schedule(this::maintain, left + 1_000_000L); // a tail that is old already is due now
    }

    private boolean due() {
        return (ready(HostEnv.BuildKind.SEAL) && sealDue()) || (ready(HostEnv.BuildKind.MERGE) && pickMerge(view.segs, false) != null);
    }

    private void submit() {
        if (work.isHeldByCurrentThread()) return; // inside a build of this thread (a same-thread executor): the work is looked at again later
        if (!queued.compareAndSet(false, true)) return; // the task that is queued or running looks again when it ends
        try {
            env.executor().execute(this::maintenanceTask);
            rejected = false;
        } catch (RuntimeException refused) {
            queued.set(false);
            rejected = true;
            lastError = refused;
            count(failures, C_FAILURES, 1);
        }
    }

    private void maintenanceTask() {
        try {
            while (!closed) {
                boolean did;
                work.lock();
                try {
                    did = !closed && runOne();
                } finally {
                    work.unlock();
                }
                if (!did) break;
            }
        } catch (Throwable t) {
            if (!closed) {
                lastError = t;
                count(failures, C_FAILURES, 1);
            }
        } finally {
            queued.set(false);
        }
        if (!closed && due()) submit(); // what became due after the last look, while the task still held its place
    }

    /** under the work lock: one background seal or merge, when one is due and not waiting for a retry; true when it ran */
    private boolean runOne() {
        if (sealDue() && ready(HostEnv.BuildKind.SEAL)) {
            boolean ran = false;
            try {
                boolean did = doSeal(false);
                ran = true;
                return did;
            } finally {
                if (!ran && !closed) {
                    sealFailStreak++;
                    if (hosted) retryLater(HostEnv.BuildKind.SEAL);
                }
            }
        }
        if (ready(HostEnv.BuildKind.MERGE) && pickMerge(view.segs, false) != null) {
            boolean ran = false;
            try {
                boolean did = doMerge(false);
                ran = true;
                return did;
            } finally {
                if (!ran && !closed) {
                    mergeFailStreak++;
                    if (hosted) retryLater(HostEnv.BuildKind.MERGE);
                }
            }
        }
        return false;
    }

    private boolean ready(HostEnv.BuildKind kind) {
        long at = retryAt.get(kind.ordinal());
        return at == NOW || System.nanoTime() - at >= 0;
    }

    /** a background build of this kind waits before its next try, twice as long as the last time, up to five seconds */
    private void retryLater(HostEnv.BuildKind kind) {
        int k = kind.ordinal();
        long wait = Math.min(MAX_BACKOFF, Math.max(FIRST_BACKOFF, backoff[k] * 2));
        backoff[k] = wait;
        retryAt.set(k, System.nanoTime() + wait);
        env.schedule(this::maintain, wait); // a host without a timer returns false: the next write or maintain() tries again
    }

    /** the permit for a build of {@code rows} rows, or null when the host refused it, and then a background build waits before its next try */
    private HostEnv.Permit permit(HostEnv.BuildKind kind, long rows) {
        HostEnv.Permit p = env.acquireBuild(kind, estimatedBuildBytes(rows));
        if (p == null) {
            count(deferrals, C_DEFERRED, 1);
            retryLater(kind);
        } else {
            retryAt.set(kind.ordinal(), NOW);
        }
        return p;
    }

    /** under the work lock: a build of this kind ended well, so the next refusal or failure starts at the first back-off again */
    private boolean built(HostEnv.BuildKind kind, boolean done) {
        backoff[kind.ordinal()] = 0; // not on a granted permit: a build that fails every time keeps backing off
        return done;
    }

    /**
     * An estimate of the memory that a graph build over {@code rows} rows needs at its peak: the graph being built and its sealed copy, each
     * with the vectors (and codes) and about 12 * M bytes of links per row, and 200 bytes per row around them. It is arithmetic and has not
     * been compared with measurements (issue 13).
     */
    long estimatedBuildBytes(long rows) {
        GraphIndex.Params g = cfg.graph();
        long vector = 4L * dim + (g.sq8() ? dim + 12L : 0L);
        return rows * (2L * (vector + 12L * g.M()) + 200L);
    }

    /** the seal threshold in force: {@code tailLimit}, or less when the memory budget of the host is small (about an eighth of it for the tail) */
    int effectiveTailLimit() {
        int limit = cfg.tailLimit();
        if (!hosted) return limit;
        long budget = env.memoryBudgetBytes();
        if (budget == Long.MAX_VALUE) return limit;
        long perRow = 8L * dim + 128L; // the flat copy and the raw copy of a tail vector, and the maps around them
        return (int) Math.max(Math.min(MIN_TAIL, limit), Math.min(limit, budget / 8 / perRow));
    }

    private int hardTail() {
        return (int) Math.min(Integer.MAX_VALUE / 2, (long) cfg.hardTailFactor() * Math.max(effectiveTailLimit(), cfg.smallLimit()));
    }

    /** after a write, outside any lock: wakes the worker when the tail is full, or lets a hosted tier look for work */
    private void afterWrite(boolean tailFull) {
        if (hosted) maintain();
        else if (tailFull && wake.availablePermits() == 0) wake.release();
    }
    public int segmentCount() { return view.segs.length; }
    /** vectors in the tails (active plus frozen): what a SEALED_ONLY search does not see */
    public int pendingTail() {
        View v = view;
        return v.active.raw.size() + (v.frozen != null ? v.frozen.raw.size() : 0);
    }

    /** test hooks: run on the sealing thread after the segment is built and before it is published, and right after the swap */
    public void setSealHooks(Runnable beforePublish, Runnable afterPublish) {
        hookBeforePublish = beforePublish;
        hookAfterPublish = afterPublish;
    }

    /** test hook: runs inside {@link #apply}, under the write lock, after the adds of a batch and before its deletes */
    void setBatchHook(Runnable betweenAddsAndDeletes) {
        hookInBatch = betweenAddsAndDeletes;
    }

    // ---- writes ----

    /** under mu: whether a segment of the view holds a visible copy of id */
    private static boolean visibleInSegments(View v, long id) {
        for (Segment s : v.segs) {
            int row = s.rowOf(id);
            if (row >= 0 && !s.isHidden(row)) return true;
        }
        return false;
    }

    /** under mu: hides every visible copy of id in the segments of the view, at the version {@code ver}; true when there was one */
    private static boolean hideInSegments(View v, long id, long ver) {
        boolean was = false;
        for (Segment s : v.segs) {
            int row = s.rowOf(id);
            if (row >= 0 && !s.isHidden(row)) {
                s.hide(row, ver);
                was = true;
            }
        }
        return was;
    }

    private static int hiddenRows(Segment[] segs) {
        int d = 0;
        for (Segment s : segs) d += s.dead;
        return d;
    }

    @Override
    public void add(long id, float[] v) {
        if (v.length != dim) throw new IllegalArgumentException("dim " + v.length + " != " + dim);
        float[] copy = v.clone();
        awaitBackpressure();
        boolean wakeUp;
        mu.lock();
        try {
            checkOpen();
            wakeUp = addLocked(view, id, copy);
            publish();
            if (capture != null) capture.add(new Change(new long[] {id}, new float[][] {copy}, new long[0]));
        } finally {
            mu.unlock();
        }
        afterWrite(wakeUp);
    }

    /**
     * under mu: the one place the current view is replaced. The new view holds every write up to the version counter, so the caller has
     * finished the writes that took those versions.
     */
    private void publish(Segment[] segs, Tail frozen, Tail active) {
        view = new View(segs, frozen, active, counter);
    }

    /** under mu: lets the searches that start from now on see the writes since the last view, all of them at once */
    private void publish() {
        View v = view;
        publish(v.segs, v.frozen, v.active);
    }

    /**
     * under mu: the body of add (copy is the tier's own copy of the vector) on the parts of {@code vw}; no search sees it before the
     * caller publishes. True when the active tail has reached tailLimit.
     */
    private boolean addLocked(View vw, long id, float[] copy) {
        long ver = ++counter;
        addsCount++;
        boolean inActive = vw.active.raw.containsKey(id);
        boolean inFrozen = vw.frozen != null && vw.frozen.raw.containsKey(id);
        vw.active.put(id, new Rec(copy, ver));            // 1. the new version of the id, in a new slot of the active tail
        boolean inSegment = hideInSegments(vw, id, ver);  // 2. an add hides every older copy in the segments
        if (inFrozen) vw.frozen.hide(id, ver);            // 3. the old frozen copy goes
        if (!(inActive || inFrozen || inSegment)) live.incrementAndGet();
        return vw.active.raw.size() >= effectiveTailLimit();
    }

    @Override
    public boolean delete(long id) {
        boolean was;
        mu.lock();
        try {
            checkOpen();
            was = deleteLocked(view, id);
            if (was) publish();
            if (capture != null) capture.add(new Change(new long[0], new float[0][], new long[] {id}));
        } finally {
            mu.unlock();
        }
        afterWrite(false);
        return was;
    }

    /** under mu: the body of delete on the parts of {@code vw}; no search sees it before the caller publishes. True when the id was live. */
    private boolean deleteLocked(View vw, long id) {
        boolean inActive = vw.active.raw.containsKey(id);
        boolean inFrozen = vw.frozen != null && vw.frozen.raw.containsKey(id);
        if (!(inActive || inFrozen || visibleInSegments(vw, id))) return false;
        long ver = ++counter;
        if (inActive) vw.active.remove(id, ver);
        if (inFrozen) vw.frozen.hide(id, ver);
        hideInSegments(vw, id, ver); // a delete hides every copy in the segments
        live.decrementAndGet();
        return true;
    }

    /**
     * Applies the changes of one committed transaction so that a search sees all of them or none of them: the adds first (in list order), then the deletes.
     * After the call returns, a search that starts later sees the adds and not the deletes. Applying the same batch again changes nothing visible, because an
     * add of a held id replaces it with the same vector and a delete of an absent id changes nothing (the host guarantees that an id denotes one vector).
     *
     * <p>The call takes the write lock once for the whole batch. It never waits for the worker (a seal or a merge that is running), and it does not look at
     * the hard tail limit: a batch may take the tail above it, and the host asks {@link #awaitCapacity} before the call when it wants back pressure. A vector
     * of a wrong dimension throws {@link IllegalArgumentException} before anything is applied.
     *
     * <p>How it is atomic. Every change takes a version, and nothing that a search reads shows a change before it is published: new rows
     * lie beyond the slot counts of the view that the search holds, and new marks are above its version. The call publishes one view after
     * all of its changes, with one volatile write. A search that read an older view answers without the batch, and one that reads the new
     * view sees all of it. So a batch never delays a search, and a search never scans again, however fast the batches follow each other.
     * {@link #add} and {@link #delete} publish the same way, one write at a time.
     */
    public void apply(Batch batch) {
        int na = batch.adds().size();
        float[][] copies = new float[na][];
        for (int i = 0; i < na; i++) {
            float[] v = batch.adds().get(i).v();
            if (v.length != dim) throw new IllegalArgumentException("dim " + v.length + " != " + dim);
            copies[i] = v.clone();
        }
        boolean wakeUp;
        mu.lock();
        try {
            checkOpen();
            View vw = view;
            try {
                for (int i = 0; i < na; i++) addLocked(vw, batch.adds().get(i).id(), copies[i]);
                Runnable between = hookInBatch;
                if (between != null) between.run();
                for (long id : batch.deletes()) deleteLocked(vw, id);
                if (capture != null) {
                    long[] addIds = new long[na];
                    for (int i = 0; i < na; i++) addIds[i] = batch.adds().get(i).id();
                    long[] delIds = new long[batch.deletes().size()];
                    for (int i = 0; i < delIds.length; i++) delIds[i] = batch.deletes().get(i);
                    capture.add(new Change(addIds, copies, delIds));
                }
            } finally {
                publish(); // one write after every change of the batch: from here on a search sees all of it
            }
            wakeUp = view.active.raw.size() >= effectiveTailLimit();
        } finally {
            mu.unlock();
        }
        afterWrite(wakeUp);
    }

    /**
     * Back pressure as a pre-check for {@link #apply}. When the active tail holds {@code hardTailFactor * max(tailLimit, smallLimit)} vectors or more, waits
     * (it wakes the worker and parks) until the tail has room, and returns {@code false} when the deadline passes first, when the tier is closed, or when the
     * worker is dead. Returns {@code true} at once when there is room. The deadline is on the {@link System#nanoTime()} clock. A call that is waiting returns
     * {@code true} as soon as a seal or a delete makes room. {@link #apply} never calls it; {@link #add} calls it with a deadline 10 s away and adds anyway.
     */
    public boolean awaitCapacity(long deadlineNanos) {
        while (view.active.raw.size() >= hardTail()) {
            if (closed || (!hosted && !worker.isAlive()) || deadlineNanos - System.nanoTime() <= 0L) return false;
            count(backpressure, C_BACKPRESSURE, 1);
            afterWrite(true);
            LockSupport.parkNanos(200_000L);
        }
        return true;
    }

    private void awaitBackpressure() {
        awaitCapacity(System.nanoTime() + 10_000_000_000L);
    }

    // ---- search ----

    @Override
    public Hit[] search(float[] q, int k, LongPredicate allow) {
        return search(q, k, allow, Freshness.FRESH).hits();
    }

    /**
     * {@link #search(float[], int, LongPredicate)} that stops when {@code context} says so.
     *
     * @param q the query vector
     * @param k the most hits to return
     * @param allow the ids that may be returned, or null for no filter
     * @param context polled during the search (see {@link SearchContext}), or null for none
     * @return the hits, best first
     * @throws SearchCancelledException when the context expired before the search was done
     */
    public Hit[] search(float[] q, int k, LongPredicate allow, SearchContext context) {
        return search(q, k, allow, Freshness.FRESH, context).hits();
    }

    public SearchResult search(float[] q, int k, LongPredicate allow, Freshness freshness) {
        return searchImpl(q, k, allow, freshness, cfg.graph().efSearch(), null);
    }

    /**
     * {@link #search(float[], int, LongPredicate, Freshness)} that stops when {@code context} says so.
     *
     * @param q the query vector
     * @param k the most hits to return
     * @param allow the ids that may be returned, or null for no filter
     * @param freshness whether the tails are searched
     * @param context polled during the search (see {@link SearchContext}), or null for none
     * @return the hits with their meta data
     * @throws SearchCancelledException when the context expired before the search was done
     */
    public SearchResult search(float[] q, int k, LongPredicate allow, Freshness freshness, SearchContext context) {
        return searchImpl(q, k, allow, freshness, cfg.graph().efSearch(), context);
    }

    /** FRESH search with an explicit beam width for the segments (the harness sweeps ef through this overload) */
    public Hit[] search(float[] q, int k, LongPredicate allow, int ef) {
        return searchImpl(q, k, allow, Freshness.FRESH, ef, null).hits();
    }

    /**
     * {@link #search(float[], int, LongPredicate, int)} that stops when {@code context} says so.
     *
     * @param q the query vector
     * @param k the most hits to return
     * @param allow the ids that may be returned, or null for no filter
     * @param ef the beam width of a segment search
     * @param context polled during the search (see {@link SearchContext}), or null for none
     * @return the hits, best first
     * @throws SearchCancelledException when the context expired before the search was done
     */
    public Hit[] search(float[] q, int k, LongPredicate allow, int ef, SearchContext context) {
        return searchImpl(q, k, allow, Freshness.FRESH, ef, context).hits();
    }

    private SearchResult searchImpl(float[] q, int k, LongPredicate allow, Freshness freshness, int efSearch, SearchContext context) {
        checkOpen();
        if (q.length != dim) throw new IllegalArgumentException("dim " + q.length + " != " + dim);
        if (k <= 0) return new SearchResult(NONE, true, 0, 0);
        Poll poll = context == null ? null : new Poll(context);
        if (poll != null) poll.now();
        View vw = view; // the version that this search answers from, whatever is written while it scans
        try {
            return searchView(vw, q, k, allow, freshness, efSearch, poll);
        } catch (SearchCancelledException stopped) {
            // a context that throws an exception of its own gets it back unchanged: only the one of the poll can carry hits
            if (poll != null && stopped == poll.thrown && context.partialWanted()) stopped.keep(mergeTopK(poll.lists, poll.done, k));
            throw stopped;
        }
    }

    private SearchResult searchView(View vw, float[] q, int k, LongPredicate allow, Freshness freshness, int efSearch, Poll poll) {
        Segment[] segs = vw.segs;
        Hit[][] lists = new Hit[2 + segs.length][];
        int li = 0;
        int tailScanned = 0;
        if (poll != null) { // what a stop leaves behind for a partial result: a list is counted only when its source was read to the end
            poll.lists = lists;
            poll.done = 0;
        }
        if (freshness == Freshness.FRESH) {
            // a tail that was empty at this version costs nothing: no scan, no list, no merge work
            if (vw.activeRows > 0) {
                if (poll != null) poll.now();
                tailScanned += vw.activeRows;
                Hit[] h = vw.active.flat.searchAt(q, k, allow, poll, vw.activeSlots, vw.activeRows, vw.at);
                if (h.length > 0) lists[li++] = h;
                if (poll != null) poll.done = li;
            }
            if (vw.frozenRows > 0) {
                if (poll != null) poll.now();
                tailScanned += vw.frozenRows;
                Hit[] h = vw.frozen.flat.searchAt(q, k, allow, poll, vw.frozenSlots, vw.frozenRows, vw.at);
                if (h.length > 0) lists[li++] = h;
                if (poll != null) poll.done = li;
            }
        }
        int searched = 0;
        double scanBelow = filterScanBelow;
        int ef = Math.max(k, efSearch);
        boolean small = FilterPlan.smallByHint(allow, ef);
        long sizeSum = 0; // splitBeam only: rows of the segments that will be searched; 0 (the default) keeps the full beam ef in every segment
        if (cfg.splitBeam() && segs.length > 1) {
            for (Segment t : segs) if (t.n > 0 && !t.goneAt(vw.at)) sizeSum += t.n;
        }
        for (int i = segs.length - 1; i >= 0; i--) { // newest segment first
            Segment s = segs[i];
            if (s.n == 0 || s.goneAt(vw.at)) continue;
            IntPredicate rows = s.rowsAt(vw.at);
            int efS = sizeSum > 0 ? Math.max(k, (int) ((ef * (long) s.n + sizeSum - 1) / sizeSum)) : ef; // ceil(ef * rows(s) / sizeSum), at least k
            if (poll != null) poll.now();
            Hit[] h = small ? s.g.scan(q, k, allow, rows, poll) : s.g.search(q, k, allow, rows, efS, scanBelow, poll);
            if (h.length > 0) lists[li++] = h;
            if (poll != null) poll.done = li;
            searched++;
        }
        boolean exact = (searched == 0 || small) && (cfg.tailQuant() == Quant.NONE || (long) FLAT_RERANK * k >= tailScanned);
        return new SearchResult(mergeTopK(lists, li, k), exact, tailScanned, searched);
    }

    private static final Comparator<Hit> BEST_FIRST = (a, b) -> {
        int c = Float.compare(b.score(), a.score());
        return c != 0 ? c : Long.compare(a.id(), b.id());
    };

    /** lists are in newest-source-first order, each sorted best first; the first copy of an id wins; result best first, at most k */
    private static Hit[] mergeTopK(Hit[][] lists, int nl, int k) {
        int total = 0;
        for (int i = 0; i < nl; i++) total += lists[i].length;
        if (total == 0 || nl == 0) return NONE;
        if (nl == 1) return lists[0].length <= k ? lists[0] : Arrays.copyOf(lists[0], k);
        int cap = 8;
        while (cap < total * 2) cap <<= 1;
        int mask = cap - 1;
        long[] keys = new long[cap];
        boolean[] used = new boolean[cap];
        Hit[] out = new Hit[total];
        int n = 0;
        for (int i = 0; i < nl; i++) {
            for (Hit h : lists[i]) {
                int slot = (int) ((h.id() * 0x9E3779B97F4A7C15L) >>> 40) & mask;
                boolean dup = false;
                while (used[slot]) {
                    if (keys[slot] == h.id()) { dup = true; break; }
                    slot = (slot + 1) & mask;
                }
                if (dup) continue;
                used[slot] = true;
                keys[slot] = h.id();
                out[n++] = h;
            }
        }
        Arrays.sort(out, 0, n, BEST_FIRST);
        return Arrays.copyOf(out, Math.min(k, n));
    }

    // ---- worker: seal and merge, one at a time ----

    private void workerLoop() {
        while (!closed) {
            try {
                wake.tryAcquire(50, TimeUnit.MILLISECONDS);
                wake.drainPermits();
                while (!closed) {
                    boolean did = false;
                    work.lock();
                    try {
                        if (closed) break;
                        did = runOne();
                    } finally {
                        work.unlock();
                    }
                    if (!did) break;
                }
            } catch (InterruptedException e) {
                return;
            } catch (Throwable t) {
                if (closed) return;
                lastError = t;
                count(failures, C_FAILURES, 1);
                LockSupport.parkNanos(200_000_000L); // do not spin on a failing task
            }
        }
    }

    private boolean sealDue() {
        View vw = view;
        if (vw.frozen != null) return true; // a failed seal is retried
        int n = vw.active.raw.size();
        if (n == 0 || live.get() < cfg.smallLimit()) return false;
        if (n >= effectiveTailLimit()) return true;
        long age = cfg.tailAgeMs();
        return age > 0 && System.nanoTime() - vw.active.firstNs > age * 1_000_000L;
    }

    /** the segments one merge would consume, or null when none is due; force: all segments (or the lone one when it has hidden copies) */
    private List<Segment> pickMerge(Segment[] segs, boolean force) {
        int ns = segs.length;
        if (ns == 0) return null;
        if (force) {
            if (ns >= 2 || segs[0].dead > 0) return new ArrayList<>(Arrays.asList(segs));
            return null;
        }
        // the size-ratio trigger; mergeTriggerRatio 0 (the default) skips this block, and cand below is segs itself, so the count rule sees every segment
        if (cfg.mergeTriggerRatio() > 0.0 && ns >= 2) {
            int big = largestIndex(segs);
            long bigLive = segs[big].n - segs[big].dead, smallLive = 0;
            for (int i = 0; i < ns; i++) if (i != big) smallLive += segs[i].n - segs[i].dead;
            if ((double) smallLive > cfg.mergeTriggerRatio() * (double) bigLive) return new ArrayList<>(Arrays.asList(segs));
        }
        if (ns > cfg.maxSegments()) {
            Segment[] cand = segs;
            if (cfg.mergeTriggerRatio() > 0.0 && cfg.maxSegments() >= 2) { // the small segments merge among themselves; the big one only through the ratio rule above
                int big = largestIndex(segs);
                cand = new Segment[ns - 1];
                for (int i = 0, j = 0; i < ns; i++) if (i != big) cand[j++] = segs[i];
            }
            Segment[] byLive = cand.clone();
            Arrays.sort(byLive, (a, b) -> Integer.compare(a.n - a.dead, b.n - b.dead));
            int k = Math.max(2, Math.max(ns - cfg.maxSegments() + 1, Math.min(cfg.mergeFan(), cand.length)));
            k = Math.min(k, cand.length);
            return new ArrayList<>(Arrays.asList(byLive).subList(0, k));
        }
        Segment best = null;
        double bestFrac = 0;
        for (Segment s : segs) {
            if (s.n > 0 && s.dead > 0) {
                double f = (double) s.dead / s.n;
                if (f >= cfg.deadFraction() && f > bestFrac) {
                    bestFrac = f;
                    best = s;
                }
            }
        }
        if (best == null) return null;
        List<Segment> one = new ArrayList<>();
        one.add(best);
        return one;
    }

    /** index of the segment with the most live rows (n - dead); ties go to the oldest (lowest index) */
    private static int largestIndex(Segment[] segs) {
        int best = 0;
        for (int i = 1; i < segs.length; i++) if (segs[i].n - segs[i].dead > segs[best].n - segs[best].dead) best = i;
        return best;
    }

    /** under the work lock: a seal when there is something to seal and the host permits the build; ignoreSmall is the call of {@link #sealNow} */
    private boolean doSeal(boolean ignoreSmall) {
        View v = view;
        Tail tail = v.frozen != null ? v.frozen : v.active;
        if (v.frozen == null && (tail.raw.isEmpty() || (!ignoreSmall && live.get() < cfg.smallLimit()))) return false;
        HostEnv.Permit granted = permit(HostEnv.BuildKind.SEAL, tail.raw.size());
        if (granted == null) {
            if (ignoreSmall) throw new BuildRefusedException("the host refused the build of a seal");
            return false;
        }
        try (granted) {
            return built(HostEnv.BuildKind.SEAL, seal(ignoreSmall));
        }
    }

    private boolean seal(boolean ignoreSmall) {
        long t0 = System.nanoTime();
        activity = 1;
        try {
            Tail frozen;
            long[] ids;
            Rec[] recs;
            int n;
            mu.lock();
            try {
                View v = view;
                if (v.frozen == null) {
                    if (v.active.raw.isEmpty()) return false;
                    if (!ignoreSmall && live.get() < cfg.smallLimit()) return false;
                    publish(v.segs, v.active, new Tail(dim, metric, cfg.tailQuant()));
                    frozen = v.active;
                } else {
                    frozen = v.frozen;
                }
                n = frozen.raw.size();
                ids = new long[n];
                recs = new Rec[n];
                int i = 0;
                for (Map.Entry<Long, Rec> e : frozen.raw.entrySet()) {
                    if (i == n) break;
                    ids[i] = e.getKey();
                    recs[i] = e.getValue();
                    i++;
                }
                if (i != n) throw new IllegalStateException("frozen tail changed under the lock");
            } finally {
                mu.unlock();
            }
            if (n == 0) {
                mu.lock();
                try {
                    View v = view;
                    publish(v.segs, null, v.active);
                } finally {
                    mu.unlock();
                }
                return true;
            }
            long[] vers0 = new long[n];
            for (int i = 0; i < n; i++) vers0[i] = recs[i].ver;
            int[] ord = orderByVersion(vers0, n);
            final Rec[] fr = new Rec[n];
            long[] vers = new long[n];
            ids = permute(ids, ord);
            for (int i = 0; i < n; i++) {
                fr[i] = recs[ord[i]];
                vers[i] = vers0[ord[i]];
            }
            SealedGraph sg = build(ids, n, j -> fr[j].v, false);
            if (sg.n != n) throw new IllegalStateException("sealed rows " + sg.n + " != " + n);
            Segment seg = new Segment(sg, ids, vers);
            Runnable before = hookBeforePublish;
            if (before != null) before.run();
            mu.lock();
            try {
                View v = view;
                for (Mark m : frozen.late) seg.hideOlder(m.id(), m.ver()); // the rows that were hidden while the build ran
                Segment[] ns = Arrays.copyOf(v.segs, v.segs.length + 1);
                ns[ns.length - 1] = seg;
                publish(ns, null, v.active); // the atomic swap
            } finally {
                mu.unlock();
            }
            Runnable after = hookAfterPublish;
            if (after != null) after.run();
            sealFailStreak = 0;
            count(seals, C_SEALS, 1);
            count(sealedVectors, C_SEALED, n);
            final long sealEnd = System.nanoTime();
            final long sealDt = sealEnd - t0;
            logTask(1, t0, sealEnd, n);
            sealNanos.addAndGet(sealDt);
            maxSealNanos.accumulateAndGet(sealDt, Math::max);
            return true;
        } finally {
            activity = 0;
        }
    }

    /**
     * under the work lock: a merge when one is due and the host permits the build. A background merge also waits while the bytes held plus the
     * estimate are above the memory budget; force is the call of {@link #mergeNow}.
     */
    private boolean doMerge(boolean force) {
        List<Segment> due = pickMerge(view.segs, force);
        if (due == null || due.isEmpty()) return false;
        long rows = 0;
        for (Segment s : due) rows += s.n;
        if (!force && hosted) {
            long budget = env.memoryBudgetBytes();
            if (budget != Long.MAX_VALUE && bytesHeld() + estimatedBuildBytes(rows) > budget) {
                count(deferrals, C_DEFERRED, 1);
                retryLater(HostEnv.BuildKind.MERGE);
                return false;
            }
        }
        HostEnv.Permit granted = permit(HostEnv.BuildKind.MERGE, rows);
        if (granted == null) {
            if (force) throw new BuildRefusedException("the host refused the build of a merge");
            return false;
        }
        try (granted) {
            return built(HostEnv.BuildKind.MERGE, merge(force));
        }
    }

    private boolean merge(boolean force) {
        long t0 = System.nanoTime();
        View v0 = view;
        List<Segment> picked = pickMerge(v0.segs, force);
        if (picked == null || picked.isEmpty()) return false;
        final Segment largest = v0.segs[largestIndex(v0.segs)];
        final boolean intoLargest = picked.contains(largest); // this merge rewrites the largest segment
        // MergeMode.INSERT: the graph of the largest segment is copied and the rest inserted, unless too much of it is hidden
        final boolean insert = cfg.mergeMode() == MergeMode.INSERT && intoLargest && picked.size() > 1 && largest.dead <= cfg.deadFraction() * largest.n;
        activity = 2;
        try {
            int total = 0;
            for (Segment s : picked) total += s.g.n;
            long[] ids = new long[total];
            long[] vers = new long[total];
            Segment[] srcSeg = new Segment[total];
            int[] srcRow = new int[total];
            int m = 0;
            final boolean[] kept = insert ? new boolean[largest.g.n] : null; // the rows of the copied graph that stay in it
            // how many rows each input had hidden when the reading began: those are seen below, the later ones are carried over at the swap
            final int[] seenDead = new int[picked.size()];
            mu.lock();
            try {
                for (int i = 0; i < seenDead.length; i++) seenDead[i] = picked.get(i).dead;
            } finally {
                mu.unlock();
            }
            for (Segment s : picked) {
                for (int r = 0; r < s.g.n; r++) {
                    if (s.isHidden(r)) continue; // hidden copy: dropped here
                    long id = s.g.ids[r];
                    long ver = s.vers[s.indexOf(id)];
                    ids[m] = id;
                    vers[m] = ver;
                    srcSeg[m] = s;
                    srcRow[m] = r;
                    if (insert && s == largest) kept[r] = true;
                    m++;
                }
            }
            Segment merged = null;
            int built = m; // the rows that this merge put into a graph
            if (m > 0) {
                int[] ord = orderByVersion(Arrays.copyOf(vers, m), m);
                final Segment[] fss = new Segment[m];
                final int[] frr = new int[m];
                for (int i = 0; i < m; i++) {
                    fss[i] = srcSeg[ord[i]];
                    frr[i] = srcRow[ord[i]];
                }
                final long[] mi = permute(Arrays.copyOf(ids, m), ord);
                final long[] mv = permute(Arrays.copyOf(vers, m), ord);
                final VecSource rows = j -> fss[j].g.rawVector(frr[j]);
                final SealedGraph sg;
                if (insert) {
                    GraphIndex gi = GraphIndex.fromSealed(largest.g);
                    for (int r = 0; r < kept.length; r++) {
                        if (!kept[r]) gi.delete(largest.g.ids[r]); // before the inserts: another input may hold the newer vector of the id
                    }
                    final long[] others = new long[m];
                    final int[] at = new int[m];
                    int no = 0;
                    for (int i = 0; i < m; i++) {
                        if (fss[i] != largest) {
                            others[no] = mi[i];
                            at[no++] = i;
                        }
                    }
                    insertAll(gi, others, no, j -> rows.vec(at[j]), true);
                    built = no;
                    sg = gi.seal(); // seal repairs the neighbor lists that pointed at deleted rows
                } else {
                    sg = build(mi, m, rows, true);
                }
                if (sg.n != m) throw new IllegalStateException("merged rows " + sg.n + " != " + m);
                merged = new Segment(sg, mi, mv);
            }
            mu.lock();
            try {
                View v = view;
                List<Segment> keep = new ArrayList<>();
                for (Segment s : v.segs) if (!picked.contains(s)) keep.add(s);
                if (merged != null) {
                    carryOver(picked, seenDead, merged); // the rows that were hidden while the merge ran
                    keep.add(merged);
                }
                Segment[] ns = keep.toArray(new Segment[0]);
                publish(ns, v.frozen, v.active); // the atomic swap
            } finally {
                mu.unlock();
            }
            mergeFailStreak = 0;
            count(merges, C_MERGES, 1);
            count(mergedVectors, C_MERGED, built);
            if (intoLargest) { count(bigMerges, C_BIG, 1); count(bigMergedVectors, C_BIG_MERGED, built); }
            final long mergeEnd = System.nanoTime();
            final long mergeDt = mergeEnd - t0;
            logTask(2, t0, mergeEnd, built);
            mergeNanos.addAndGet(mergeDt);
            maxMergeNanos.accumulateAndGet(mergeDt, Math::max);
            return true;
        } finally {
            activity = 0;
        }
    }

    /**
     * under mu: hides in the merged segment the copies of the rows that were hidden in an input after the merge counted them, because the
     * merge may have read them as visible. Marks never go away, so input {@code i} has them from position {@code seenDead[i]} on.
     */
    private static void carryOver(List<Segment> inputs, int[] seenDead, Segment merged) {
        for (int i = 0; i < seenDead.length; i++) {
            Segment s = inputs.get(i);
            for (int j = seenDead[i]; j < s.dead; j++) {
                int row = s.hiddenRow(j);
                merged.hideOlder(s.g.ids[row], s.hiddenVersion(row));
            }
        }
    }

    /** build and seal a graph over n rows (ids[i], src.vec(i)); throttle: apply the merge rate cap */
    private SealedGraph build(long[] ids, int n, VecSource src, boolean throttle) {
        GraphIndex gi = new GraphIndex(dim, metric, cfg.graph());
        insertAll(gi, ids, n, src, throttle);
        return gi.seal();
    }

    /** insert n rows (ids[i], src.vec(i)) into a graph with the build threads; throttle: apply the merge rate cap */
    private void insertAll(GraphIndex gi, long[] ids, int n, VecSource src, boolean throttle) {
        int th = Math.max(1, Math.min(cfg.buildThreads(), n / 256 + 1));
        int rate = throttle ? cfg.mergeRatePerSec() : sealRate; // sealRate is 0 (unthrottled) unless a benchmark sets it
        if (th == 1) {
            addRange(gi, ids, src, 0, n, rate);
        } else if (hosted) {
            // th ranges, claimed one by one by the calling thread and by helper tasks of the host's executor. The caller waits only for ranges
            // that a helper has claimed, so a helper that never starts (an executor with one thread) cannot stop the build.
            final int per = (n + th - 1) / th;
            final int perRate = rate > 0 ? Math.max(1, rate / th) : 0;
            final int ranges = th;
            final AtomicInteger next = new AtomicInteger();
            final CountDownLatch done = new CountDownLatch(ranges);
            final AtomicReference<Throwable> err = new AtomicReference<>();
            Runnable claim = () -> {
                for (int t = next.getAndIncrement(); t < ranges; t = next.getAndIncrement()) {
                    try {
                        if (err.get() == null) addRange(gi, ids, src, Math.min(n, t * per), Math.min(n, (t + 1) * per), perRate);
                    } catch (Throwable e) {
                        err.compareAndSet(null, e);
                    } finally {
                        done.countDown();
                    }
                }
            };
            for (int t = 1; t < ranges; t++) {
                try {
                    env.executor().execute(claim);
                } catch (RuntimeException refused) {
                    break; // the calling thread claims what no helper takes
                }
            }
            claim.run();
            try {
                done.await();
            } catch (InterruptedException e) {
                throw new IllegalStateException(e);
            }
            if (err.get() instanceof ClosedException c) throw c;
            if (err.get() != null) throw new IllegalStateException(err.get());
        } else {
            final int per = (n + th - 1) / th;
            final int perRate = rate > 0 ? Math.max(1, rate / th) : 0;
            final Throwable[] err = new Throwable[1];
            Thread[] ts = new Thread[th];
            for (int t = 0; t < th; t++) {
                final int lo = Math.min(n, t * per);
                final int hi = Math.min(n, lo + per);
                ts[t] = new Thread(() -> {
                    try {
                        addRange(gi, ids, src, lo, hi, perRate);
                    } catch (Throwable e) {
                        err[0] = e;
                    }
                }, "tiered-build-" + t);
                ts[t].start();
            }
            for (Thread t : ts) {
                try {
                    t.join();
                } catch (InterruptedException e) {
                    throw new IllegalStateException(e);
                }
            }
            if (err[0] instanceof ClosedException c) throw c;
            if (err[0] != null) throw new IllegalStateException(err[0]);
        }
    }

    private void addRange(GraphIndex gi, long[] ids, VecSource src, int lo, int hi, int rate) {
        long t0 = System.nanoTime();
        for (int i = lo; i < hi; i++) {
            if (closed) throw new ClosedException("the index was closed during a build");
            gi.add(ids[i], src.vec(i));
            if (rate > 0 && ((i - lo) & 127) == 127) {
                long due = t0 + (long) (i - lo + 1) * 1_000_000_000L / rate;
                long now = System.nanoTime();
                if (due > now) LockSupport.parkNanos(due - now);
            }
        }
    }

    // ---- control for tests and reports ----

    /** seal the active tail now, on the calling thread, even below smallLimit; waits for a running worker task first */
    public void sealNow() {
        work.lock();
        try {
            checkOpen();
            doSeal(true);
        } finally {
            work.unlock();
        }
    }

    /** merge every segment into one (or compact the lone one when it has hidden copies), on the calling thread */
    public void mergeNow() {
        work.lock();
        try {
            checkOpen();
            doMerge(true);
        } finally {
            work.unlock();
        }
    }

    /** waits until no seal or merge is due or running (for a quiet writer); false on timeout. Searches that are running do not delay it. */
    public boolean awaitIdle(long timeoutMs) {
        long end = System.nanoTime() + timeoutMs * 1_000_000L;
        while (System.nanoTime() < end) {
            if (!sealDue() && pickMerge(view.segs, false) == null && work.tryLock()) {
                work.unlock();
                if (!sealDue() && pickMerge(view.segs, false) == null) return true;
            }
            if (hosted ? closed : !worker.isAlive()) return false;
            maintain(); // a standalone tier ignores it
            LockSupport.parkNanos(2_000_000L);
        }
        return false;
    }

    /**
     * Rebuild everything from raw vectors, as the engine would after a crash (records are the truth): tail, segments and marks are
     * replaced. n below smallLimit: the vectors go to the exact tail; otherwise one segment is built from all of them. Writers wait for
     * the duration; searches keep using the old state until the new one is published. A search that started before answers from the old
     * state, which it keeps as long as it runs. A later duplicate id replaces an earlier one. This form keeps the epoch as it is; a host
     * that writes while the rebuild runs uses {@link #rebuildFrom(Supplier, long)}.
     */
    public void rebuildFrom(Iterable<IdVector> src) {
        work.lock();
        mu.lock();
        try {
            checkOpen();
            View next = prepare(buildFrom(src));
            publish(next.segs, null, next.active);
        } finally {
            mu.unlock();
            work.unlock();
        }
    }

    /**
     * Rebuild while the host keeps writing (docs/ID-EPOCH-AND-REPLAY.md, "Rebuild while the host keeps writing"). The library starts to record every
     * {@link #add}, {@link #delete} and {@link #apply} it accepts, and only then asks {@code source} for the {@link Snapshot}: the records and the snapshot
     * position, the highest id that the records cover. The build runs without the write lock, so writes and searches go on against the old state,
     * which serves them until the swap. At the swap, under one hold of the write lock, the recorded writes are applied to the rebuilt state in arrival
     * order, the adds whose id is at or below the snapshot position excepted (the snapshot already holds them), then the rebuilt state is published
     * with {@code epoch}, as one version (a search sees the old state or the whole new one, and never waits), and recording stops. A write that
     * arrives after that goes to the new state directly. So a write is either in the snapshot or recorded, and a recorded one is applied once: none is lost, none twice (adding
     * a held id again would change nothing visible anyway).
     *
     * <p>The host must take the snapshot inside {@code source}, and serialize that with its own calls of {@code apply} (for instance with the monitor
     * that it holds around commit and apply): a write that it applied before the library started to record, and that the snapshot does not cover,
     * would be lost. The supplier runs on the calling thread while the library holds its work lock (a seal, a merge and a checkpoint wait for it) and
     * not the write lock. When {@code source} or the build throws, nothing changes. A seal or a merge does not run during the build, so the old tail
     * grows, and {@link #add} waits for back pressure as usual; {@link #apply} never waits. The catch-up at the swap takes time proportional to the
     * number of writes that were recorded.
     *
     * @param epoch the epoch of the host's numbering of ids from now on; it is not {@link Manifest#NONE}
     */
    public void rebuildFrom(Supplier<Snapshot> source, long epoch) {
        checkEpoch(epoch);
        work.lock();
        try {
            mu.lock();
            try {
                checkOpen();
                capture = new ArrayList<>();
            } finally {
                mu.unlock();
            }
            boolean swapped = false;
            try {
                Snapshot snap = source.get();
                Built built = buildFrom(snap.records());
                mu.lock();
                try {
                    View next = prepare(built); // the rebuilt state, which no search sees yet
                    try {
                        this.epoch = epoch;
                        for (Change c : capture) replay(next, c, snap.position());
                    } finally {
                        publish(next.segs, null, next.active); // one write: a search sees the old state, or this one with every recorded write
                    }
                    swapped = true;
                } finally {
                    capture = null;
                    mu.unlock();
                }
            } finally {
                if (!swapped) {
                    mu.lock();
                    try {
                        capture = null;
                    } finally {
                        mu.unlock();
                    }
                }
            }
        } finally {
            work.unlock();
        }
        afterWrite(view.active.raw.size() >= effectiveTailLimit());
    }

    /** {@link #rebuildFrom(Supplier, long)} for a host that hands over its records and the snapshot position, and that holds back its own writes from the moment it took the position until this call has started */
    public void rebuildFrom(Iterable<IdVector> src, long epoch, long snapshotPosition) {
        rebuildFrom(() -> new Snapshot(snapshotPosition, src), epoch);
    }

    /** the id epoch of the tier: the one the last rebuild with an epoch, or open with a record source, gave it, or the one a manifest carried; {@link Manifest#NONE} when it never had one */
    public long epoch() {
        return epoch;
    }

    private static void checkEpoch(long epoch) {
        if (epoch == Manifest.NONE) throw new IllegalArgumentException("epoch " + epoch + " is the value that stands for no epoch");
    }

    /** the records of a rebuild, with the graph that was built from them when there are enough of them */
    private record Built(long[] ids, float[][] vecs, SealedGraph graph) {}

    /** reads the records and builds the graph; takes no lock of its own (a caller holds the work lock) */
    private Built buildFrom(Iterable<IdVector> src) {
        LinkedHashMap<Long, float[]> m = new LinkedHashMap<>();
        for (IdVector iv : src) {
            if (iv.v().length != dim) throw new IllegalArgumentException("dim " + iv.v().length + " != " + dim);
            m.put(iv.id(), iv.v());
        }
        int n = m.size();
        long[] ids = new long[n];
        final float[][] vecs = new float[n][];
        int i = 0;
        for (Map.Entry<Long, float[]> e : m.entrySet()) {
            ids[i] = e.getKey();
            vecs[i] = e.getValue();
            i++;
        }
        SealedGraph sg = null;
        if (n >= cfg.smallLimit() && n > 0) {
            HostEnv.Permit granted = env.acquireBuild(HostEnv.BuildKind.REBUILD, estimatedBuildBytes(n));
            if (granted == null) throw new BuildRefusedException("the host refused a rebuild of " + n + " rows");
            try (granted) {
                sg = build(ids, n, j -> vecs[j].clone(), false);
            }
            if (sg.n != n) throw new IllegalStateException("rebuilt rows " + sg.n + " != " + n);
        }
        return new Built(ids, vecs, sg);
    }

    /** under mu and the work lock: the state of a rebuild as a view that is not published, and the live count of that state */
    private View prepare(Built b) {
        int n = b.ids().length;
        Tail nt = new Tail(dim, metric, cfg.tailQuant());
        Segment[] segs = new Segment[0];
        if (b.graph() != null) {
            long[] vers = new long[n];
            for (int j = 0; j < n; j++) vers[j] = ++counter;
            segs = new Segment[] {new Segment(b.graph(), b.ids(), vers)};
        } else {
            for (int j = 0; j < n; j++) nt.put(b.ids()[j], new Rec(b.vecs()[j].clone(), ++counter));
        }
        live.set(n);
        return new View(segs, null, nt, counter);
    }

    /** under mu: one write that arrived during a rebuild, applied to the rebuilt state; an add at or below the snapshot position is in the snapshot already */
    private void replay(View next, Change c, long position) {
        for (int i = 0; i < c.addIds().length; i++) {
            if (c.addIds()[i] > position) addLocked(next, c.addIds()[i], c.addVecs()[i]);
        }
        for (long id : c.deletes()) deleteLocked(next, id);
    }

    // ---- persistence: segments, their hidden rows, and the tail ----

    /**
     * Writes the tier into a directory as a new generation: one file per segment (a segment that an earlier checkpoint or open already put in
     * this directory is not written again), a tombstone file that marks the rows that a delete or a replace has hidden, a tail file with the
     * rows that no segment holds yet (none when the tail is empty), and a manifest, in the publish order of {@link Manifest#publish}. A process
     * kill at any point leaves the previous generation or this one, never a mixture. Files that the new generation no longer names are deleted
     * afterwards. Hidden rows stay in their segment files until a merge rewrites the segment.
     *
     * <p>No segment is built: the tail is written as it is, every checkpoint writes the whole tail again, and the next {@link #open} puts it
     * back into the tail. It refuses a directory that holds files of a newer format, with a {@link SegmentFormat.UnsupportedException}, and
     * changes nothing there. Writers and the seal and merge worker wait while it runs; searches do not. The manifest records the epoch of the
     * tier and, inside an epoch, the largest id that the segments or the tail hold.
     *
     * <p>One process owns a directory at a time. The call locks the file {@code LOCK} in it and throws {@link DirectoryLockedException}, having
     * changed nothing, when another process holds that lock. The tier keeps the lock until {@link #close}, or until a checkpoint into another
     * directory has succeeded; the tiers of one process share it.
     */
    public void checkpoint(Path dir) throws IOException {
        checkpoint(dir, FaultHook.NONE);
    }

    /** {@link #checkpoint(Path)} with a hook between the publish steps, for the crash tests */
    void checkpoint(Path dir, FaultHook hook) throws IOException {
        work.lock();
        mu.lock();
        try {
            checkOpen();
            View v = view;
            long[][] hidden = hiddenBitmaps(v.segs);
            TailFile.Rows tail = tailRows(v);
            if (!Manifest.hasManifest(dir)) SegmentFormat.createDirectoriesDurably(dir); // a first generation is not lost with its directory
            publishGeneration(dir, v, hidden, tail, hook);
        } finally {
            mu.unlock();
            work.unlock();
        }
    }

    /**
     * under mu and the work lock: the part of a checkpoint that reads and writes an existing directory, under the lock of that directory, which
     * the tier keeps when the generation was published
     */
    private void publishGeneration(Path dir, View v, long[][] hidden, TailFile.Rows tail, FaultHook hook) throws IOException {
        DirectoryLock claim = DirectoryLock.acquire(dir);
        boolean kept = false;
        try {
            Manifest cur = dir.equals(persistedDir) ? persistedManifest : null;
            if (cur == null) {
                try {
                    Manifest.readLatest(dir); // throws when the directory holds a format this version does not understand: it is not overwritten
                } catch (SegmentFormat.CorruptException damaged) {
                    // damaged files are replaced by the new generation and deleted by the cleanup below
                }
            }
            long[] hi = Manifest.highestOnDisk(dir);
            long gen = Math.max(hi[0], cur != null ? cur.generation() : 0L) + 1;
            long nextSeg = Math.max(hi[1] + 1, cur != null ? cur.nextSegId() : 1L);
            List<Manifest.Entry> entries = new ArrayList<>();
            long[] fileIds = new long[v.segs.length];
            for (int i = 0; i < v.segs.length; i++) {
                Segment s = v.segs[i];
                Manifest.Entry reuse = null;
                if (cur != null && s.fileId >= 0) {
                    for (Manifest.Entry e : cur.entries()) {
                        if (e.segId() == s.fileId) reuse = e;
                    }
                }
                if (reuse != null) {
                    entries.add(reuse);
                    fileIds[i] = reuse.segId();
                    continue;
                }
                long id = nextSeg++;
                SegmentInfo si = SegmentWriter.write(s.g, s.versByRow(), dir.resolve(Manifest.segName(id)));
                entries.add(new Manifest.Entry(id, si.n(), si.fileLength(), si.headerCrc(), si.verMin(), si.verMax()));
                fileIds[i] = id;
            }
            GraphIndex.Params gp = cfg.graph();
            Manifest.Fingerprint fp = new Manifest.Fingerprint(dim, metric, gp.M(), gp.efConstruction(), gp.sq8());
            long hw = epoch == Manifest.NONE ? Manifest.NONE : highWater(v); // the high-water id means something only inside an epoch
            Manifest m = new Manifest(gen, epoch, hw, counter, nextSeg, fp, entries, 0L, 0L, 0, 0L, 0L, 0, Manifest.FORMAT_MAJOR);
            Manifest done = Manifest.publish(dir, m, hidden, tail, hook);
            for (int i = 0; i < v.segs.length; i++) v.segs[i].fileId = fileIds[i];
            persistedDir = dir;
            persistedManifest = done;
            Manifest.gc(dir, done);
            if (!closed) { // a closed tier holds no directory
                if (dirLock != null) dirLock.close();
                dirLock = claim;
                kept = true;
            }
        } finally {
            if (!kept) claim.close();
        }
    }

    /** a tier takes the claim that an open made on its directory; when a checkpoint inside that open has given it one already, the spare is closed */
    private void hold(DirectoryLock claim) {
        if (claim == null) return;
        mu.lock();
        try {
            if (dirLock == null) dirLock = claim;
            else claim.close();
        } finally {
            mu.unlock();
        }
    }

    /**
     * Opens the tier that {@link #checkpoint} wrote. Takes the newest manifest that validates, checks its fingerprint against the
     * configuration, reads every segment with every section checksummed, the tombstone file and the tail file, deletes files that the manifest
     * does not name, and restores the version counter, the marks of the hidden rows, and the tail, so that the rows that were hidden at the checkpoint stay
     * hidden and the rows that were in the tail are in the tail again. The dimension and the metric come from the manifest. A restored tail is
     * on the heap, and the seal rules apply to it from this call on (the age of the tail counts from the open). A directory written before the
     * tail was persisted (manifest major 1) opens too. Any failure throws: {@link CorruptStateException} for damage (its cause is the
     * {@link SegmentFormat.CorruptException}), {@link SegmentFormat.UnsupportedException} for a newer format, {@link SegmentFormat.MismatchException} when the files
     * belong to another index definition, {@link NoSuchFileException} when the directory has no manifest. A caller that holds the
     * records falls back to {@link #rebuildFrom}, or uses the overload with a rebuild source that does it. A null config means the defaults.
     * {@link Config#vectorStorage()} says whether the vectors of the segments are copied into the heap or stay in the mapped files.
     *
     * <p>The directory is locked before it is read ({@link DirectoryLockedException} when another process holds it), and the returned tier keeps
     * the lock until {@link #close}, or until a checkpoint into another directory has succeeded. A failed open releases it. This holds for every
     * overload of {@code open}.
     */
    public static TieredIndex open(Path dir, Config cfg) throws IOException {
        return open(dir, cfg, HostEnv.standalone());
    }

    /** {@link #open(Path, Config)} for a tier that runs on a host environment ({@link #TieredIndex(int, Metric, Config, HostEnv)}) */
    public static TieredIndex open(Path dir, Config cfg, HostEnv env) throws IOException {
        Config c = cfg != null ? cfg : Config.defaults();
        DirectoryLock claim = DirectoryLock.acquireIfPresent(dir);
        boolean kept = false;
        try {
            Manifest m = Manifest.readLatest(dir);
            if (m == null) throw new NoSuchFileException(dir.toString(), null, "no manifest in the directory");
            TieredIndex t = restore(dir, m, m.fingerprint().dim(), m.fingerprint().metric(), c, env);
            t.hold(claim);
            kept = true;
            t.maintain();
            return t;
        } catch (SegmentFormat.CorruptException damaged) {
            throw new CorruptStateException(damaged.getMessage(), damaged);
        } finally {
            if (!kept && claim != null) claim.close();
        }
    }

    /**
     * Like {@link #open(Path, Config)} for a caller that knows the index definition and holds the records: on any failure to read the
     * directory (damage, a newer format, another fingerprint, no manifest) it builds the tier from {@code rebuildSource} with
     * {@link #rebuildFrom}, which is always correct and costs a build. The directory is not touched in that case; the next
     * {@link #checkpoint} replaces what is in it, except when the cause is a {@link SegmentFormat.UnsupportedException}: a checkpoint refuses
     * to overwrite files of a newer format. {@link #openFallbackCause()} tells which failure it was.
     */
    public static TieredIndex open(Path dir, int dim, Metric metric, Config cfg, Supplier<? extends Iterable<IdVector>> rebuildSource) {
        return open(dir, dim, metric, cfg, rebuildSource, HostEnv.standalone());
    }

    /** {@link #open(Path, int, Metric, Config, Supplier)} for a tier that runs on a host environment */
    public static TieredIndex open(Path dir, int dim, Metric metric, Config cfg, Supplier<? extends Iterable<IdVector>> rebuildSource,
                                   HostEnv env) {
        Config c = cfg != null ? cfg : Config.defaults();
        DirectoryLock claim = null;
        boolean kept = false;
        try {
            Throwable cause;
            try {
                claim = DirectoryLock.acquireIfPresent(dir);
                Manifest m = Manifest.readLatest(dir);
                if (m != null) {
                    TieredIndex t = restore(dir, m, dim, metric, c, env);
                    t.hold(claim);
                    kept = true;
                    t.maintain();
                    return t;
                }
                cause = new NoSuchFileException(dir.toString(), null, "no manifest in the directory");
            } catch (IOException e) {
                cause = e;
            }
            TieredIndex t = new TieredIndex(dim, metric, c, env);
            try {
                t.rebuildFrom(rebuildSource.get());
            } catch (RuntimeException e) {
                t.close();
                throw e;
            }
            t.fallbackCause = cause;
            t.hold(claim);
            kept = true;
            return t;
        } finally {
            if (!kept && claim != null) claim.close();
        }
    }

    /**
     * Opens the persisted tier and reconciles it with the host's records (docs/ID-EPOCH-AND-REPLAY.md, the replay protocol). The newest valid manifest is
     * read. When there is none, when it is damaged or in another format, when its fingerprint is not this index's, or when its epoch is not the epoch
     * of {@code host}, the persisted state is discarded as a whole and the tier is built from {@code host.allLive()} with {@code host}'s epoch (a cold
     * index); the directory is left as it was, and the next {@link #checkpoint} replaces its contents (except after a
     * {@link SegmentFormat.UnsupportedException}, which it throws again); {@link #openFallbackCause()} tells which case it was.
     * Otherwise the segments and the tail are read and then: (1) every held id above {@code host.recoveredMax()} is dropped (the host forgot the commit that
     * wrote it, and may mint the id again), and when any row was dropped a new generation with the lowered high-water id is published before this method
     * returns, so that a dropped row never comes back in a later generation (the segments are rebuilt first when one of them holds a dropped row; if
     * either step fails, the tier goes cold);
     * (2) every other held id for which {@code host.isLive} says no is hidden, or removed from the tail (in memory: the files are not rewritten by this
     * call; an open of the same generation asks again, and the next {@link #checkpoint} records it);
     * (3) every live record of the host above the high-water id of the generation (of its segments and its tail) is added to the tail, in increasing id
     * order, in batches. The tier is ready when the call returns.
     */
    public static TieredIndex open(Path dir, int dim, Metric metric, Config cfg, RecordSource host) {
        return open(dir, dim, metric, cfg, host, HostEnv.standalone());
    }

    /**
     * {@link #open(Path, int, Metric, Config, RecordSource)} for a tier that runs on a host environment. A refused permit for the build of a cold
     * index, or for the rebuild that drops ids above the recovered maximum, makes the call throw {@link BuildRefusedException}.
     */
    public static TieredIndex open(Path dir, int dim, Metric metric, Config cfg, RecordSource host, HostEnv env) {
        Config c = cfg != null ? cfg : Config.defaults();
        final long hostEpoch = host.epoch();
        checkEpoch(hostEpoch);
        DirectoryLock claim = null;
        boolean kept = false;
        try {
            Throwable cause;
            try {
                claim = DirectoryLock.acquireIfPresent(dir);
                Manifest m = Manifest.readLatest(dir);
                if (m == null) {
                    cause = new NoSuchFileException(dir.toString(), null, "no manifest in the directory");
                } else if (m.epoch() != hostEpoch) {
                    cause = new SegmentFormat.MismatchException("the persisted state belongs to epoch " + (m.epoch() == Manifest.NONE ? "none" : Long.toString(m.epoch()))
                            + ", the host is at epoch " + hostEpoch);
                } else {
                    TieredIndex t = restore(dir, m, dim, metric, c, env);
                    try {
                        t.reconcile(dir, host, m.hwm());
                        t.hold(claim);
                        kept = true;
                        t.maintain();
                        return t;
                    } catch (IOException e) {
                        t.close();
                        cause = e;
                    } catch (RuntimeException e) {
                        t.close();
                        throw e;
                    }
                }
            } catch (IOException e) {
                cause = e;
            }
            TieredIndex t = new TieredIndex(dim, metric, c, env);
            try {
                t.rebuildFrom(host.allLive());
                t.epoch = hostEpoch;
            } catch (RuntimeException e) {
                t.close();
                throw e;
            }
            t.fallbackCause = cause;
            t.hold(claim);
            kept = true;
            return t;
        } finally {
            if (!kept && claim != null) claim.close();
        }
    }

    /** why {@link #open(Path, int, Metric, Config, Supplier)} rebuilt instead of reading files, or null when the segments came from files */
    public Throwable openFallbackCause() {
        return fallbackCause;
    }

    /** the largest id that any of the segments holds (hidden copies included), {@link Manifest#NONE} when they hold no row */
    private static long highWater(Segment[] segs) {
        long hw = Manifest.NONE;
        for (Segment s : segs) {
            if (s.n > 0) hw = Math.max(hw, s.ids[s.n - 1]);
        }
        return hw;
    }

    /** the largest id that the view holds, in a segment (hidden copies included) or in a tail, {@link Manifest#NONE} when it holds no row */
    private static long highWater(View v) {
        long hw = highWater(v.segs);
        for (long id : tailIds(v)) hw = Math.max(hw, id);
        return hw;
    }

    /** the ids that the tails of a view hold, the frozen one included */
    private static long[] tailIds(View v) {
        LongStream ids = v.active.raw.keySet().stream().mapToLong(Long::longValue);
        if (v.frozen != null) ids = LongStream.concat(ids, v.frozen.raw.keySet().stream().mapToLong(Long::longValue));
        return ids.toArray();
    }

    /**
     * under mu: the rows of the tails of a view, the active one and a frozen one that a failed seal left behind (an id is in one of them at
     * most), in the order of their versions, which is the order in which they arrived. The vectors are the tier's own arrays, which nothing
     * changes.
     */
    private static TailFile.Rows tailRows(View v) {
        long[] ids = tailIds(v);
        int n = ids.length;
        if (n == 0) return TailFile.Rows.NONE;
        Rec[] recs = new Rec[n];
        long[] vers = new long[n];
        for (int i = 0; i < n; i++) {
            Rec r = v.active.raw.get(ids[i]);
            recs[i] = r != null ? r : v.frozen.raw.get(ids[i]);
            vers[i] = recs[i].ver;
        }
        int[] ord = orderByVersion(vers, n);
        float[][] vecs = new float[n][];
        for (int i = 0; i < n; i++) vecs[i] = recs[ord[i]].v;
        return new TailFile.Rows(permute(ids, ord), permute(vers, ord), vecs);
    }

    /** steps 5 to 7 of the replay protocol on a tier that {@link #restore} just made and that no other thread holds: see {@link #open(Path, int, Metric, Config, RecordSource)} */
    private void reconcile(Path dir, RecordSource host, long persistedHighWater) throws IOException {
        final long recovered = host.recoveredMax();
        boolean dropped = false;
        long above = persistedHighWater; // the high-water id of the manifest, not of the rows in memory: the worker may already have compacted hidden rows away
        work.lock(); // the worker neither seals nor merges while the held ids are looked at and the drop is published
        try {
            // the tail first: a row only ever moves from the tail into a segment, so a row that is not seen here is seen in the segments
            for (long id : tailIds(view)) {
                if (id > recovered) {
                    delete(id);
                    dropped = true;
                } else if (!host.isLive(id)) {
                    delete(id);
                }
            }
            for (Segment s : view.segs) {
                for (int i = 0; i < s.n; i++) {
                    long id = s.ids[i];
                    if (id > recovered) { // a hidden row counts too: it must not stay in a segment under an id that the host may mint again
                        delete(id);
                        dropped = true;
                    } else if (!s.isHidden(s.g.sortedRows[i]) && !host.isLive(id)) {
                        delete(id);
                    }
                }
            }
            if (dropped) {
                // a segment that holds a dropped row is rebuilt: the row (and the rows the host deleted) is not copied into the new segment.
                // A drop from the tail alone builds nothing
                if (highWater(view.segs) > recovered) mergeNow();
                checkpoint(dir); // the drop is durable before the index is ready; the high-water id of the new generation is the largest id that is left
                above = highWater(view);
            }
        } finally {
            work.unlock();
        }
        List<IdVector> chunk = new ArrayList<>();
        for (IdVector iv : host.liveAbove(above)) {
            chunk.add(iv);
            if (chunk.size() == 1024) {
                apply(new Batch(chunk, List.of()));
                chunk.clear();
            }
        }
        if (!chunk.isEmpty()) apply(new Batch(chunk, List.of()));
    }

    private static TieredIndex restore(Path dir, Manifest m, int dim, Metric metric, Config cfg, HostEnv env) throws IOException {
        Manifest.Fingerprint fp = m.fingerprint();
        GraphIndex.Params gp = cfg.graph();
        if (fp.dim() != dim || fp.metric() != metric || fp.M() != gp.M() || fp.efConstruction() != gp.efConstruction() || fp.sq8() != gp.sq8()) {
            throw new SegmentFormat.MismatchException("the persisted segments were built for " + fp + ", not for dim " + dim + " " + metric + " M "
                    + gp.M() + " efConstruction " + gp.efConstruction() + " sq8 " + gp.sq8());
        }
        List<Segment> segs = new ArrayList<>();
        long[][] hidden = Manifest.readTombstones(dir, m);
        // the rows of the tail, with their checksum compared before any of them is used
        TailFile.Rows tail = m.tailRows() == 0 ? TailFile.Rows.NONE : TailFile.read(dir.resolve(Manifest.tailName(m.generation())), m);
        Map<Long, Long> hiddenBelow = new HashMap<>(); // per id with a marked row: the smallest version above every marked copy
        long total = 0;
        long hiddenTotal = 0;
        long maxVer = 0;
        for (Manifest.Entry e : m.entries()) {
            SegmentReader.Loaded l = SegmentReader.readFull(dir.resolve(Manifest.segName(e.segId())), true, cfg.vectorStorage());
            SegmentInfo si = l.info();
            if (l.vers() == null) throw new SegmentFormat.CorruptException("segment " + e.segId() + " has no version section", "VERS");
            if (si.n() != e.n() || si.fileLength() != e.fileLength() || si.headerCrc() != e.headerCrc()) {
                throw new SegmentFormat.CorruptException("segment " + e.segId() + " changed since the manifest was read", "MANIFEST");
            }
            Segment s = new Segment(l.graph(), l.vers());
            s.fileId = e.segId();
            long[] bits = hidden[segs.size()];
            for (int row = 0; row < s.n; row++) {
                if ((bits[row >>> 6] & (1L << row)) != 0L) {
                    hiddenBelow.merge(s.g.ids[row], l.vers()[row] + 1, Math::max);
                    s.hide(row, RESTORED);
                }
            }
            hiddenTotal += s.dead;
            segs.add(s);
            total += s.n;
            maxVer = Math.max(maxVer, si.verMax());
        }
        if (maxVer > m.versionCounter()) throw new SegmentFormat.CorruptException("a segment holds version " + maxVer + " above the manifest's counter " + m.versionCounter(), "MANIFEST");
        checkHidden(segs, hidden, hiddenBelow);
        checkTail(segs, hidden, tail, m);
        int live = Math.toIntExact(total - hiddenTotal + tail.size());
        Manifest.gc(dir, m); // orphans: files of a crashed publish, older generations
        TieredIndex t = new TieredIndex(dim, metric, cfg, env);
        t.work.lock();
        t.mu.lock();
        try {
            // the rows of the tail go back into the tail, in the order in which they arrived and with their versions
            for (int i = 0; i < tail.size(); i++) t.view.active.put(tail.ids()[i], new Rec(tail.vecs()[i], tail.vers()[i]));
            t.counter = m.versionCounter();
            t.publish(segs.toArray(new Segment[0]), null, t.view.active);
            t.epoch = m.epoch();
            t.live.set(live);
            t.persistedDir = dir;
            t.persistedManifest = m;
        } finally {
            t.mu.unlock();
            t.work.unlock();
        }
        return t;
    }

    /** under mu: for each segment the bitmap of its hidden rows, in the layout of a tombstone file */
    private static long[][] hiddenBitmaps(Segment[] segs) {
        long[][] bits = new long[segs.length][];
        for (int i = 0; i < segs.length; i++) {
            Segment s = segs[i];
            bits[i] = new long[(s.n + 63) >>> 6];
            for (int j = 0; j < s.dead; j++) {
                int row = s.hiddenRow(j);
                bits[i][row >>> 6] |= 1L << row;
            }
        }
        return bits;
    }

    /**
     * A tier hides only older copies of an id: a row that the tombstone file leaves visible is newer than every marked row of its id. A file
     * that says otherwise was not written by a tier and is refused.
     */
    private static void checkHidden(List<Segment> segs, long[][] hidden, Map<Long, Long> hiddenBelow) throws SegmentFormat.CorruptException {
        for (Map.Entry<Long, Long> e : hiddenBelow.entrySet()) {
            for (int i = 0; i < segs.size(); i++) {
                Segment s = segs.get(i);
                int ix = s.indexOf(e.getKey());
                if (ix < 0 || s.vers[ix] >= e.getValue()) continue;
                int row = s.g.sortedRows[ix];
                if ((hidden[i][row >>> 6] & (1L << row)) == 0L) {
                    throw new SegmentFormat.CorruptException("the tombstone file hides a newer copy of id " + e.getKey() + " and leaves an older one visible", "TOMBSTONES");
                }
            }
        }
    }

    /**
     * A tail row is the only visible copy of its id and the newest one: the tail holds an id once, every copy of that id in a segment is
     * older and marked in the tombstone file, and inside an epoch no id is above the high-water id. A tail that says otherwise was not written
     * by a tier and is refused.
     */
    private static void checkTail(List<Segment> segs, long[][] hidden, TailFile.Rows tail, Manifest m) throws SegmentFormat.CorruptException {
        Set<Long> seen = new HashSet<>();
        for (int r = 0; r < tail.size(); r++) {
            long id = tail.ids()[r];
            if (!seen.add(id)) throw new SegmentFormat.CorruptException("the tail holds id " + id + " twice", "TAIL");
            if (m.epoch() != Manifest.NONE && id > m.hwm()) {
                throw new SegmentFormat.CorruptException("the tail holds id " + id + " above the high-water id " + m.hwm(), "TAIL");
            }
            for (int i = 0; i < segs.size(); i++) {
                Segment s = segs.get(i);
                int ix = s.indexOf(id);
                if (ix < 0) continue;
                int row = s.g.sortedRows[ix];
                if ((hidden[i][row >>> 6] & (1L << row)) == 0L || s.vers[ix] >= tail.vers()[r]) {
                    throw new SegmentFormat.CorruptException("a segment holds a visible or a newer copy of id " + id + ", which the tail holds", "TAIL");
                }
            }
        }
    }

    // ---- reports ----

    @Override
    public long bytesHeld() {
        View v = view;
        long b = 512;
        b += tailBytes(v.active);
        if (v.frozen != null) b += tailBytes(v.frozen);
        for (Segment s : v.segs) b += s.bytes();
        return b;
    }

    private long tailBytes(Tail t) {
        return t.flat.bytesHeld() + (long) t.raw.size() * ((long) dim * 4 + 16 + 24 + 64);
    }

    public Stats stats() {
        View v = view;
        final long a = addsCount, sv = sealedVectors.get(), mv = mergedVectors.get();
        final int tc = taskCount; // published task slots only
        return new Stats(seals.get(), merges.get(), pendingTail(), v.segs.length, bytesHeld(), sealNanos.get() / 1_000_000L,
                mergeNanos.get() / 1_000_000L, hiddenRows(v.segs), failures.get(), live.get(), backpressure.get(), sealedVectors.get(),
                mergedVectors.get(), sizesOf(v), a, a > 0 ? (double) mv / a : 0.0, a > 0 ? (double) (sv + mv) / a : 0.0,
                maxSealNanos.get() / 1_000_000L, maxMergeNanos.get() / 1_000_000L, Arrays.copyOf(taskKind, tc), Arrays.copyOf(taskStart, tc),
                Arrays.copyOf(taskEnd, tc), Arrays.copyOf(taskRows, tc), taskDropped.get(), bigMerges.get(), bigMergedVectors.get());
    }

    /** rows of each segment of the current view, oldest first (a fresh array; the segments are immutable) */
    public int[] segmentSizes() {
        return sizesOf(view);
    }

    private static int[] sizesOf(View v) {
        int[] r = new int[v.segs.length];
        for (int i = 0; i < r.length; i++) r[i] = v.segs[i].n;
        return r;
    }

    /**
     * Closes the tier: later calls that read or change vectors or files throw {@link ClosedException}, a build that runs stops, and the lock of
     * the directory is released. A standalone tier stops its worker thread. A hosted tier waits (up to 30 s) for a build that holds the work
     * lock, and a task that the executor has not started yet does nothing when it starts. A second call is harmless.
     */
    @Override
    public void close() {
        closed = true;
        try {
            if (hosted) {
                if (work.tryLock(30, TimeUnit.SECONDS)) work.unlock();
            } else {
                wake.release();
                worker.interrupt();
                worker.join(30_000L);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        mu.lock(); // after a checkpoint that is running: it publishes under the lock of the directory
        try {
            if (dirLock != null) dirLock.close();
            dirLock = null;
        } finally {
            mu.unlock();
        }
    }
}
