package com.humemai.vec;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.LockSupport;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.LongPredicate;
import java.util.function.Supplier;

/**
 * Tiered vector index for agent memory: a bounded exact TAIL in front of immutable graph SEGMENTS (the LSM shape of the 2026-11-07
 * vector program, requirements R1 to R10).
 *
 * <p>Source of truth. The vectors in the records are the truth; the segments (graphs) and the floor map are DERIVED data. After a crash
 * the engine would re-read its records and call {@link #rebuildFrom(Iterable)}: everything (tail, segments, floor) is reconstructed
 * from raw vectors and nothing else. In the same way a segment or the whole tier can always be thrown away and rebuilt from the tail
 * plus the records, so the code never has to make the derived state durable. Porting to LSM_VECTOR: the tail is the engine's delta
 * list (its exact scan stays), a segment is one sealed graph file written beside the records, the floor map is rebuilt from the
 * record versions, and the seal/merge worker is the engine's background rebuild with a bounded queue.
 *
 * <p>Writes. {@link #add} puts the vector into the active tail (a {@link FlatIndex}, exact fp32 or SQ8) and a raw copy next to it
 * (the raw copy is what a seal and a rebuild read). It is visible to the next search by any thread before add returns.
 * {@link #delete} removes it from the tails and hides every segment copy at once.
 *
 * <p>Search = exact scan of the tail(s) merged with the approximate search of every segment, minus hidden ids, best score first,
 * the allow-list applied in both. Duplicate ids (a replace in flight) are resolved newest source first: active tail, frozen tail,
 * then segments. {@link Freshness#SEALED_ONLY} skips the tails: it may lag (a vector added or replaced since the last seal is not
 * in it, a replaced id is absent until its new version is sealed), and it never returns a deleted id, a stale vector or an id that
 * was never added. Below {@code smallLimit} live vectors no graph is ever built and every search is an exact scan, which removes the
 * rebuild on the query path of the engine today.
 *
 * <p>Seal. When the active tail reaches {@code tailLimit} vectors (or its first vector is older than {@code tailAgeMs}) and the
 * index holds at least {@code smallLimit} vectors, the worker thread (1) freezes the tail under the write lock (a new empty tail
 * takes over writes, the view now names a frozen tail and an active tail), (2) builds a GraphIndex from the frozen contents and
 * seals it, outside any lock, while searches and writes continue (searches scan both tails), (3) publishes the segment by one
 * volatile write of an immutable view that drops the frozen tail. A vector is always in the frozen tail, the segment, or both for an
 * instant, and the dedupe resolves the overlap, so it is never missing. A search that started before the swap finishes on the old
 * view, which still holds everything it needs.
 *
 * <p>Delete and replace. Versions come from one counter under the write lock. The floor map says "a segment copy of id is valid only
 * when its version is at least floor[id]"; replace and delete raise the floor (only when a segment or the frozen tail holds a copy of
 * the id, so the map stays small). Searches filter segments through it inside the beam, so a deleted vector never appears, with no
 * wait for a merge. Each segment counts its hidden copies ({@code dead}); a segment with no hidden copy is searched with the plain
 * allow-list.
 *
 * <p>Merge. When the segment count exceeds {@code maxSegments}, or one segment is {@code deadFraction} dead, the worker rebuilds the
 * smallest segments (or the dead one) into ONE segment from their valid vectors only, throttled to {@code mergeRatePerSec} inserts
 * per second so the cores that serve queries are not starved, and swaps it in atomically. Floor entries whose stale copies are gone
 * are dropped at the swap, except those that a search still running on an older view can need: a search counts itself on the view it
 * reads, a replaced view with readers is held, and the remaining entries are dropped once those searches are done (by the worker, or by
 * awaitIdle, mergeNow, sealNow or checkpoint on the calling thread).
 *
 * <p>Size-ratio trigger (fix7, {@code mergeTriggerRatio} X, default 0 = off and the code is then the old code). The segment with the most live rows is the BIG one. When the live rows of all the other
 * segments together exceed X times the live rows of the big one, ALL segments are merged into one (the small ones go into the big one). Otherwise, when the count exceeds {@code maxSegments} and
 * maxSegments is at least 2, the count rule picks among the SMALL segments only (the big one is never a candidate), so the small segments merge among themselves and the big one is rewritten only by the
 * ratio rule; with maxSegments 1 the count rule includes the big one (a merge at every seal). With a large maxSegments the small segments pile up until the ratio fires. A merge into the big segment
 * rewrites (1 + X) * rows(big) rows per X * rows(big) inserts, about (1 + X) / X vectors per insert, on top of the small-segment merges.
 *
 * <p>Bounds. One worker thread, one task at a time (seal or merge, a lock named work), a cap on segments, no allocation proportional
 * to n on the query path (per query: the hit lists and a hash table of size about twice the hits). Back pressure: when the active tail
 * holds {@code hardTailFactor * max(tailLimit, smallLimit)} vectors, writers wait (bounded, 10 s) for the worker.
 *
 * <p>Thread safety: add and delete take the write lock {@code mu} (short: one flat add, a few binary searches); searches take no
 * lock (each does one atomic add and one atomic subtract on the reader count of the view it reads). {@link #rebuildFrom} and {@link #sealNow} and {@link #mergeNow} take the work lock.
 *
 * <p>Persistence (first cut). A settled tier (empty tail, no hidden copies) can be written to a directory with {@link #checkpoint} and
 * read back with {@link #open}: segment files, a tombstone file and a manifest per generation, see docs/SEGMENT-FORMAT.md. The records
 * stay the truth; when {@code open} cannot read the directory, the host rebuilds from its records as above.
 */
public final class TieredIndex implements VecIndex {

    /** FRESH (default): tails included, read-your-writes. SEALED_ONLY: segments only, may lag. */
    public enum Freshness { FRESH, SEALED_ONLY }

    /**
     * Result of a search with its meta (what the facets report as exact/patched/lag).
     * @param exact true when no segment was searched and the tail scan was exact (the answer equals brute force)
     * @param tailScanned live tail vectors scanned (0 for SEALED_ONLY)
     * @param segments number of segments searched
     */
    public record SearchResult(Hit[] hits, boolean exact, int tailScanned, int segments) {}

    /** one raw vector with its id, the unit {@link #rebuildFrom} reads (the engine would stream its records) */
    public record IdVector(long id, float[] v) {}

    /**
     * Parameters. tailLimit: seal when the active tail has this many vectors (default 2000). tailAgeMs: also seal when the oldest tail
     * vector is older than this (0 = never by age). smallLimit: below this many live vectors never build a graph (default 1000).
     * maxSegments: merge when exceeded (default 8). mergeFan: smallest segments merged at once (default 4). deadFraction: compact a
     * segment when this share of it is hidden (default 0.3). mergeRatePerSec: merge builder insert rate cap, 0 = unlimited.
     * hardTailFactor: back pressure threshold. tailQuant: NONE (exact fp32) or SQ8. graph: segment graph parameters (efSearch is the
     * search beam). buildThreads: threads of one seal or merge build. splitBeam (fix5, 2026-11-07, default false = the old behaviour, the
     * full beam in every segment): when true and more than one segment is searched, segment s gets the beam max(k, ceil(ef * rows(s) / rows of
     * all searched segments)) instead of ef. tailQuant is the tail kernel: NONE scans fp32, SQ8 scans int8 codes (FlatIndex default kernel
     * F8X4) and reranks the best 5 * k candidates in fp32. mergeTriggerRatio (fix7, 2026-11-07, default 0 = off, the old behaviour bit for bit): the size-ratio merge trigger. When it is above 0 and
     * the live rows of all segments except the largest exceed mergeTriggerRatio times the live rows of the largest, the worker merges ALL segments into one (the small ones go into the big one); otherwise
     * the count rule (maxSegments, mergeFan) applies to the small segments only (the largest is not a candidate) when maxSegments is at least 2, and to all segments when it is 1 (merge at every seal).
     */
    public record Config(int tailLimit, long tailAgeMs, int smallLimit, int maxSegments, int mergeFan, double deadFraction,
                         int mergeRatePerSec, int hardTailFactor, Quant tailQuant, GraphIndex.Params graph, int buildThreads, boolean splitBeam, double mergeTriggerRatio) {
        /** the 12-component form from before fix7: mergeTriggerRatio 0, the old behaviour (existing callers and tests keep compiling) */
        public Config(int tailLimit, long tailAgeMs, int smallLimit, int maxSegments, int mergeFan, double deadFraction,
                      int mergeRatePerSec, int hardTailFactor, Quant tailQuant, GraphIndex.Params graph, int buildThreads, boolean splitBeam) {
            this(tailLimit, tailAgeMs, smallLimit, maxSegments, mergeFan, deadFraction, mergeRatePerSec, hardTailFactor, tailQuant, graph, buildThreads, splitBeam, 0.0);
        }

        /** the 11-component form from before fix5: splitBeam false, the old behaviour (existing callers and tests keep compiling) */
        public Config(int tailLimit, long tailAgeMs, int smallLimit, int maxSegments, int mergeFan, double deadFraction,
                      int mergeRatePerSec, int hardTailFactor, Quant tailQuant, GraphIndex.Params graph, int buildThreads) {
            this(tailLimit, tailAgeMs, smallLimit, maxSegments, mergeFan, deadFraction, mergeRatePerSec, hardTailFactor, tailQuant, graph, buildThreads, false, 0.0);
        }

        public Config {
            if (tailLimit < 1 || smallLimit < 0 || maxSegments < 1 || mergeFan < 2 || hardTailFactor < 2 || buildThreads < 1 || !(mergeTriggerRatio >= 0.0) || Double.isInfinite(mergeTriggerRatio))
                throw new IllegalArgumentException("bad config");
            if (tailQuant == Quant.BINARY) throw new IllegalArgumentException("tail quant must be NONE or SQ8");
        }

        public static Config defaults() {
            return new Config(2000, 60_000L, 1000, 8, 4, 0.3, 50_000, 4, Quant.NONE, GraphIndex.Params.defaults(),
                    Math.max(1, Math.min(4, Runtime.getRuntime().availableProcessors() - 1)), false, 0.0);
        }

        public Config withTail(int tailLimit, int smallLimit) {
            return new Config(tailLimit, tailAgeMs, smallLimit, maxSegments, mergeFan, deadFraction, mergeRatePerSec, hardTailFactor, tailQuant, graph, buildThreads, splitBeam, mergeTriggerRatio);
        }

        public Config withMaxSegments(int m) {
            return new Config(tailLimit, tailAgeMs, smallLimit, m, mergeFan, deadFraction, mergeRatePerSec, hardTailFactor, tailQuant, graph, buildThreads, splitBeam, mergeTriggerRatio);
        }

        public Config withTailQuant(Quant q) {
            return new Config(tailLimit, tailAgeMs, smallLimit, maxSegments, mergeFan, deadFraction, mergeRatePerSec, hardTailFactor, q, graph, buildThreads, splitBeam, mergeTriggerRatio);
        }

        public Config withGraph(GraphIndex.Params p) {
            return new Config(tailLimit, tailAgeMs, smallLimit, maxSegments, mergeFan, deadFraction, mergeRatePerSec, hardTailFactor, tailQuant, p, buildThreads, splitBeam, mergeTriggerRatio);
        }

        public Config withBuildThreads(int t) {
            return new Config(tailLimit, tailAgeMs, smallLimit, maxSegments, mergeFan, deadFraction, mergeRatePerSec, hardTailFactor, tailQuant, graph, t, splitBeam, mergeTriggerRatio);
        }

        public Config withMergeRate(int perSec) {
            return new Config(tailLimit, tailAgeMs, smallLimit, maxSegments, mergeFan, deadFraction, perSec, hardTailFactor, tailQuant, graph, buildThreads, splitBeam, mergeTriggerRatio);
        }

        public Config withMergeFan(int f) {
            return new Config(tailLimit, tailAgeMs, smallLimit, maxSegments, f, deadFraction, mergeRatePerSec, hardTailFactor, tailQuant, graph, buildThreads, splitBeam, mergeTriggerRatio);
        }

        public Config withSplitBeam(boolean sb) {
            return new Config(tailLimit, tailAgeMs, smallLimit, maxSegments, mergeFan, deadFraction, mergeRatePerSec, hardTailFactor, tailQuant, graph, buildThreads, sb, mergeTriggerRatio);
        }

        public Config withMergeTriggerRatio(double r) {
            return new Config(tailLimit, tailAgeMs, smallLimit, maxSegments, mergeFan, deadFraction, mergeRatePerSec, hardTailFactor, tailQuant, graph, buildThreads, splitBeam, r);
        }
    }

    /** counters for the report. segmentSizes (fix3, 2026-11-07): rows of each segment, oldest first, hidden copies included; its length is segments. fix5 (2026-11-07): adds (calls of add), mergeWriteAmp (mergedVectors / adds, 0 when adds is 0),
     *  totalWriteAmp ((sealedVectors + mergedVectors) / adds), maxSealMillis and maxMergeMillis (the longest single seal or merge).
     *  fix6 (2026-11-07): the task log, one slot per finished seal or merge in finish order: taskKinds (1 seal, 2 merge), taskStartNs and taskEndNs (System.nanoTime(), the harness clock; end is the instant after the swap),
     *  taskRows (rows built); taskDropped counts tasks that did not fit the 4096 slots.
     *  fix7 (2026-11-07): bigMerges and bigMergedVectors count the merges that rewrote the segment with the most live rows at pick time (a forced mergeNow included) and the rows they built; both are parts of merges and mergedVectors */
    public record Stats(long seals, long merges, int tailSize, int segments, long bytesHeld, long sealMillis, long mergeMillis,
                        int floorEntries, long failures, int live, long backpressureWaits, long sealedVectors, long mergedVectors,
                        int[] segmentSizes, long adds, double mergeWriteAmp, double totalWriteAmp, long maxSealMillis, long maxMergeMillis,
                        int[] taskKinds, long[] taskStartNs, long[] taskEndNs, int[] taskRows, long taskDropped, long bigMerges, long bigMergedVectors) {}

    private static final Hit[] NONE = new Hit[0];
    private static final int FLAT_RERANK = 5; // the default of FlatIndex for SQ8

    // ---- state ----

    private static final class Rec {
        final float[] v;
        final long ver;

        Rec(float[] v, long ver) {
            this.v = v;
            this.ver = ver;
        }
    }

    /** a bounded tail: the flat index scanned by searches and the raw copies a seal reads. Mutated only under mu. */
    private static final class Tail {
        final FlatIndex flat;
        final ConcurrentHashMap<Long, Rec> raw = new ConcurrentHashMap<>();
        volatile long firstNs;

        Tail(int dim, Metric metric, Quant q) {
            flat = new FlatIndex(dim, metric, q);
        }

        void put(long id, Rec r) {
            if (raw.isEmpty()) firstNs = System.nanoTime();
            flat.add(id, r.v);
            raw.put(id, r);
        }

        void remove(long id) {
            raw.remove(id);
            flat.delete(id);
        }
    }

    private static final class Segment {
        final SealedGraph g;
        final int n;
        final long[] ids;  // sorted
        final long[] vers; // aligned with ids
        volatile int dead; // copies hidden by the floor map; written under mu
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

        long bytes() {
            return g.bytesHeld() + 16L * n + 64;
        }
    }

    /** immutable: replaced as a whole by one volatile write; only the reader count changes */
    private static final class View {
        final Segment[] segs;
        final Tail frozen; // being sealed, or null
        final Tail active;
        /** searches that are running on this view (see {@code enter}); a floor entry that a segment of the view needs outlives them */
        final AtomicInteger readers = new AtomicInteger();

        View(Segment[] segs, Tail frozen, Tail active) {
            this.segs = segs;
            this.frozen = frozen;
            this.active = active;
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
    private final Config cfg;
    private final int hardTail;
    private final ReentrantLock mu = new ReentrantLock();
    private final ReentrantLock work = new ReentrantLock();
    private volatile View view;
    /** replaced views that a search may still be scanning (their reader count was not zero at the swap); guarded by mu */
    private final List<View> held = new ArrayList<>();
    /** a merge swap could not clean the floor map completely (a search holds an old view, or a seal is running): the worker finishes it */
    private volatile boolean floorDue;
    private final ConcurrentHashMap<Long, Long> floor = new ConcurrentHashMap<>();
    private long counter; // guarded by mu
    private final AtomicInteger live = new AtomicInteger();
    private final Semaphore wake = new Semaphore(0);
    private volatile boolean closed;
    private volatile int activity; // 0 idle, 1 sealing, 2 merging
    private volatile Throwable lastError;
    private Path persistedDir;           // the directory of the last checkpoint or open, guarded by mu
    private Manifest persistedManifest;  // the generation that directory holds, guarded by mu
    private volatile Throwable fallbackCause; // why open fell back to a rebuild, null when the segments were read from files
    private volatile Runnable hookBeforePublish;
    private volatile Runnable hookAfterPublish;
    private final AtomicLong seals = new AtomicLong(), merges = new AtomicLong(), sealNanos = new AtomicLong(), mergeNanos = new AtomicLong();
    private final AtomicLong failures = new AtomicLong(), backpressure = new AtomicLong(), sealedVectors = new AtomicLong(), mergedVectors = new AtomicLong();
    private final AtomicLong maxSealNanos = new AtomicLong(), maxMergeNanos = new AtomicLong();
    private final AtomicLong bigMerges = new AtomicLong(), bigMergedVectors = new AtomicLong(); // fix7: merges that rewrote the largest segment, and the rows they built
    private volatile long addsCount; // add() calls (fix5); incremented under mu
    private final Thread worker;

    // fix6 (2026-11-07, README "Tier p99 under heavy inserts"): the seal and merge intervals as System.nanoTime() stamps, the clock the harness uses for its own timestamps. doSeal and doMerge always run
    // under the work lock (the worker loop, sealNow, mergeNow), so there is one writer; stats() reads the slots below taskCount, which is written after the slot (volatile).
    private static final int TASK_LOG_CAP = 4096;
    private final int[] taskKind = new int[TASK_LOG_CAP];
    private final int[] taskRows = new int[TASK_LOG_CAP];
    private final long[] taskStart = new long[TASK_LOG_CAP];
    private final long[] taskEnd = new long[TASK_LOG_CAP];
    private volatile int taskCount;
    private final AtomicLong taskDropped = new AtomicLong();
    /** fix6: seal builds are throttled to this many inserts per second (0 = unthrottled, the old behaviour); a harness knob set by setSealRatePerSec, not a Config component */
    private volatile int sealRate;

    /** fix6: one finished task (kind 1 seal, 2 merge) into the log; the caller holds the work lock */
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
        if (dim < 1) throw new IllegalArgumentException("dim");
        this.dim = dim;
        this.metric = metric;
        this.cfg = cfg;
        this.hardTail = (int) Math.min(Integer.MAX_VALUE / 2, (long) cfg.hardTailFactor() * Math.max(cfg.tailLimit(), cfg.smallLimit()));
        this.view = new View(new Segment[0], null, new Tail(dim, metric, cfg.tailQuant()));
        this.worker = new Thread(this::workerLoop, "tiered-worker");
        this.worker.setDaemon(true);
        this.worker.start();
    }

    @Override public int dim() { return dim; }
    @Override public Metric metric() { return metric; }
    @Override public int size() { return live.get(); }
    public Config config() { return cfg; }
    /** 0 idle, 1 a seal is running, 2 a merge is running (for per-phase latency reports) */
    public int activity() { return activity; }
    public Throwable lastError() { return lastError; }
    public boolean workerAlive() { return worker.isAlive(); }
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

    // ---- writes ----

    private record Probe(boolean inActive, boolean inFrozen, Segment valid, boolean phys) {}

    /** where id lives right now; call under mu */
    private Probe probe(View v, long id) {
        boolean a = v.active.raw.containsKey(id);
        boolean f = v.frozen != null && v.frozen.raw.containsKey(id);
        Segment valid = null;
        boolean phys = false;
        if (v.segs.length > 0) {
            Long fl = floor.get(id);
            long floorV = fl == null ? Long.MIN_VALUE : fl;
            for (Segment s : v.segs) {
                int ix = s.indexOf(id);
                if (ix >= 0) {
                    phys = true;
                    if (s.vers[ix] >= floorV) valid = s;
                }
            }
        }
        return new Probe(a, f, valid, phys);
    }

    @Override
    public void add(long id, float[] v) {
        if (v.length != dim) throw new IllegalArgumentException("dim " + v.length + " != " + dim);
        float[] copy = v.clone();
        awaitBackpressure();
        boolean wakeUp;
        mu.lock();
        try {
            long ver = ++counter;
            addsCount++;
            View vw = view;
            Probe p = probe(vw, id);
            vw.active.put(id, new Rec(copy, ver));            // 1. the new version becomes visible
            if (p.phys() || p.inFrozen()) floor.put(id, ver); // 2. older copies in segments (or in a frozen tail that is being sealed) stop counting
            if (p.valid() != null) p.valid().dead = p.valid().dead + 1;
            if (p.inFrozen()) vw.frozen.remove(id);           // 3. the old frozen copy goes
            if (!(p.inActive() || p.inFrozen() || p.valid() != null)) live.incrementAndGet();
            wakeUp = vw.active.raw.size() >= cfg.tailLimit();
        } finally {
            mu.unlock();
        }
        if (wakeUp && wake.availablePermits() == 0) wake.release();
    }

    @Override
    public boolean delete(long id) {
        mu.lock();
        try {
            View vw = view;
            Probe p = probe(vw, id);
            if (!(p.inActive() || p.inFrozen() || p.valid() != null)) return false;
            if (p.inActive()) vw.active.remove(id);
            if (p.inFrozen()) vw.frozen.remove(id);
            if (p.phys() || p.inFrozen()) floor.put(id, ++counter);
            if (p.valid() != null) p.valid().dead = p.valid().dead + 1;
            live.decrementAndGet();
            return true;
        } finally {
            mu.unlock();
        }
    }

    private void awaitBackpressure() {
        if (view.active.raw.size() < hardTail) return;
        long deadline = System.nanoTime() + 10_000_000_000L;
        while (view.active.raw.size() >= hardTail && !closed && worker.isAlive() && System.nanoTime() < deadline) {
            backpressure.incrementAndGet();
            if (wake.availablePermits() == 0) wake.release();
            LockSupport.parkNanos(200_000L);
        }
    }

    // ---- search ----

    @Override
    public Hit[] search(float[] q, int k, LongPredicate allow) {
        return search(q, k, allow, Freshness.FRESH).hits();
    }

    public SearchResult search(float[] q, int k, LongPredicate allow, Freshness freshness) {
        return searchImpl(q, k, allow, freshness, cfg.graph().efSearch());
    }

    /** FRESH search with an explicit beam width for the segments (the harness sweeps ef through this overload) */
    public Hit[] search(float[] q, int k, LongPredicate allow, int ef) {
        return searchImpl(q, k, allow, Freshness.FRESH, ef).hits();
    }

    private SearchResult searchImpl(float[] q, int k, LongPredicate allow, Freshness freshness, int efSearch) {
        if (q.length != dim) throw new IllegalArgumentException("dim " + q.length + " != " + dim);
        if (k <= 0) return new SearchResult(NONE, true, 0, 0);
        View vw = enter();
        try {
            return searchView(vw, q, k, allow, freshness, efSearch);
        } finally {
            vw.readers.decrementAndGet();
        }
    }

    /**
     * The view a search scans, counted as read until the search leaves it. The count is raised first and the view is read again: when
     * the view is still current, a swap has not happened yet, so the swap sees the count (it reads it after its volatile write); when
     * it is not, the count is dropped and the search tries the new view. No lock, no thread, one atomic add each way.
     */
    private View enter() {
        for (;;) {
            View vw = view;
            vw.readers.incrementAndGet();
            if (view == vw) return vw;
            vw.readers.decrementAndGet();
        }
    }

    private SearchResult searchView(View vw, float[] q, int k, LongPredicate allow, Freshness freshness, int efSearch) {
        Segment[] segs = vw.segs;
        Hit[][] lists = new Hit[2 + segs.length][];
        int li = 0;
        int tailScanned = 0;
        if (freshness == Freshness.FRESH) {
            // an empty tail costs nothing: no scan, no list, no merge work (an add that returned has already put its raw copy)
            int na = vw.active.raw.size();
            if (na > 0) {
                tailScanned += na;
                Hit[] h = vw.active.flat.search(q, k, allow);
                if (h.length > 0) lists[li++] = h;
            }
            if (vw.frozen != null) {
                int nf = vw.frozen.raw.size();
                if (nf > 0) {
                    tailScanned += nf;
                    Hit[] h = vw.frozen.flat.search(q, k, allow);
                    if (h.length > 0) lists[li++] = h;
                }
            }
        }
        int searched = 0;
        int ef = Math.max(k, efSearch);
        long sizeSum = 0; // fix5 splitBeam only: rows of the segments that will be searched; 0 (the default) keeps the full beam ef in every segment
        if (cfg.splitBeam() && segs.length > 1) {
            for (Segment t : segs) if (t.n > 0 && t.dead < t.n) sizeSum += t.n;
        }
        for (int i = segs.length - 1; i >= 0; i--) { // newest segment first
            Segment s = segs[i];
            if (s.n == 0 || s.dead >= s.n) continue;
            LongPredicate p = s.dead > 0 ? segFilter(s, allow) : allow;
            int efS = sizeSum > 0 ? Math.max(k, (int) ((ef * (long) s.n + sizeSum - 1) / sizeSum)) : ef; // ceil(ef * rows(s) / sizeSum), at least k
            Hit[] h = s.g.search(q, k, p, efS);
            if (h.length > 0) lists[li++] = h;
            searched++;
        }
        boolean exact = searched == 0 && (cfg.tailQuant() == Quant.NONE || (long) FLAT_RERANK * k >= tailScanned);
        return new SearchResult(mergeTopK(lists, li, k), exact, tailScanned, searched);
    }

    private LongPredicate segFilter(Segment s, LongPredicate allow) {
        return id -> {
            if (allow != null && !allow.test(id)) return false;
            Long f = floor.get(id);
            if (f == null) return true;
            int ix = s.indexOf(id);
            return ix >= 0 && s.vers[ix] >= f;
        };
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

    /**
     * under mu: the one place a new view replaces the current one. A search that holds the replaced view may still scan its segments, so
     * the view is kept in {@code held} while its reader count is not zero. The count is read after the volatile write: a search that
     * entered the view before the write is counted, one that comes later reads the new view (see {@link #enter}).
     */
    private void publish(View nv) {
        View old = view;
        view = nv;
        if (old.readers.get() != 0) held.add(old);
    }

    /**
     * under mu: a merge swap, or the worker later, removes what no search can need. A hidden copy in a segment of the current view or of
     * a held view keeps its floor entry; the others go. While a seal is running (a frozen tail) or a held view still has readers, the
     * work is not complete and {@code floorDue} stays set for the worker.
     */
    private void settleFloor() {
        held.removeIf(h -> h.readers.get() == 0); // no search is in it, and none can enter it again
        View v = view;
        if (v.frozen == null) cleanupFloor(v.segs);
        floorDue = v.frozen != null || !held.isEmpty();
    }

    /** finish a deferred floor cleanup on the calling thread (a no-op when none is pending): the calls that promise a settled tier use it */
    private void settleFloorIfDue() {
        if (!floorDue) return;
        mu.lock();
        try {
            settleFloor();
        } finally {
            mu.unlock();
        }
    }

    // ---- worker: seal and merge, one at a time ----

    private void workerLoop() {
        while (!closed) {
            try {
                wake.tryAcquire(floorDue ? 5 : 50, TimeUnit.MILLISECONDS); // a pending floor cleanup waits for a search to leave an old view: look again soon
                wake.drainPermits();
                settleFloorIfDue();
                while (!closed) {
                    boolean did = false;
                    work.lock();
                    try {
                        if (closed) break;
                        if (sealDue()) did = doSeal(false);
                        else if (pickMerge(view.segs, false) != null) did = doMerge(false);
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
                failures.incrementAndGet();
                LockSupport.parkNanos(200_000_000L); // do not spin on a failing task
            }
        }
    }

    private boolean sealDue() {
        View vw = view;
        if (vw.frozen != null) return true; // a failed seal is retried
        int n = vw.active.raw.size();
        if (n == 0 || live.get() < cfg.smallLimit()) return false;
        if (n >= cfg.tailLimit()) return true;
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
        // fix7: the size-ratio trigger; mergeTriggerRatio 0 (the default) skips this block, and cand below is segs itself, so the rest is the old code
        if (cfg.mergeTriggerRatio() > 0.0 && ns >= 2) {
            int big = largestIndex(segs);
            long bigLive = segs[big].n - segs[big].dead, smallLive = 0;
            for (int i = 0; i < ns; i++) if (i != big) smallLive += segs[i].n - segs[i].dead;
            if ((double) smallLive > cfg.mergeTriggerRatio() * (double) bigLive) return new ArrayList<>(Arrays.asList(segs));
        }
        if (ns > cfg.maxSegments()) {
            Segment[] cand = segs;
            if (cfg.mergeTriggerRatio() > 0.0 && cfg.maxSegments() >= 2) { // fix7: the small segments merge among themselves; the big one only through the ratio rule above
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

    /** fix7: index of the segment with the most live rows (n - dead); ties go to the oldest (lowest index) */
    private static int largestIndex(Segment[] segs) {
        int best = 0;
        for (int i = 1; i < segs.length; i++) if (segs[i].n - segs[i].dead > segs[best].n - segs[best].dead) best = i;
        return best;
    }

    private boolean doSeal(boolean ignoreSmall) {
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
                    publish(new View(v.segs, v.active, new Tail(dim, metric, cfg.tailQuant())));
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
                    publish(new View(v.segs, null, v.active));
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
                countDead(seg);
                Segment[] ns = Arrays.copyOf(v.segs, v.segs.length + 1);
                ns[ns.length - 1] = seg;
                publish(new View(ns, null, v.active)); // the atomic swap
            } finally {
                mu.unlock();
            }
            Runnable after = hookAfterPublish;
            if (after != null) after.run();
            seals.incrementAndGet();
            sealedVectors.addAndGet(n);
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

    private boolean doMerge(boolean force) {
        long t0 = System.nanoTime();
        View v0 = view;
        List<Segment> picked = pickMerge(v0.segs, force);
        if (picked == null || picked.isEmpty()) return false;
        final boolean intoLargest = picked.contains(v0.segs[largestIndex(v0.segs)]); // fix7: this merge rewrites the largest segment
        activity = 2;
        try {
            int total = 0;
            for (Segment s : picked) total += s.g.n;
            long[] ids = new long[total];
            long[] vers = new long[total];
            Segment[] srcSeg = new Segment[total];
            int[] srcRow = new int[total];
            int m = 0;
            for (Segment s : picked) {
                for (int r = 0; r < s.g.n; r++) {
                    long id = s.g.ids[r];
                    int ix = s.indexOf(id);
                    long ver = s.vers[ix];
                    Long f = floor.get(id);
                    if (f != null && ver < f) continue; // hidden copy: dropped here
                    ids[m] = id;
                    vers[m] = ver;
                    srcSeg[m] = s;
                    srcRow[m] = r;
                    m++;
                }
            }
            Segment merged = null;
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
                final int d = dim;
                SealedGraph sg = build(mi, m, j -> {
                    int o = frr[j] * d;
                    return Arrays.copyOfRange(fss[j].g.vec, o, o + d);
                }, true);
                if (sg.n != m) throw new IllegalStateException("merged rows " + sg.n + " != " + m);
                merged = new Segment(sg, mi, mv);
            }
            mu.lock();
            try {
                View v = view;
                List<Segment> keep = new ArrayList<>();
                for (Segment s : v.segs) if (!picked.contains(s)) keep.add(s);
                if (merged != null) {
                    countDead(merged);
                    keep.add(merged);
                }
                Segment[] ns = keep.toArray(new Segment[0]);
                publish(new View(ns, v.frozen, v.active)); // the atomic swap
                settleFloor();
            } finally {
                mu.unlock();
            }
            merges.incrementAndGet();
            mergedVectors.addAndGet(m);
            if (intoLargest) { bigMerges.incrementAndGet(); bigMergedVectors.addAndGet(m); }
            final long mergeEnd = System.nanoTime();
            final long mergeDt = mergeEnd - t0;
            logTask(2, t0, mergeEnd, m);
            mergeNanos.addAndGet(mergeDt);
            maxMergeNanos.accumulateAndGet(mergeDt, Math::max);
            return true;
        } finally {
            activity = 0;
        }
    }

    /** under mu: hidden copies of s are the rows whose version is below the floor of their id */
    private void countDead(Segment s) {
        int d = 0;
        for (Map.Entry<Long, Long> e : floor.entrySet()) {
            int ix = s.indexOf(e.getKey());
            if (ix >= 0 && s.vers[ix] < e.getValue()) d++;
        }
        s.dead = d;
    }

    /** under mu, with no frozen tail and no seal running: a floor entry is needed only while some segment of the current view, or of a view a search still holds, has a copy below it */
    private void cleanupFloor(Segment[] segs) {
        for (Map.Entry<Long, Long> e : floor.entrySet()) {
            boolean needed = holdsBelow(segs, e);
            for (int h = 0; !needed && h < held.size(); h++) needed = holdsBelow(held.get(h).segs, e);
            if (!needed) floor.remove(e.getKey());
        }
    }

    private static boolean holdsBelow(Segment[] segs, Map.Entry<Long, Long> e) {
        for (Segment s : segs) {
            int ix = s.indexOf(e.getKey());
            if (ix >= 0 && s.vers[ix] < e.getValue()) return true;
        }
        return false;
    }

    /** build and seal a graph over n rows (ids[i], src.vec(i)); throttle: apply the merge rate cap */
    private SealedGraph build(long[] ids, int n, VecSource src, boolean throttle) {
        GraphIndex gi = new GraphIndex(dim, metric, cfg.graph());
        int th = Math.max(1, Math.min(cfg.buildThreads(), n / 256 + 1));
        int rate = throttle ? cfg.mergeRatePerSec() : sealRate; // fix6: sealRate is 0 (unthrottled) unless the harness sets it
        if (th == 1) {
            addRange(gi, ids, src, 0, n, rate);
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
            if (err[0] != null) throw new IllegalStateException(err[0]);
        }
        return gi.seal();
    }

    private void addRange(GraphIndex gi, long[] ids, VecSource src, int lo, int hi, int rate) {
        long t0 = System.nanoTime();
        for (int i = lo; i < hi; i++) {
            if (closed) throw new IllegalStateException("closed");
            gi.add(ids[i], src.vec(i));
            if (rate > 0 && ((i - lo) & 127) == 127) {
                long due = t0 + (long) (i - lo + 1) * 1_000_000_000L / rate;
                long now = System.nanoTime();
                if (due > now) LockSupport.parkNanos(due - now);
            }
        }
    }

    // ---- control for tests and reports ----

    /** seal the active tail now, on the calling thread, even below smallLimit; waits for a running worker task first; finishes a pending floor cleanup */
    public void sealNow() {
        work.lock();
        try {
            doSeal(true);
            settleFloorIfDue();
        } finally {
            work.unlock();
        }
    }

    /** merge every segment into one (or compact the lone one when it has hidden copies), on the calling thread; finishes a pending floor cleanup */
    public void mergeNow() {
        work.lock();
        try {
            doMerge(true);
            settleFloorIfDue();
        } finally {
            work.unlock();
        }
    }

    /**
     * waits until no seal or merge is due or running and no floor cleanup is pending (for a quiet writer); false on timeout. The floor
     * cleanup waits for the searches that still run on a replaced view, so a search that outlasts the timeout makes this return false.
     */
    public boolean awaitIdle(long timeoutMs) {
        long end = System.nanoTime() + timeoutMs * 1_000_000L;
        while (System.nanoTime() < end) {
            settleFloorIfDue(); // on this thread: no wait for the worker's next wake
            if (!floorDue && !sealDue() && pickMerge(view.segs, false) == null && work.tryLock()) {
                work.unlock();
                if (!floorDue && !sealDue() && pickMerge(view.segs, false) == null) return true;
            }
            if (!worker.isAlive()) return false;
            LockSupport.parkNanos(2_000_000L);
        }
        return false;
    }

    /**
     * Rebuild everything from raw vectors, as the engine would after a crash (records are the truth): tail, segments and floor map are
     * replaced. n below smallLimit: the vectors go to the exact tail; otherwise one segment is built from all of them. Writers wait for
     * the duration; searches keep using the old state until the swap. A later duplicate id replaces an earlier one.
     */
    public void rebuildFrom(Iterable<IdVector> src) {
        work.lock();
        mu.lock();
        try {
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
            Tail nt = new Tail(dim, metric, cfg.tailQuant());
            Segment[] segs = new Segment[0];
            if (n >= cfg.smallLimit() && n > 0) {
                long[] vers = new long[n];
                for (int j = 0; j < n; j++) vers[j] = ++counter;
                SealedGraph sg = build(ids, n, j -> vecs[j].clone(), false);
                if (sg.n != n) throw new IllegalStateException("rebuilt rows " + sg.n + " != " + n);
                segs = new Segment[] {new Segment(sg, ids, vers)};
            } else {
                for (int j = 0; j < n; j++) nt.put(ids[j], new Rec(vecs[j].clone(), ++counter));
            }
            floor.clear();
            view = new View(segs, null, nt);
            live.set(n);
        } finally {
            mu.unlock();
            work.unlock();
        }
    }

    // ---- persistence (first cut: a settled tier, segments only) ----

    /**
     * Writes the settled tier into a directory as a new generation: one file per segment (a segment that an earlier checkpoint or open
     * already put in this directory is not written again), a tombstone file and a manifest, in the publish order of {@link Manifest#publish}.
     * A process kill at any point leaves the previous generation or this one, never a mixture. Files that the new generation no longer
     * names are deleted afterwards.
     *
     * <p>This first cut persists segments only, so it refuses a tier that holds anything else, with an {@link IllegalStateException}: a
     * non-empty tail (call {@link #sealNow} first), an entry in the floor map, or a segment with hidden copies. Writers and the seal and
     * merge worker wait while it runs. One process owns a directory at a time; nothing here locks it.
     */
    public void checkpoint(Path dir) throws IOException {
        checkpoint(dir, FaultHook.NONE);
    }

    /** {@link #checkpoint(Path)} with a hook between the publish steps, for the crash tests */
    void checkpoint(Path dir, FaultHook hook) throws IOException {
        work.lock();
        mu.lock();
        try {
            View v = view;
            if (v.frozen != null || !v.active.raw.isEmpty()) {
                throw new IllegalStateException("checkpoint needs an empty tail: call sealNow() first (this version persists segments only)");
            }
            if (floorDue) settleFloor(); // a merge that overlapped searches may have left entries that no search needs any more
            if (!floor.isEmpty() && held.isEmpty()) { // with a held view, entries may serve only that view: the segments of this view are checked below
                throw new IllegalStateException("checkpoint needs an empty floor map, " + floor.size() + " ids have a replaced or deleted copy in a segment (this version persists no tombstones)");
            }
            for (Segment s : v.segs) {
                if (s.dead != 0) throw new IllegalStateException("checkpoint needs segments without hidden copies (this version persists no tombstones)");
            }
            Files.createDirectories(dir);
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
            Manifest m = new Manifest(gen, Manifest.NONE, Manifest.NONE, counter, nextSeg, fp, entries, 0L, 0L, 0);
            Manifest done = Manifest.publish(dir, m, hook);
            for (int i = 0; i < v.segs.length; i++) v.segs[i].fileId = fileIds[i];
            persistedDir = dir;
            persistedManifest = done;
            Manifest.gc(dir, done);
        } finally {
            mu.unlock();
            work.unlock();
        }
    }

    /**
     * Opens the tier that {@link #checkpoint} wrote. Takes the newest manifest that validates, checks its fingerprint against the
     * configuration, reads every segment with every section checksummed, deletes files that the manifest does not name, and restores the
     * version counter. The dimension and the metric come from the manifest. Any failure throws: {@link SegmentFormat.CorruptException}
     * for damage, {@link SegmentFormat.UnsupportedException} for a newer format, {@link SegmentFormat.MismatchException} when the files
     * belong to another index definition, {@link NoSuchFileException} when the directory has no manifest. A caller that holds the
     * records falls back to {@link #rebuildFrom}, or uses the overload with a rebuild source that does it. A null config means the defaults.
     */
    public static TieredIndex open(Path dir, Config cfg) throws IOException {
        Config c = cfg != null ? cfg : Config.defaults();
        Manifest m = Manifest.readLatest(dir);
        if (m == null) throw new NoSuchFileException(dir.toString(), null, "no manifest in the directory");
        return restore(dir, m, m.fingerprint().dim(), m.fingerprint().metric(), c);
    }

    /**
     * Like {@link #open(Path, Config)} for a caller that knows the index definition and holds the records: on any failure to read the
     * directory (damage, a newer format, another fingerprint, no manifest) it builds the tier from {@code rebuildSource} with
     * {@link #rebuildFrom}, which is always correct and costs a build. The directory is not touched in that case; the next
     * {@link #checkpoint} replaces what is in it. {@link #openFallbackCause()} tells which failure it was.
     */
    public static TieredIndex open(Path dir, int dim, Metric metric, Config cfg, Supplier<? extends Iterable<IdVector>> rebuildSource) {
        Config c = cfg != null ? cfg : Config.defaults();
        Throwable cause;
        try {
            Manifest m = Manifest.readLatest(dir);
            if (m != null) return restore(dir, m, dim, metric, c);
            cause = new NoSuchFileException(dir.toString(), null, "no manifest in the directory");
        } catch (IOException e) {
            cause = e;
        }
        TieredIndex t = new TieredIndex(dim, metric, c);
        try {
            t.rebuildFrom(rebuildSource.get());
        } catch (RuntimeException e) {
            t.close();
            throw e;
        }
        t.fallbackCause = cause;
        return t;
    }

    /** why {@link #open(Path, int, Metric, Config, Supplier)} rebuilt instead of reading files, or null when the segments came from files */
    public Throwable openFallbackCause() {
        return fallbackCause;
    }

    private static TieredIndex restore(Path dir, Manifest m, int dim, Metric metric, Config cfg) throws IOException {
        Manifest.Fingerprint fp = m.fingerprint();
        GraphIndex.Params gp = cfg.graph();
        if (fp.dim() != dim || fp.metric() != metric || fp.M() != gp.M() || fp.efConstruction() != gp.efConstruction() || fp.sq8() != gp.sq8()) {
            throw new SegmentFormat.MismatchException("the persisted segments were built for " + fp + ", not for dim " + dim + " " + metric + " M "
                    + gp.M() + " efConstruction " + gp.efConstruction() + " sq8 " + gp.sq8());
        }
        List<Segment> segs = new ArrayList<>();
        long total = 0;
        long maxVer = 0;
        for (Manifest.Entry e : m.entries()) {
            SegmentReader.Loaded l = SegmentReader.readFull(dir.resolve(Manifest.segName(e.segId())), true);
            SegmentInfo si = l.info();
            if (l.vers() == null) throw new SegmentFormat.CorruptException("segment " + e.segId() + " has no version section", "VERS");
            if (si.n() != e.n() || si.fileLength() != e.fileLength() || si.headerCrc() != e.headerCrc()) {
                throw new SegmentFormat.CorruptException("segment " + e.segId() + " changed since the manifest was read", "MANIFEST");
            }
            Segment s = new Segment(l.graph(), l.vers());
            s.fileId = e.segId();
            segs.add(s);
            total += s.n;
            maxVer = Math.max(maxVer, si.verMax());
        }
        if (maxVer > m.versionCounter()) throw new SegmentFormat.CorruptException("a segment holds version " + maxVer + " above the manifest's counter " + m.versionCounter(), "MANIFEST");
        int live = Math.toIntExact(total);
        Manifest.gc(dir, m); // orphans: files of a crashed publish, older generations
        TieredIndex t = new TieredIndex(dim, metric, cfg);
        t.work.lock();
        t.mu.lock();
        try {
            t.view = new View(segs.toArray(new Segment[0]), null, t.view.active);
            t.counter = m.versionCounter();
            t.live.set(live);
            t.persistedDir = dir;
            t.persistedManifest = m;
        } finally {
            t.mu.unlock();
            t.work.unlock();
        }
        return t;
    }

    // ---- reports ----

    @Override
    public long bytesHeld() {
        View v = view;
        long b = 512;
        b += tailBytes(v.active);
        if (v.frozen != null) b += tailBytes(v.frozen);
        for (Segment s : v.segs) b += s.bytes();
        b += floor.size() * 80L;
        return b;
    }

    private long tailBytes(Tail t) {
        return t.flat.bytesHeld() + (long) t.raw.size() * ((long) dim * 4 + 16 + 24 + 64);
    }

    public Stats stats() {
        View v = view;
        final long a = addsCount, sv = sealedVectors.get(), mv = mergedVectors.get();
        final int tc = taskCount; // fix6: published task slots only
        return new Stats(seals.get(), merges.get(), pendingTail(), v.segs.length, bytesHeld(), sealNanos.get() / 1_000_000L,
                mergeNanos.get() / 1_000_000L, floor.size(), failures.get(), live.get(), backpressure.get(), sealedVectors.get(),
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

    @Override
    public void close() {
        closed = true;
        wake.release();
        worker.interrupt();
        try {
            worker.join(30_000L);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
