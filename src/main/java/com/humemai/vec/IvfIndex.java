package com.humemai.vec;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.LongPredicate;

/**
 * IVF (inverted file) index: k-means centroids, one append-only list per centroid, a search probes the nearest nprobe lists.
 * Written from the published algorithms (Sivic and Zisserman's inverted file, the IVF part of Jegou et al.'s IVFADC, Lloyd k-means with
 * k-means++ seeding); no third-party source was read or adapted.
 *
 * <p>Why IVF here (VECTOR-PROGRAM.md R1, R4, R7): an insert is one pass over the centroids plus an append, a delete is a tombstone, a build
 * is k-means instead of a graph build. The price is that the centroids go stale when the data drifts; {@link #needsRetrain()} measures that
 * and {@link #retrain()} rebuilds in the background and swaps atomically.
 *
 * <p>Layout. A {@link Model} is one immutable partitioning: centroids, one {@link Lst} per centroid, and a per-stripe directory
 * (id to list index) used only by writers. A list is a chain of {@link Seg}s: a Seg has fixed capacity arrays (ids, fp32 vectors kept for the
 * rerank, optional SQ8 codes with a scale per vector, a live bitmap) and a volatile count; the list holds a volatile reference to the
 * current Seg. A reader reads the Seg reference, then its count, and scans slots below the count: no lock. A writer (holding the list
 * monitor) writes a slot, then publishes the count (volatile write). When a Seg is full the writer copies it into a bigger one, writes the
 * new slot there, and publishes the new Seg; compaction builds a fresh Seg of the live slots and publishes it the same way. An old Seg
 * stays valid for a reader that still holds it.
 *
 * <p>Writers. add and delete take the stripe lock of the id (64 stripes), then the list monitor(s): the lock order is stripe, then list.
 * Replace is "append the new copy, then tombstone the old one", so an id is never absent while it is replaced (a search racing the replace
 * can see both copies; the result is de-duplicated). The directory holds the list of each id so delete and replace find the list; the slot
 * inside the list is found by scanning the list ids (lists hold about sqrt(n)/4 vectors), which keeps compaction free of directory updates.
 *
 * <p>Retrain (also the first training when the live count reaches trainAt). Under all stripe locks the writers are paused for an instant
 * while a change log is switched on; from then on every add and delete is applied to the current model AND appended to the log (inside the
 * stripe lock, so the log keeps the per-id order). The retrainer snapshots the live vectors, runs k-means on a sample, assigns every vector,
 * builds the new model off to the side, replays the log (the bulk of it without locks, the last few entries under all stripe locks), and
 * publishes the new model with one volatile write. Searches never wait: they use the old model until the swap. The snapshot may contain a
 * vector that the log also contains (or a vector deleted after the snapshot); the replay is idempotent (an add of an existing id replaces,
 * a delete of a missing id is a no-op), so the new model equals the old one at the swap.
 *
 * <p>Metrics. Stored vectors are normalized for COSINE. Centroids are plain L2 k-means centroids of the stored vectors for every metric;
 * a vector goes to the centroid maximising 2 x.c - |c|^2 (the nearest in L2). A search ranks the centroids by the same expression for L2 and
 * COSINE and by x.c for DOT. SQ8 is the per-vector symmetric scheme of the flat tier (scale = max|x| / 127, kernel F8X4, for L2 the exact
 * squared norm is stored); scores are higher-is-better as in {@link Hit}. With SQ8 the scan keeps rerankFactor * k candidates and the final
 * score is exact fp32.
 *
 * <p>Filters. allow is applied inside the scan. When fewer than k candidates (fewer than rerankFactor * k when filtering) were found after
 * nprobe lists, probing continues down the centroid ranking until enough qualify; the limit of that is a full scan, which is the exact
 * fallback (every qualifying vector is then reranked in fp32).
 *
 * <p>Not done: serialize/load, PQ or residual codes, per-list locks beyond the monitor, a snapshot that avoids the temporary double copy of
 * the vectors during a rebuild (a rebuild of n vectors needs about n * dim * 4 extra bytes).
 */
public final class IvfIndex implements VecIndex {
    private static final Hit[] EMPTY = new Hit[0];
    private static final VarHandle LIVE = MethodHandles.arrayElementVarHandle(long[].class);
    static final int STRIPES = 64;

    /** All tunables. Defaults follow the experiment brief. */
    public static final class Params {
        /** number of lists; 0 = round(4 * sqrt(n)) at train time */
        public int nlist = 0;
        /** lists probed per search; 0 = max(8, nlist / 16) */
        public int nprobe = 0;
        /** train lazily when the live count first reaches this; until then every search is an exact scan of one flat list */
        public int trainAt = 2000;
        /** NONE (fp32 scan) or SQ8 (codes scan plus fp32 rerank) */
        public Quant quant = Quant.SQ8;
        /** SQ8: candidates kept for the fp32 rerank = rerankFactor * k */
        public int rerankFactor = 5;
        /** 1 or 2: assign each vector to its nearest 1 or 2 lists (boundary overlap) */
        public int overlap = 1;
        /** Lloyd iterations (stops early when no assignment changes) */
        public int kmeansIters = 10;
        /** at most this many vectors train the centroids */
        public int trainSample = 50000;
        public long seed = 42;
        /** worker threads of k-means and of the final assignment; 1 = single threaded */
        public int threads = Math.max(1, Math.min(4, Runtime.getRuntime().availableProcessors()));
        /** needsRetrain: mean squared distance to the assigned centroid of the vectors added since the build, relative to the build value */
        public double driftRatio = 1.5;
        /** needsRetrain: largest list over the mean list size */
        public double skewRatio = 8.0;
        /** needsRetrain: live count over the live count at the last build */
        public double growthRatio = 4.0;
        /** the drift test needs at least this many adds since the build */
        public int minAddsForDrift = 256;
        /** compactIfNeeded compacts a list when its dead fraction exceeds this (and it holds at least 16 dead slots) */
        public double compactDeadFraction = 0.25;

        public Params copy() {
            Params q = new Params();
            q.nlist = nlist;
            q.nprobe = nprobe;
            q.trainAt = trainAt;
            q.quant = quant;
            q.rerankFactor = rerankFactor;
            q.overlap = overlap;
            q.kmeansIters = kmeansIters;
            q.trainSample = trainSample;
            q.seed = seed;
            q.threads = threads;
            q.driftRatio = driftRatio;
            q.skewRatio = skewRatio;
            q.growthRatio = growthRatio;
            q.minAddsForDrift = minAddsForDrift;
            q.compactDeadFraction = compactDeadFraction;
            return q;
        }

        /** keys: nlist nprobe trainAt quant rerankFactor overlap kmeansIters(iters) trainSample(sample) seed threads driftRatio skewRatio growthRatio minAddsForDrift compactDeadFraction */
        public static Params fromMap(Map<String, String> m) {
            Params q = new Params();
            if (m == null) return q;
            for (Map.Entry<String, String> e : m.entrySet()) {
                String k = e.getKey(), v = e.getValue();
                switch (k) {
                    case "nlist": q.nlist = Integer.parseInt(v); break;
                    case "nprobe": q.nprobe = Integer.parseInt(v); break;
                    case "trainAt": q.trainAt = Integer.parseInt(v); break;
                    case "quant": q.quant = Quant.valueOf(v); break;
                    case "rerankFactor": q.rerankFactor = Integer.parseInt(v); break;
                    case "overlap": q.overlap = Integer.parseInt(v); break;
                    case "kmeansIters":
                    case "iters": q.kmeansIters = Integer.parseInt(v); break;
                    case "trainSample":
                    case "sample": q.trainSample = Integer.parseInt(v); break;
                    case "seed": q.seed = Long.parseLong(v); break;
                    case "threads": q.threads = Integer.parseInt(v); break;
                    case "driftRatio": q.driftRatio = Double.parseDouble(v); break;
                    case "skewRatio": q.skewRatio = Double.parseDouble(v); break;
                    case "growthRatio": q.growthRatio = Double.parseDouble(v); break;
                    case "minAddsForDrift": q.minAddsForDrift = Integer.parseInt(v); break;
                    case "compactDeadFraction": q.compactDeadFraction = Double.parseDouble(v); break;
                    default: throw new IllegalArgumentException("unknown IVF parameter " + k);
                }
            }
            return q;
        }
    }

    // ---------------------------------------------------------------- storage

    /** fixed capacity arrays of one list; a slot is written once; count is the publication point */
    private static final class Seg {
        final int cap;
        final long[] ids;
        final float[] vec;
        final byte[] code;
        final float[] scale;
        final float[] nn;
        final long[] live;
        volatile int count;

        Seg(int cap, int dim, boolean codes, boolean l2) {
            this.cap = cap;
            ids = new long[cap];
            vec = new float[cap * dim];
            code = codes ? new byte[cap * dim] : null;
            scale = codes ? new float[cap] : null;
            nn = codes && l2 ? new float[cap] : null;
            live = new long[(cap + 63) >>> 6];
        }

        boolean isLive(int s) {
            return (((long) LIVE.getAcquire(live, s >>> 6)) >>> (s & 63) & 1L) != 0L;
        }

        void setLive(int s) {
            LIVE.setRelease(live, s >>> 6, live[s >>> 6] | (1L << (s & 63)));
        }

        void clearLive(int s) {
            LIVE.setRelease(live, s >>> 6, live[s >>> 6] & ~(1L << (s & 63)));
        }

        long bytes(int dim) {
            long b = 96 + 8L * cap + 4L * cap * dim + 8L * live.length;
            if (code != null) b += (long) cap * dim + 4L * cap;
            if (nn != null) b += 4L * cap;
            return b;
        }
    }

    /** one inverted list; readers use seg (volatile) and seg.count; writers hold this monitor */
    private static final class Lst {
        final int dim;
        final boolean codes, l2;
        volatile Seg seg;
        volatile int liveN, dead;

        Lst(int dim, boolean codes, boolean l2, int cap) {
            this.dim = dim;
            this.codes = codes;
            this.l2 = l2;
            this.seg = new Seg(Math.max(cap, 16), dim, codes, l2);
        }

        private void write(Seg s, int slot, long id, float[] v, int vo) {
            s.ids[slot] = id;
            System.arraycopy(v, vo, s.vec, slot * dim, dim);
            if (codes) {
                encodeSq8(v, vo, s.code, slot * dim, dim, s.scale, slot);
                if (s.nn != null) s.nn[slot] = Kernels.dot(v, vo, v, vo, dim);
            }
            s.setLive(slot);
        }

        /** @return the slot of the new copy */
        synchronized int append(long id, float[] v, int vo) {
            Seg s = seg;
            int n = s.count;
            if (n == s.cap) {
                Seg t = new Seg(s.cap * 2, dim, codes, l2);
                System.arraycopy(s.ids, 0, t.ids, 0, n);
                System.arraycopy(s.vec, 0, t.vec, 0, n * dim);
                if (codes) {
                    System.arraycopy(s.code, 0, t.code, 0, n * dim);
                    System.arraycopy(s.scale, 0, t.scale, 0, n);
                    if (s.nn != null) System.arraycopy(s.nn, 0, t.nn, 0, n);
                }
                System.arraycopy(s.live, 0, t.live, 0, s.live.length);
                t.count = n;
                write(t, n, id, v, vo);
                t.count = n + 1;
                liveN = liveN + 1;
                seg = t; // publication of the new Seg (volatile write)
                return n;
            }
            write(s, n, id, v, vo);
            liveN = liveN + 1;
            s.count = n + 1; // publication of the slot
            return n;
        }

        /**
         * Tombstones the OLDEST live slot holding id; @return true when one was found. A replace appends the new copy first, so while both
         * copies are live in one list the old one has the lower slot, and compaction and growth keep the order. (An earlier version
         * skipped the new copy by its slot number; a compaction between the append and the tombstone renumbers the slots, the stale number
         * pointed elsewhere and the NEW copy was tombstoned: seen once in the 10 s concurrency run as "replaced id not found right after
         * add".) The parameter except is kept for the callers and ignored.
         */
        synchronized boolean tombstone(long id, int except) {
            Seg s = seg;
            int n = s.count;
            for (int i = 0; i < n; i++) {
                if (s.ids[i] == id && s.isLive(i)) {
                    s.clearLive(i);
                    liveN = liveN - 1;
                    dead = dead + 1;
                    return true;
                }
            }
            return false;
        }

        /** rewrites the live slots into a fresh Seg and publishes it */
        synchronized boolean compact() {
            if (dead == 0) return false;
            Seg s = seg;
            int n = s.count;
            int ln = liveN;
            Seg t = new Seg(Math.max(16, ln + (ln >> 2) + 1), dim, codes, l2);
            int j = 0;
            for (int i = 0; i < n; i++) {
                if (!s.isLive(i)) continue;
                t.ids[j] = s.ids[i];
                System.arraycopy(s.vec, i * dim, t.vec, j * dim, dim);
                if (codes) {
                    System.arraycopy(s.code, i * dim, t.code, j * dim, dim);
                    t.scale[j] = s.scale[i];
                    if (s.nn != null) t.nn[j] = s.nn[i];
                }
                t.live[j >>> 6] |= 1L << (j & 63);
                j++;
            }
            t.count = j;
            liveN = j;
            dead = 0;
            seg = t;
            return true;
        }
    }

    /** symmetric int8, scale = max|x| / 127 (the flat tier scheme) */
    private static void encodeSq8(float[] src, int so, byte[] out, int oo, int dim, float[] scales, int si) {
        float m = 0f;
        for (int i = 0; i < dim; i++) m = Math.max(m, Math.abs(src[so + i]));
        if (m == 0f) {
            for (int i = 0; i < dim; i++) out[oo + i] = 0;
            scales[si] = 0f;
            return;
        }
        float inv = 127f / m;
        for (int i = 0; i < dim; i++) {
            int v = Math.round(src[so + i] * inv);
            out[oo + i] = (byte) (v > 127 ? 127 : Math.max(v, -127));
        }
        scales[si] = m / 127f;
    }

    /** one immutable partitioning; only the lists, the directory maps (under their stripe locks) and the adders change */
    private static final class Model {
        final int nlist; // 0 = untrained: one flat list, exact scan
        final float[] cent, cn;
        final Lst[] lists;
        final LongIntMap[] d1, d2; // per stripe: id to primary list, id to secondary list (d2 null unless overlap > 1)
        final boolean codes;
        final double meanDist0;
        final int builtN;
        final DoubleAdder distSum = new DoubleAdder();
        final LongAdder distCnt = new LongAdder();

        Model(int nlist, float[] cent, float[] cn, Lst[] lists, LongIntMap[] d1, LongIntMap[] d2, boolean codes, double meanDist0, int builtN) {
            this.nlist = nlist;
            this.cent = cent;
            this.cn = cn;
            this.lists = lists;
            this.d1 = d1;
            this.d2 = d2;
            this.codes = codes;
            this.meanDist0 = meanDist0;
            this.builtN = builtN;
        }
    }

    private static final class Entry {
        final long id;
        final float[] v; // null = delete

        Entry(long id, float[] v) {
            this.id = id;
            this.v = v;
        }
    }

    private interface Range {
        void run(int from, int to);
    }

    /** open addressing set of longs (not thread-safe) */
    private static final class LongSet {
        long[] keys;
        boolean[] used;
        int n;

        LongSet(int expect) {
            int c = 16;
            while (c < expect * 2) c <<= 1;
            keys = new long[c];
            used = new boolean[c];
        }

        private static int hash(long k) {
            k ^= k >>> 33;
            k *= 0xff51afd7ed558ccdL;
            k ^= k >>> 33;
            return (int) k;
        }

        /** @return true when newly added */
        boolean add(long k) {
            if ((n + 1) * 2 > keys.length) grow();
            int mask = keys.length - 1;
            int i = hash(k) & mask;
            while (used[i]) {
                if (keys[i] == k) return false;
                i = (i + 1) & mask;
            }
            used[i] = true;
            keys[i] = k;
            n++;
            return true;
        }

        private void grow() {
            long[] ok = keys;
            boolean[] ou = used;
            keys = new long[ok.length * 2];
            used = new boolean[ok.length * 2];
            n = 0;
            for (int i = 0; i < ok.length; i++) if (ou[i]) add(ok[i]);
        }
    }

    /** bounded selection of the best cap entries (root = worst kept), same order as TopK, plus the Seg each entry came from */
    private static final class Heap {
        final int cap;
        final float[] s;
        final long[] id;
        final int[] slot;
        final Seg[] sg;
        int n;

        Heap(int cap) {
            this.cap = Math.max(cap, 1);
            s = new float[this.cap];
            id = new long[this.cap];
            slot = new int[this.cap];
            sg = new Seg[this.cap];
        }

        void offer(float sc, long ident, int sl, Seg seg) {
            if (n < cap) {
                int i = n++;
                while (i > 0) {
                    int p = (i - 1) >> 1;
                    if (TopK.worse(sc, ident, s[p], id[p])) {
                        s[i] = s[p];
                        id[i] = id[p];
                        slot[i] = slot[p];
                        sg[i] = sg[p];
                        i = p;
                    } else break;
                }
                put(i, sc, ident, sl, seg);
            } else if (TopK.worse(s[0], id[0], sc, ident)) {
                siftDown(0, sc, ident, sl, seg, n);
            }
        }

        private void put(int i, float sc, long ident, int sl, Seg seg) {
            s[i] = sc;
            id[i] = ident;
            slot[i] = sl;
            sg[i] = seg;
        }

        private void siftDown(int i, float sc, long ident, int sl, Seg seg, int size) {
            int half = size >> 1;
            while (i < half) {
                int c = 2 * i + 1;
                int r = c + 1;
                if (r < size && TopK.worse(s[r], id[r], s[c], id[c])) c = r;
                if (TopK.worse(s[c], id[c], sc, ident)) {
                    s[i] = s[c];
                    id[i] = id[c];
                    slot[i] = slot[c];
                    sg[i] = sg[c];
                    i = c;
                } else break;
            }
            put(i, sc, ident, sl, seg);
        }

        /** empties the heap; entries best first */
        Hit[] drain() {
            int m = n;
            Hit[] out = new Hit[m];
            for (int i = m - 1; i >= 0; i--) {
                out[i] = new Hit(id[0], s[0]);
                n--;
                if (n > 0) siftDown(0, s[n], id[n], slot[n], sg[n], n);
            }
            return out;
        }
    }

    private static final class Asg {
        int a = -1, b = -1;
        float best;
    }

    private static final class AsgAll {
        int[] best, sec;
        float[] dist;
    }

    private static final class Snap {
        long[] ids;
        float[] data;
        int n;
    }

    // ---------------------------------------------------------------- state

    private final int dim;
    private final Metric metric;
    private final boolean l2;
    private final Params p;
    private volatile int nprobeCfg, rerankCfg;
    private final ReentrantLock[] stripes = new ReentrantLock[STRIPES];
    private final ReentrantLock rebuildLock = new ReentrantLock();
    private final LongAdder size = new LongAdder();
    private final AtomicInteger rebuilds = new AtomicInteger();
    private volatile Model model;
    private volatile ConcurrentLinkedQueue<Entry> log;
    private volatile Throwable lastError;
    private volatile boolean trainFailed;
    private volatile ScheduledThreadPoolExecutor auto;

    public IvfIndex(int dim, Metric metric) {
        this(dim, metric, new Params());
    }

    /** harness shape: string parameters, see {@link Params#fromMap} */
    public IvfIndex(int dim, Metric metric, Map<String, String> params) {
        this(dim, metric, Params.fromMap(params));
    }

    public IvfIndex(int dim, Metric metric, Params params) {
        if (dim <= 0) throw new IllegalArgumentException("dim");
        if (params.quant == Quant.BINARY) throw new IllegalArgumentException("IVF supports quant NONE and SQ8");
        if (params.overlap < 1 || params.overlap > 2) throw new IllegalArgumentException("overlap must be 1 or 2");
        this.dim = dim;
        this.metric = metric;
        this.l2 = metric == Metric.L2;
        this.p = params.copy();
        this.nprobeCfg = p.nprobe;
        this.rerankCfg = p.rerankFactor;
        for (int i = 0; i < STRIPES; i++) stripes[i] = new ReentrantLock();
        this.model = flatModel();
    }

    /**
     * Builds an IVF index over the given vectors in one go (the segment type of a tiered index: ids distinct, flat is n * dim row-major, not
     * modified, not retained). Trains k-means when n >= params.trainAt, otherwise leaves one exact flat list (a tiny segment needs no
     * centroids). The result is a normal live index: add, delete, search, retrain and compact all work on it.
     */
    public static IvfIndex fromVectors(long[] ids, float[] flat, int dim, Metric metric, Params params) {
        int n = ids.length;
        if ((long) n * dim != flat.length) throw new IllegalArgumentException("flat.length must be ids.length * dim");
        IvfIndex x = new IvfIndex(dim, metric, params);
        LongSet dup = new LongSet(n);
        for (long id : ids) if (!dup.add(id)) throw new IllegalArgumentException("duplicate id " + id);
        float[] data = flat;
        if (metric == Metric.COSINE) {
            data = new float[flat.length];
            for (int i = 0; i < n; i++) {
                float[] r = Kernels.normalized(Arrays.copyOfRange(flat, i * dim, (i + 1) * dim));
                System.arraycopy(r, 0, data, i * dim, dim);
            }
        }
        for (int i = 0; i < flat.length; i++) if (!Float.isFinite(flat[i])) throw new IllegalArgumentException("non-finite component");
        if (n < x.p.trainAt || n == 0) {
            for (int i = 0; i < n; i++) x.addStored(ids[i], data, i * dim);
            return x;
        }
        try {
            x.model = x.buildModel(ids, data, n, null);
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        x.size.add(n);
        x.rebuilds.incrementAndGet();
        return x;
    }

    public static IvfIndex fromVectors(long[] ids, float[][] vs, int dim, Metric metric, Params params) {
        float[] flat = new float[ids.length * dim];
        for (int i = 0; i < ids.length; i++) {
            if (vs[i].length != dim) throw new IllegalArgumentException("dim");
            System.arraycopy(vs[i], 0, flat, i * dim, dim);
        }
        return fromVectors(ids, flat, dim, metric, params);
    }

    private void addStored(long id, float[] data, int off) { // single threaded bulk helper, before publication of the index
        float[] v = Arrays.copyOfRange(data, off, off + dim);
        int s = stripeOf(id);
        if (!applyAdd(model, id, v, s)) size.increment();
    }

    private Model flatModel() {
        Lst[] ls = {new Lst(dim, false, l2, 16)};
        return new Model(0, null, null, ls, newDirs(), p.overlap > 1 ? newDirs() : null, false, 0.0, 0);
    }

    private static LongIntMap[] newDirs() {
        LongIntMap[] d = new LongIntMap[STRIPES];
        for (int i = 0; i < STRIPES; i++) d[i] = new LongIntMap();
        return d;
    }

    private static int stripeOf(long id) {
        return (int) ((id * 0x9E3779B97F4A7C15L) >>> 58);
    }

    @Override
    public int dim() {
        return dim;
    }

    @Override
    public Metric metric() {
        return metric;
    }

    @Override
    public int size() {
        return (int) size.sum();
    }

    public boolean isTrained() {
        return model.nlist > 0;
    }

    public int nlist() {
        return model.nlist;
    }

    public int rebuildCount() {
        return rebuilds.get();
    }

    public Throwable lastError() {
        return lastError;
    }

    public Params params() {
        return p.copy();
    }

    public void setNprobe(int n) {
        nprobeCfg = n;
    }

    public void setRerankFactor(int f) {
        rerankCfg = Math.max(1, f);
    }

    // ---------------------------------------------------------------- writes

    @Override
    public void add(long id, float[] v) {
        if (v.length != dim) throw new IllegalArgumentException("dim " + v.length + " != " + dim);
        for (int i = 0; i < dim; i++) if (!Float.isFinite(v[i])) throw new IllegalArgumentException("non-finite component");
        float[] st = metric == Metric.COSINE ? Kernels.normalized(v) : v;
        int s = stripeOf(id);
        ReentrantLock lk = stripes[s];
        lk.lock();
        try {
            ConcurrentLinkedQueue<Entry> lg = log;
            if (lg != null) lg.add(new Entry(id, metric == Metric.COSINE ? st : v.clone()));
            if (!applyAdd(model, id, st, s)) size.increment();
        } finally {
            lk.unlock();
        }
        if (p.trainAt > 0 && !trainFailed && model.nlist == 0 && size.sum() >= p.trainAt) maybeTrain();
    }

    @Override
    public boolean delete(long id) {
        int s = stripeOf(id);
        ReentrantLock lk = stripes[s];
        lk.lock();
        try {
            boolean ex = applyDel(model, id, s);
            if (ex) {
                size.decrement();
                ConcurrentLinkedQueue<Entry> lg = log;
                if (lg != null) lg.add(new Entry(id, null));
            }
            return ex;
        } finally {
            lk.unlock();
        }
    }

    /** appends v (stored form) to its list(s), then tombstones the old copy; the caller holds the stripe lock or owns the model alone. @return true when the id existed */
    private boolean applyAdd(Model m, long id, float[] v, int stripe) {
        LongIntMap a = m.d1[stripe];
        int old1 = a.get(id);
        int old2 = m.d2 != null ? m.d2[stripe].get(id) : -1;
        int l1, l2 = -1;
        if (m.nlist == 0) {
            l1 = 0;
        } else {
            Asg g = nearest(m, v, 0, p.overlap > 1 && m.nlist > 1);
            l1 = g.a;
            l2 = g.b;
            float d = Kernels.dot(v, 0, v, 0, dim) - g.best;
            m.distSum.add(d > 0f ? d : 0f);
            m.distCnt.increment();
        }
        int slot1 = m.lists[l1].append(id, v, 0);
        int slot2 = -1;
        if (l2 >= 0) slot2 = m.lists[l2].append(id, v, 0);
        a.put(id, l1);
        if (m.d2 != null) {
            if (l2 >= 0) m.d2[stripe].put(id, l2);
            else if (old2 != -1) m.d2[stripe].remove(id);
        }
        if (old1 != -1) {
            m.lists[old1].tombstone(id, old1 == l1 ? slot1 : (old1 == l2 ? slot2 : -1));
            if (old2 != -1) m.lists[old2].tombstone(id, old2 == l1 ? slot1 : (old2 == l2 ? slot2 : -1));
        }
        return old1 != -1;
    }

    private boolean applyDel(Model m, long id, int stripe) {
        int l1 = m.d1[stripe].remove(id);
        if (l1 == -1) return false;
        int l2 = m.d2 != null ? m.d2[stripe].remove(id) : -1;
        m.lists[l1].tombstone(id, -1);
        if (l2 >= 0) m.lists[l2].tombstone(id, -1);
        return true;
    }

    /** nearest (and second nearest) centroid: maximises 2 x.c - |c|^2, which is the L2 nearest */
    private Asg nearest(Model m, float[] v, int vo, boolean wantSecond) {
        Asg r = new Asg();
        float bs = Float.NEGATIVE_INFINITY, ss = Float.NEGATIVE_INFINITY;
        int nl = m.nlist;
        float[] c = m.cent, cn = m.cn;
        for (int j = 0; j < nl; j++) {
            float sc = 2f * Kernels.dot(v, vo, c, j * dim, dim) - cn[j];
            if (sc > bs) {
                ss = bs;
                r.b = r.a;
                bs = sc;
                r.a = j;
            } else if (wantSecond && sc > ss) {
                ss = sc;
                r.b = j;
            }
        }
        if (r.a < 0) r.a = 0;
        if (!wantSecond || r.b == r.a) r.b = -1;
        r.best = bs;
        return r;
    }

    // ---------------------------------------------------------------- search

    @Override
    public Hit[] search(float[] q, int k, LongPredicate allow) {
        return search(q, k, allow, 0);
    }

    /** nprobe <= 0 means the configured default (max(8, nlist / 16) unless set) */
    public Hit[] search(float[] q, int k, LongPredicate allow, int nprobe) {
        if (k <= 0) return EMPTY;
        if (q.length != dim) throw new IllegalArgumentException("dim " + q.length + " != " + dim);
        Model m = model;
        float[] qq = metric == Metric.COSINE ? Kernels.normalized(q) : q;
        int nl = m.nlist;
        long[] keys = null;
        int np = 1;
        if (nl > 0) {
            keys = rank(m, qq);
            int cfg = nprobe > 0 ? nprobe : nprobeCfg;
            np = cfg > 0 ? cfg : Math.max(8, nl / 16);
            np = Math.min(np, nl);
        }
        boolean codes = m.codes;
        int rr = Math.max(rerankCfg, 1);
        int cand = codes ? (int) Math.min((long) rr * k, 1L << 26) : k;
        if (cand < k) cand = k;
        int need = allow == null ? k : cand;
        Heap h = new Heap(cand + 2);
        float qn = l2 ? Kernels.dot(qq, 0, qq, 0, dim) : 0f;
        LongSet seen = m.d2 != null ? new LongSet(64) : null;
        int total = nl == 0 ? 1 : nl;
        for (int i = 0; i < total; i++) {
            if (i >= np && h.n >= need) break;
            Lst lst = m.lists[nl == 0 ? 0 : (int) (keys[nl - 1 - i] & 0xffffffffL)];
            scan(lst.seg, qq, qn, allow, h, codes, seen);
        }
        Hit[] res;
        if (codes) {
            Heap r = new Heap(k + 2);
            for (int i = 0; i < h.n; i++) {
                Seg s = h.sg[i];
                int slot = h.slot[i];
                if (!s.isLive(slot)) continue;
                float sc = l2 ? -Kernels.l2sq(qq, 0, s.vec, slot * dim, dim) : Kernels.dot(qq, 0, s.vec, slot * dim, dim);
                r.offer(sc, h.id[i], slot, s);
            }
            res = r.drain();
        } else {
            res = h.drain();
        }
        res = dedupe(res);
        return res.length > k ? Arrays.copyOf(res, k) : res;
    }

    /** centroid indices sorted ascending by (score bits, index); the best centroid is the LAST key */
    private long[] rank(Model m, float[] qq) {
        int nl = m.nlist;
        long[] keys = new long[nl];
        boolean dotRank = metric == Metric.DOT;
        for (int j = 0; j < nl; j++) {
            float d = Kernels.dot(qq, 0, m.cent, j * dim, dim);
            float sc = dotRank ? d : 2f * d - m.cn[j];
            int bits = Float.floatToIntBits(sc);
            bits ^= (bits >> 31) & 0x7fffffff;
            keys[j] = ((long) bits << 32) | (long) j;
        }
        Arrays.sort(keys);
        return keys;
    }

    private void scan(Seg s, float[] qq, float qn, LongPredicate allow, Heap h, boolean codes, LongSet seen) {
        int n = s.count;
        long[] ids = s.ids;
        for (int i = 0; i < n; i++) {
            if (!s.isLive(i)) continue;
            long id = ids[i];
            if (allow != null && !allow.test(id)) continue;
            float sc;
            if (codes) {
                float d = Kernels.dotF8x4(qq, 0, s.code, i * dim, dim) * s.scale[i];
                sc = l2 ? -(qn - 2f * d + s.nn[i]) : d;
            } else {
                sc = l2 ? -Kernels.l2sq(qq, 0, s.vec, i * dim, dim) : Kernels.dot(qq, 0, s.vec, i * dim, dim);
            }
            if (h.n == h.cap && sc < h.s[0]) continue;
            if (seen != null && !seen.add(id)) continue;
            h.offer(sc, id, i, s);
        }
    }

    /** a replace in flight can show an id twice (old and new copy); keep the better one */
    private static Hit[] dedupe(Hit[] r) {
        if (r.length < 2) return r;
        long[] ids = new long[r.length];
        for (int i = 0; i < r.length; i++) ids[i] = r[i].id();
        Arrays.sort(ids);
        boolean dup = false;
        for (int i = 1; i < ids.length; i++) {
            if (ids[i] == ids[i - 1]) {
                dup = true;
                break;
            }
        }
        if (!dup) return r;
        Hit[] out = new Hit[r.length];
        int n = 0;
        outer:
        for (Hit h : r) { // best first, so the first occurrence is the best
            for (int j = 0; j < n; j++) if (out[j].id() == h.id()) continue outer;
            out[n++] = h;
        }
        return Arrays.copyOf(out, n);
    }

    // ---------------------------------------------------------------- maintenance

    /** compacts every list that has tombstones; safe during searches and writes */
    public void compact() {
        for (Lst l : model.lists) l.compact();
    }

    /** compacts the lists whose dead fraction exceeds compactDeadFraction; @return the number compacted */
    public int compactIfNeeded() {
        int c = 0;
        for (Lst l : model.lists) {
            int d = l.dead;
            if (d >= 16 && d > p.compactDeadFraction * (l.liveN + d) && l.compact()) c++;
        }
        return c;
    }

    /** the mean squared distance to the assigned centroid of the vectors added since the last build, over the value at the build (1.0 when there is too little evidence) */
    public double driftRatio() {
        Model m = model;
        if (m.nlist == 0 || m.meanDist0 <= 1e-12) return 1.0;
        long c = m.distCnt.sum();
        if (c < p.minAddsForDrift) return 1.0;
        return (m.distSum.sum() / c) / m.meanDist0;
    }

    /** largest list over the mean list size (1.0 when untrained or empty) */
    public double listSkew() {
        Model m = model;
        long n = size.sum();
        if (m.nlist == 0 || n == 0) return 1.0;
        int max = 0;
        for (Lst l : m.lists) max = Math.max(max, l.liveN);
        return max / ((double) n / m.nlist);
    }

    /** live count of every list */
    public int[] listSizes() {
        Model m = model;
        int[] r = new int[m.lists.length];
        for (int i = 0; i < r.length; i++) r[i] = m.lists[i].liveN;
        return r;
    }

    public double deadFraction() {
        long live = 0, dead = 0;
        for (Lst l : model.lists) {
            live += l.liveN;
            dead += l.dead;
        }
        return live + dead == 0 ? 0.0 : (double) dead / (live + dead);
    }

    /** the drift trigger: untrained and past trainAt; or grown by growthRatio since the build; or drift above driftRatio; or one list above skewRatio times the mean */
    public boolean needsRetrain() {
        Model m = model;
        long n = size.sum();
        if (m.nlist == 0) return p.trainAt > 0 && n >= p.trainAt && !trainFailed;
        if (n >= p.growthRatio * Math.max(m.builtN, 1)) return true;
        if (driftRatio() > p.driftRatio) return true;
        if (n >= 4L * m.nlist) {
            int max = 0;
            for (Lst l : m.lists) max = Math.max(max, l.liveN);
            if (max >= 64 && max > p.skewRatio * ((double) n / m.nlist)) return true;
        }
        return false;
    }

    /** retrains when needsRetrain() and no rebuild is running; @return true when a rebuild ran */
    public boolean retrainIfNeeded() {
        if (!needsRetrain() || !rebuildLock.tryLock()) return false;
        try {
            rebuild(null);
            return true;
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new RuntimeException(e);
        } finally {
            rebuildLock.unlock();
        }
    }

    /** harness hook: a full (re)build on the current contents */
    public void build() {
        retrain();
    }

    /**
     * Re-runs k-means on the current vectors and publishes new lists by an atomic swap; searches, adds and deletes continue meanwhile
     * (the changes made during the rebuild are replayed into the new model). Blocks the caller until the swap; concurrent callers queue.
     */
    public void retrain() {
        rebuildLock.lock();
        try {
            rebuild(null);
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            if (e instanceof InterruptedException) Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        } finally {
            rebuildLock.unlock();
        }
    }

    /** trains the centroids with k-means on the given sample (nlist from the params or round(4 sqrt(max(sample, live)))), then rebuilds the lists on them */
    public void train(float[][] sample) {
        if (sample.length == 0) throw new IllegalArgumentException("empty sample");
        float[] x = new float[sample.length * dim];
        for (int i = 0; i < sample.length; i++) {
            if (sample[i].length != dim) throw new IllegalArgumentException("dim");
            float[] r = metric == Metric.COSINE ? Kernels.normalized(sample[i]) : sample[i];
            System.arraycopy(r, 0, x, i * dim, dim);
        }
        rebuildLock.lock();
        ExecutorService pool = newPool();
        try {
            int want = p.nlist > 0 ? p.nlist : (int) Math.round(4.0 * Math.sqrt(Math.max(sample.length, size.sum())));
            int nl = Math.max(1, Math.min(want, sample.length));
            float[] c = kmeans(x, sample.length, nl, pool);
            rebuild(c);
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            if (e instanceof InterruptedException) Thread.currentThread().interrupt();
            throw new RuntimeException(e);
        } finally {
            closePool(pool);
            rebuildLock.unlock();
        }
    }

    private void maybeTrain() {
        if (!rebuildLock.tryLock()) return;
        try {
            if (model.nlist != 0 || size.sum() < p.trainAt) return;
            rebuild(null);
        } catch (Throwable t) {
            lastError = t;
            trainFailed = true;
        } finally {
            rebuildLock.unlock();
        }
    }

    /** starts a daemon thread that runs compactIfNeeded and retrainIfNeeded every periodMillis; errors go to lastError() */
    public synchronized void startAutoMaintenance(long periodMillis) {
        if (auto != null) return;
        ScheduledThreadPoolExecutor ex = new ScheduledThreadPoolExecutor(1, r -> {
            Thread t = new Thread(r, "ivf-maintenance");
            t.setDaemon(true);
            return t;
        });
        ex.scheduleWithFixedDelay(() -> {
            try {
                compactIfNeeded();
                retrainIfNeeded();
            } catch (Throwable t) {
                lastError = t;
            }
        }, periodMillis, periodMillis, TimeUnit.MILLISECONDS);
        auto = ex;
    }

    public synchronized void stopAutoMaintenance() {
        ScheduledThreadPoolExecutor ex = auto;
        auto = null;
        if (ex != null) ex.shutdownNow();
    }

    @Override
    public void close() {
        stopAutoMaintenance();
    }

    private void lockAll() {
        for (int i = 0; i < STRIPES; i++) stripes[i].lock();
    }

    private void unlockAll() {
        for (int i = STRIPES - 1; i >= 0; i--) stripes[i].unlock();
    }

    /** the caller holds rebuildLock. fixedCent: centroids to use (train), or null to run k-means */
    private void rebuild(float[] fixedCent) throws Exception {
        lockAll();
        try {
            log = new ConcurrentLinkedQueue<>();
        } finally {
            unlockAll();
        }
        boolean ok = false;
        try {
            Model base = model; // only a rebuild changes the model and we hold rebuildLock
            Snap sn = snapshot(base);
            Model nm = buildModel(sn.ids, sn.data, sn.n, fixedCent);
            ConcurrentLinkedQueue<Entry> lg = log;
            Entry e;
            for (int i = 0; i < 2_000_000 && (e = lg.poll()) != null; i++) applyEntry(nm, e); // the bulk of the log, no locks held
            lockAll();
            try {
                while ((e = lg.poll()) != null) applyEntry(nm, e); // nothing new can arrive: every stripe is locked
                model = nm; // the swap
                log = null;
            } finally {
                unlockAll();
            }
            rebuilds.incrementAndGet();
            ok = true;
        } finally {
            if (!ok) {
                lockAll();
                try {
                    log = null;
                } finally {
                    unlockAll();
                }
            }
        }
    }

    private void applyEntry(Model nm, Entry e) {
        int st = stripeOf(e.id);
        if (e.v == null) applyDel(nm, e.id, st);
        else applyAdd(nm, e.id, e.v, st);
    }

    /** copies the live vectors of the model (stored form), one copy per id */
    private Snap snapshot(Model base) {
        long want = Math.max(1024L, size.sum() + 1024L);
        int cap = (int) Math.min(want, 1L << 28);
        if ((long) cap * dim > Integer.MAX_VALUE - 64) throw new IllegalStateException("too many vectors for one rebuild snapshot");
        Snap s = new Snap();
        s.ids = new long[cap];
        s.data = new float[cap * dim];
        LongSet seen = new LongSet(cap);
        for (Lst l : base.lists) {
            Seg sg = l.seg;
            int n = sg.count;
            for (int i = 0; i < n; i++) {
                if (!sg.isLive(i)) continue;
                long id = sg.ids[i];
                if (!seen.add(id)) continue;
                if (s.n == s.ids.length) {
                    long nc = s.ids.length + (s.ids.length >> 1) + 16L;
                    if (nc * dim > Integer.MAX_VALUE - 64) throw new IllegalStateException("too many vectors for one rebuild snapshot");
                    s.ids = Arrays.copyOf(s.ids, (int) nc);
                    s.data = Arrays.copyOf(s.data, (int) nc * dim);
                }
                s.ids[s.n] = id;
                System.arraycopy(sg.vec, i * dim, s.data, s.n * dim, dim);
                s.n++;
            }
        }
        return s;
    }

    // ---------------------------------------------------------------- training

    private ExecutorService newPool() {
        if (p.threads <= 1) return null;
        return Executors.newFixedThreadPool(p.threads, r -> {
            Thread t = new Thread(r, "ivf-kmeans");
            t.setDaemon(true);
            return t;
        });
    }

    private static void closePool(ExecutorService pool) {
        if (pool != null) pool.shutdownNow();
    }

    private void parFor(ExecutorService pool, int n, Range body) throws Exception {
        if (pool == null || n < 2048) {
            body.run(0, n);
            return;
        }
        int t = Math.max(1, p.threads);
        int chunk = (n + t * 4 - 1) / (t * 4);
        List<Future<?>> fs = new ArrayList<>();
        for (int from = 0; from < n; from += chunk) {
            final int a = from, b = Math.min(n, from + chunk);
            fs.add(pool.submit(() -> body.run(a, b)));
        }
        for (Future<?> f : fs) f.get();
    }

    private float[] norms(float[] c, int nl) {
        float[] r = new float[nl];
        for (int j = 0; j < nl; j++) r[j] = Kernels.dot(c, j * dim, c, j * dim, dim);
        return r;
    }

    /** k-means over the first S rows of x: k-means++ seeding when S * nl * dim is at most 2e9, else distinct random rows; Lloyd iterations; deterministic for a seed and thread count independent */
    private float[] kmeans(float[] x, int S, int nl, ExecutorService pool) throws Exception {
        final int d = dim;
        Random rng = new Random(p.seed * 31L + nl);
        final float[] c = new float[nl * d];
        if ((long) S * nl * d <= 2_000_000_000L) {
            final double[] md = new double[S];
            int first = rng.nextInt(S);
            System.arraycopy(x, first * d, c, 0, d);
            updateMin(x, S, c, 0, md, true, pool);
            for (int k = 1; k < nl; k++) {
                double tot = 0;
                for (int i = 0; i < S; i++) tot += md[i];
                int pick;
                if (tot <= 0) {
                    pick = rng.nextInt(S);
                } else {
                    double r = rng.nextDouble() * tot, acc = 0;
                    pick = S - 1;
                    for (int i = 0; i < S; i++) {
                        acc += md[i];
                        if (acc >= r) {
                            pick = i;
                            break;
                        }
                    }
                }
                System.arraycopy(x, pick * d, c, k * d, d);
                updateMin(x, S, c, k, md, false, pool);
            }
        } else {
            int[] perm = new int[S];
            for (int i = 0; i < S; i++) perm[i] = i;
            for (int k = 0; k < nl; k++) {
                int j = k + rng.nextInt(S - k);
                int tmp = perm[k];
                perm[k] = perm[j];
                perm[j] = tmp;
                System.arraycopy(x, perm[k] * d, c, k * d, d);
            }
        }
        final int[] asg = new int[S];
        int[] prev = new int[S];
        Arrays.fill(prev, -1);
        float[] cnv = norms(c, nl);
        int iters = Math.max(1, p.kmeansIters);
        for (int it = 0; it < iters; it++) {
            final float[] cnn = cnv;
            parFor(pool, S, (from, to) -> {
                for (int i = from; i < to; i++) {
                    float bs = Float.NEGATIVE_INFINITY;
                    int bj = 0;
                    for (int j = 0; j < nl; j++) {
                        float sc = 2f * Kernels.dot(x, i * d, c, j * d, d) - cnn[j];
                        if (sc > bs) {
                            bs = sc;
                            bj = j;
                        }
                    }
                    asg[i] = bj;
                }
            });
            if (it > 0 && Arrays.equals(asg, prev)) break;
            System.arraycopy(asg, 0, prev, 0, S);
            double[] sum = new double[nl * d];
            int[] cnt = new int[nl];
            for (int i = 0; i < S; i++) {
                int j = asg[i];
                cnt[j]++;
                int xo = i * d, so = j * d;
                for (int t = 0; t < d; t++) sum[so + t] += x[xo + t];
            }
            for (int j = 0; j < nl; j++) {
                if (cnt[j] == 0) {
                    System.arraycopy(x, rng.nextInt(S) * d, c, j * d, d); // an empty cluster is reseeded with a random sample row
                } else {
                    double inv = 1.0 / cnt[j];
                    for (int t = 0; t < d; t++) c[j * d + t] = (float) (sum[j * d + t] * inv);
                }
            }
            cnv = norms(c, nl);
        }
        return c;
    }

    private void updateMin(float[] x, int S, float[] c, int k, double[] md, boolean init, ExecutorService pool) throws Exception {
        final int d = dim;
        parFor(pool, S, (from, to) -> {
            for (int i = from; i < to; i++) {
                double dd = Kernels.l2sq(x, i * d, c, k * d, d);
                if (init || dd < md[i]) md[i] = dd;
            }
        });
    }

    private AsgAll assignAll(float[] data, int n, float[] c, float[] cn, int nl, boolean second, ExecutorService pool) throws Exception {
        final AsgAll r = new AsgAll();
        r.best = new int[n];
        r.sec = second ? new int[n] : null;
        r.dist = new float[n];
        final int d = dim;
        parFor(pool, n, (from, to) -> {
            for (int i = from; i < to; i++) {
                float bs = Float.NEGATIVE_INFINITY, ss = Float.NEGATIVE_INFINITY;
                int bj = 0, sj = -1;
                for (int j = 0; j < nl; j++) {
                    float sc = 2f * Kernels.dot(data, i * d, c, j * d, d) - cn[j];
                    if (sc > bs) {
                        ss = bs;
                        sj = bj;
                        bs = sc;
                        bj = j;
                    } else if (second && sc > ss) {
                        ss = sc;
                        sj = j;
                    }
                }
                r.best[i] = bj;
                if (second) r.sec[i] = sj == bj ? -1 : sj;
                float dd = Kernels.dot(data, i * d, data, i * d, d) - bs;
                r.dist[i] = dd > 0f ? dd : 0f;
            }
        });
        return r;
    }

    /** builds a complete model (centroids, lists, directory) from n vectors in stored form; touches no live state */
    private Model buildModel(long[] ids, float[] data, int n, float[] fixedCent) throws Exception {
        if (n == 0 && fixedCent == null) return flatModel();
        ExecutorService pool = newPool();
        try {
            int nl;
            float[] c;
            if (fixedCent != null) {
                c = fixedCent;
                nl = c.length / dim;
            } else {
                int want = p.nlist > 0 ? p.nlist : (int) Math.round(4.0 * Math.sqrt(n));
                nl = Math.max(1, Math.min(want, n));
                int sN = (int) Math.min((long) n, Math.min((long) p.trainSample, Math.max(64L * nl, 10000L)));
                sN = Math.max(sN, nl);
                float[] samp;
                if (sN >= n) {
                    samp = data;
                    sN = n;
                } else {
                    int[] perm = new int[n];
                    for (int i = 0; i < n; i++) perm[i] = i;
                    Random rng = new Random(p.seed ^ 0x9E3779B97F4A7C15L);
                    samp = new float[sN * dim];
                    for (int k = 0; k < sN; k++) {
                        int j = k + rng.nextInt(n - k);
                        int tmp = perm[k];
                        perm[k] = perm[j];
                        perm[j] = tmp;
                        System.arraycopy(data, perm[k] * dim, samp, k * dim, dim);
                    }
                }
                c = kmeans(samp, sN, nl, pool);
            }
            float[] cn = norms(c, nl);
            boolean two = p.overlap > 1 && nl > 1;
            AsgAll a = assignAll(data, n, c, cn, nl, two, pool);
            int[] cnt = new int[nl];
            double dsum = 0;
            for (int i = 0; i < n; i++) {
                cnt[a.best[i]]++;
                if (two && a.sec[i] >= 0) cnt[a.sec[i]]++;
                dsum += a.dist[i];
            }
            boolean codes = p.quant == Quant.SQ8;
            Lst[] lists = new Lst[nl];
            for (int j = 0; j < nl; j++) lists[j] = new Lst(dim, codes, l2, cnt[j] + (cnt[j] >> 2) + 16);
            Model m = new Model(nl, c, cn, lists, newDirs(), p.overlap > 1 ? newDirs() : null, codes, dsum / n, n);
            for (int i = 0; i < n; i++) {
                long id = ids[i];
                int st = stripeOf(id);
                m.d1[st].put(id, a.best[i]);
                lists[a.best[i]].append(id, data, i * dim);
                if (two && a.sec[i] >= 0) {
                    m.d2[st].put(id, a.sec[i]);
                    lists[a.sec[i]].append(id, data, i * dim);
                }
            }
            return m;
        } finally {
            closePool(pool);
        }
    }

    // ---------------------------------------------------------------- memory

    @Override
    public long bytesHeld() {
        Model m = model;
        long b = 512;
        for (Lst l : m.lists) b += 64 + l.seg.bytes(dim);
        if (m.cent != null) b += 4L * m.cent.length + 4L * m.cn.length;
        for (int i = 0; i < STRIPES; i++) {
            stripes[i].lock();
            try {
                b += m.d1[i].bytes();
                if (m.d2 != null) b += m.d2[i].bytes();
            } finally {
                stripes[i].unlock();
            }
        }
        return b;
    }
}
