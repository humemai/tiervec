package com.humemai.vec;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.Arrays;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.LongPredicate;

/**
 * Graph tier: a concurrent, mutable HNSW-style proximity graph (written from Malkov and Yashunin, with the alpha robust prune of
 * Subramanya et al. / Vamana as a parameter and the Vamana delete-consolidation idea).
 *
 * DESIGN CHOICES (see also the report):
 *  - Multi-layer HNSW rather than single-layer Vamana: with inserts arriving one at a time and concurrently there is no stable medoid,
 *    while the HNSW entry point (the highest-level node) is maintained incrementally. alpha = 1 is the HNSW heuristic; alpha > 1 is the
 *    Vamana robust prune (distances are squared for L2, so alpha is squared there).
 *  - Nodes live in ordinal slots in fixed-size chunks (never moved). Vectors are immutable once published. A replace of an id is a new
 *    node plus a tombstone of the old one.
 *  - Neighbor lists are copy-on-write int[] arrays published with release/acquire VarHandles: readers are lock-free and always see a
 *    complete list. Writers take a striped monitor of the owning node (one at a time, never nested: no deadlock).
 *  - Visibility protocol of add(): vector, id, empty link arrays, state INIT. Select neighbors per layer (the node is invisible: nothing
 *    links to it). Publish own forward lists, then state LIVE (release), then reverse links (each a release store). Entry point updated
 *    last. A node is returnable only when LIVE and the only way to reach it is a release-published link, so a searcher never sees a
 *    half-linked node. Before add() returns, the layer-0 in-degree is checked and, when pruning dropped every reverse link, one is forced:
 *    read-your-writes. Layer-0 prune keeps a neighbor whose only in-edge is the list being pruned (in-degree counter).
 *  - delete(): map removal + state LIVE -> DELETED (immediate for readers). DELETED nodes are still traversed.
 *  - consolidate(): repairs every live node's lists that hold a tombstone (candidates = survivors plus the tombstones' survivors, then
 *    prune), marks the tombstones REMOVED, repairs orphans (BFS from the entry), waits in the background thread (never blocking a
 *    reader) until searches that started earlier have finished (epoch), then recycles the slots. Searches never lock.
 *  - Search: beam search at layer 0 (ef), greedy descent above. Filtered-out and deleted nodes are traversed but not collected. Stop rules:
 *    (1) frontier head worse than the worst of ef collected nodes, or frontier empty; (2) with a filter, after a visit budget of
 *    max(50 ef, n/20) nodes with fewer than k collected, fall back to an exact scan over the nodes that pass the filter.
 *  - Optional SQ8 (Params.sq8): per-vector min/max scalar quantization (signed byte codes; x ~ base + delta * code). Traversal distances
 *    use the codes; the ef collected candidates are re-scored in fp32 at the end. The fp32 vectors stay in memory (bytesHeld counts both),
 *    construction always uses fp32.
 *  - Cosine: vectors are normalized at add and the query at search; internal distance is 1 - dot. DOT: -dot. L2: squared L2.
 */
public final class GraphIndex implements VecIndex {

    public record Params(int M, int efConstruction, int efSearch, float alpha, boolean sq8, long seed) {
        public static Params defaults() { return new Params(16, 200, 100, 1.0f, false, 42L); }
        public Params withSq8(boolean b) { return new Params(M, efConstruction, efSearch, alpha, b, seed); }
        public Params withEfConstruction(int e) { return new Params(M, e, efSearch, alpha, sq8, seed); }
        public Params withM(int m) { return new Params(m, efConstruction, efSearch, alpha, sq8, seed); }
        public Params withAlpha(float a) { return new Params(M, efConstruction, efSearch, a, sq8, seed); }
    }

    static final int CHUNK_SHIFT = 10, CHUNK = 1 << CHUNK_SHIFT, MASK = CHUNK - 1;
    static final int INIT = 0, LIVE = 1, DELETED = 2, REMOVED = 3, FREE = 4;
    static final int L2 = 0, COS = 1, DOT = 2;
    static final int[] EMPTY = new int[0];
    static final int[][] EMPTY_LINKS = new int[][] { EMPTY };

    private static final VarHandle VH_OUTER = MethodHandles.arrayElementVarHandle(int[][][].class);
    private static final VarHandle VH_LAYERS = MethodHandles.arrayElementVarHandle(int[][].class);
    private static final VarHandle VH_INT = MethodHandles.arrayElementVarHandle(int[].class);

    static final class Chunk {
        final float[] vec; final byte[] codes; final float[] qbase, qdelta, qnorm; final long[] ids;
        final int[][][] links = new int[CHUNK][][];
        final int[] state = new int[CHUNK];
        final int[] indeg = new int[CHUNK];
        Chunk(int dim, boolean sq8) {
            vec = new float[CHUNK * dim];
            codes = sq8 ? new byte[CHUNK * dim] : null;
            qbase = sq8 ? new float[CHUNK] : null;
            qdelta = sq8 ? new float[CHUNK] : null;
            qnorm = sq8 ? new float[CHUNK] : null;
            ids = new long[CHUNK];
        }
    }

    record Entry(int node, int level) {}

    static final class Scratch {
        int[] stamp = new int[1024]; int gen;
        final Heap cand = new Heap(), res = new Heap();
        volatile long epoch = Long.MAX_VALUE;
        boolean exhausted; int visited;
        // query context
        float[] q; int qoff; float sumq, qq; float[] qbuf;
        // result arrays (ascending distance)
        int[] outN = new int[256]; float[] outD = new float[256]; int outCount;
        // selection scratch
        int[] cn = new int[512]; float[] cd = new float[512]; int[] sel = new int[512]; boolean[] taken = new boolean[512];

        boolean visit(int o) {
            if (o >= stamp.length) stamp = Arrays.copyOf(stamp, Math.max(o + 1, stamp.length * 2));
            if (stamp[o] == gen) return false;
            stamp[o] = gen; return true;
        }
        void newGen() { if (++gen == 0) { Arrays.fill(stamp, 0); gen = 1; } }
        void ensureSel(int n) {
            if (cn.length < n) { cn = new int[n * 2]; cd = new float[n * 2]; sel = new int[n * 2]; taken = new boolean[n * 2]; }
        }
        void ensureOut(int n) { if (outN.length < n) { outN = new int[n * 2]; outD = new float[n * 2]; } }
    }

    private final int dim, mode, M, maxDeg0;
    private final Metric metric;
    private final Params p;
    private final double mL;
    private final float alphaEff;

    private volatile Chunk[] chunks = new Chunk[8];
    private final Object growLock = new Object();
    private final AtomicInteger nextOrd = new AtomicInteger();
    private final AtomicInteger live = new AtomicInteger();
    private final AtomicInteger tombstones = new AtomicInteger();
    private final AtomicLong seq = new AtomicLong();
    private final ConcurrentHashMap<Long, Integer> idMap = new ConcurrentHashMap<>();
    private final Object[] stripes = new Object[1024];
    private volatile Entry entry;
    private final Object entryLock = new Object();
    private int[] freeStack = new int[64]; private int freeN; private final Object freeLock = new Object();
    private final AtomicLong globalEpoch = new AtomicLong(1);
    private final ConcurrentLinkedQueue<Scratch> scratches = new ConcurrentLinkedQueue<>();
    private final ThreadLocal<Scratch> tl = ThreadLocal.withInitial(() -> { Scratch s = new Scratch(); scratches.add(s); return s; });
    private final ReentrantLock consolidateLock = new ReentrantLock();
    private int[] pendingRemoved = EMPTY; // guarded by consolidateLock

    public GraphIndex(int dim, Metric metric, Params p) {
        this.dim = dim; this.metric = metric; this.p = p; this.M = p.M(); this.maxDeg0 = 2 * p.M();
        this.mode = metric == Metric.L2 ? L2 : metric == Metric.COSINE ? COS : DOT;
        this.mL = 1.0 / Math.log(Math.max(2, p.M()));
        this.alphaEff = mode == L2 ? p.alpha() * p.alpha() : p.alpha();
        for (int i = 0; i < stripes.length; i++) stripes[i] = new Object();
    }
    public GraphIndex(int dim, Metric metric) { this(dim, metric, Params.defaults()); }

    @Override public int dim() { return dim; }
    @Override public Metric metric() { return metric; }
    @Override public int size() { return live.get(); }
    public int tombstoneCount() { return tombstones.get(); }
    public Params params() { return p; }

    private volatile double filterScanBelow;
    /**
     * Filtered searches whose allowed share is estimated at or below this fraction scan the allowed ids exactly instead of walking the graph
     * first (see FilterPlan). Default 0 = off: a filtered search then walks and falls back to the scan after its visit budget, as before.
     */
    public void filterScanBelow(double fraction) { FilterPlan.check(fraction); filterScanBelow = fraction; }
    public double filterScanBelow() { return filterScanBelow; }

    // ---------------------------------------------------------------- storage access

    private Chunk chunk(int o) {
        Chunk[] cs = chunks; int ci = o >>> CHUNK_SHIFT;
        Chunk c;
        if (ci < cs.length && (c = cs[ci]) != null) return c;
        synchronized (growLock) {
            cs = chunks;
            if (ci >= cs.length) { cs = Arrays.copyOf(cs, Math.max(cs.length * 2, ci + 1)); }
            if (cs[ci] == null) { cs[ci] = new Chunk(dim, p.sq8()); }
            chunks = cs;
            return cs[ci];
        }
    }
    private Object stripe(int o) { return stripes[(o * 0x9E3779B1) >>> 22]; }
    private static int stateOf(Chunk c, int off) { return (int) VH_INT.getAcquire(c.state, off); }
    private int stateOf(int o) { return stateOf(chunk(o), o & MASK); }
    private static int[] list(Chunk c, int off, int layer) {
        int[][] ls = (int[][]) VH_OUTER.getAcquire(c.links, off);
        if (ls == null || layer >= ls.length) return EMPTY;
        return (int[]) VH_LAYERS.getAcquire(ls, layer);
    }
    private int levelOf(int o) {
        Chunk c = chunk(o); int[][] ls = (int[][]) VH_OUTER.getAcquire(c.links, o & MASK);
        return ls == null ? 0 : ls.length - 1;
    }
    private void addIndeg(int v, int d) { VH_INT.getAndAdd(chunk(v).indeg, v & MASK, d); }
    private int indegOf(int v) { return (int) VH_INT.getAcquire(chunk(v).indeg, v & MASK); }

    private static boolean contains(int[] a, int v) { for (int x : a) if (x == v) return true; return false; }

    /** layer-0 list replacement with in-degree bookkeeping; caller holds the owner's stripe (or owns an unpublished node). */
    private void setList(Chunk c, int off, int layer, int[] old, int[] nu) {
        int[][] ls = (int[][]) VH_OUTER.getAcquire(c.links, off);
        VH_LAYERS.setRelease(ls, layer, nu);
        if (layer == 0) {
            for (int v : old) if (!contains(nu, v)) addIndeg(v, -1);
            for (int v : nu) if (!contains(old, v)) addIndeg(v, +1);
        }
    }

    // ---------------------------------------------------------------- epochs (grace period for slot recycling)

    private Scratch enter() {
        Scratch s = tl.get();
        for (;;) { long g = globalEpoch.get(); s.epoch = g; if (globalEpoch.get() == g) break; }
        return s;
    }
    private static void exit(Scratch s) { s.epoch = Long.MAX_VALUE; }

    // ---------------------------------------------------------------- distances (lower is better)

    private float distFp(float[] q, int qo, Chunk c, int off) {
        int vo = off * dim;
        switch (mode) {
            case L2: return Kern.l2(q, qo, c.vec, vo, dim);
            case COS: return 1f - Kern.dot(q, qo, c.vec, vo, dim);
            default: return -Kern.dot(q, qo, c.vec, vo, dim);
        }
    }
    private float distQ8(Scratch s, Chunk c, int off) {
        float dv = c.qbase[off] * s.sumq + c.qdelta[off] * Kern.dotQ8(s.q, c.codes, off * dim, dim);
        return mode == L2 ? s.qq - 2f * dv + c.qnorm[off] : mode == COS ? 1f - dv : -dv;
    }
    private float distC(Scratch s, Chunk c, int off, boolean q8) { return q8 ? distQ8(s, c, off) : distFp(s.q, s.qoff, c, off); }
    private float distNodes(int a, int b) {
        Chunk ca = chunk(a), cb = chunk(b);
        return distFp(ca.vec, (a & MASK) * dim, cb, b & MASK);
    }

    private void prepareQuery(Scratch s, float[] q) {
        if (q.length != dim) throw new IllegalArgumentException("dim " + q.length + " != " + dim);
        if (s.qbuf == null) s.qbuf = new float[dim];
        System.arraycopy(q, 0, s.qbuf, 0, dim);
        if (mode == COS) normalize(s.qbuf);
        s.q = s.qbuf; s.qoff = 0;
        s.sumq = Kern.sum(s.qbuf, dim);
        s.qq = Kern.dot(s.qbuf, 0, s.qbuf, 0, dim);
    }
    private static void normalize(float[] v) {
        double n = 0; for (float x : v) n += (double) x * x;
        if (n > 0) { float inv = (float) (1.0 / Math.sqrt(n)); for (int i = 0; i < v.length; i++) v[i] *= inv; }
    }

    // ---------------------------------------------------------------- add

    private int allocOrdinal() {
        synchronized (freeLock) { if (freeN > 0) return freeStack[--freeN]; }
        return nextOrd.getAndIncrement();
    }
    private int levelFor() {
        long z = p.seed() + seq.getAndIncrement() * 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L; z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL; z ^= z >>> 31;
        double u = ((z >>> 11) + 1) * 0x1.0p-53; // (0,1]
        return Math.min(15, (int) (-Math.log(u) * mL));
    }

    @Override public void add(long id, float[] v) {
        if (v.length != dim) throw new IllegalArgumentException("dim " + v.length + " != " + dim);
        Scratch s = enter();
        try {
            int o = allocOrdinal();
            Chunk c = chunk(o); int off = o & MASK;
            VH_INT.setRelease(c.state, off, INIT);
            VH_INT.setRelease(c.indeg, off, 0);
            writeNode(c, off, id, v);
            int level = levelFor();
            int[][] ls = new int[level + 1][];
            Arrays.fill(ls, EMPTY);
            VH_OUTER.setRelease(c.links, off, ls);
            insert(s, o, c, off, level);
            live.incrementAndGet();
            Integer old = idMap.put(id, o);
            if (old != null) tombstone(old);
        } finally { exit(s); }
    }

    private void writeNode(Chunk c, int off, long id, float[] v) {
        int vo = off * dim;
        System.arraycopy(v, 0, c.vec, vo, dim);
        if (mode == COS) {
            double n = 0; for (int i = 0; i < dim; i++) n += (double) v[i] * v[i];
            if (n > 0) { float inv = (float) (1.0 / Math.sqrt(n)); for (int i = 0; i < dim; i++) c.vec[vo + i] *= inv; }
        }
        c.ids[off] = id;
        if (p.sq8()) {
            float lo = Float.POSITIVE_INFINITY, hi = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < dim; i++) { float x = c.vec[vo + i]; if (x < lo) lo = x; if (x > hi) hi = x; }
            float delta = (hi - lo) / 255f; if (!(delta > 0)) delta = 1f;
            float base = lo + 128f * delta; double nn = 0;
            for (int i = 0; i < dim; i++) {
                int code = Math.round((c.vec[vo + i] - lo) / delta); if (code < 0) code = 0; if (code > 255) code = 255;
                c.codes[vo + i] = (byte) (code - 128);
                float xr = base + delta * (code - 128); nn += (double) xr * xr;
            }
            c.qbase[off] = base; c.qdelta[off] = delta; c.qnorm[off] = (float) nn;
        }
    }

    private void insert(Scratch s, int o, Chunk c, int off, int level) {
        Entry e = entry;
        if (e == null) {
            synchronized (entryLock) {
                if (entry == null) {
                    VH_INT.setRelease(c.state, off, LIVE);
                    entry = new Entry(o, level);
                    return;
                }
                e = entry;
            }
        }
        s.q = c.vec; s.qoff = off * dim;
        int cur = e.node;
        float curD = distFp(s.q, s.qoff, chunk(cur), cur & MASK);
        for (int l = e.level; l > level; l--) cur = greedy(s, cur, curD, l, false);
        int top = Math.min(level, e.level);
        int[][] fwd = new int[level + 1][];
        for (int l = 0; l <= level; l++) fwd[l] = EMPTY;
        int[] firstCand0 = EMPTY;
        for (int l = top; l >= 0; l--) {
            float d0 = distFp(s.q, s.qoff, chunk(cur), cur & MASK);
            beam(s, cur, d0, p.efConstruction(), l, null, false, 0);
            int n = s.outCount; // ascending
            s.ensureSel(n + 1);
            int m = 0;
            for (int i = 0; i < n; i++) { if (s.outN[i] != o) { s.cn[m] = s.outN[i]; s.cd[m] = s.outD[i]; m++; } }
            if (m > 0) cur = s.cn[0];
            int k = select(s, s.cn, s.cd, m, M, false, s.sel);
            fwd[l] = Arrays.copyOf(s.sel, k);
            if (l == 0) { firstCand0 = Arrays.copyOf(s.cn, Math.min(m, 8)); }
        }
        for (int l = 0; l <= level; l++) setList(c, off, l, EMPTY, fwd[l]);
        VH_INT.setRelease(c.state, off, LIVE);
        for (int l = 0; l <= level; l++) for (int x : fwd[l]) addReverse(s, x, l, o);
        if (indegOf(o) <= 0) {
            for (int x : firstCand0) { if (stateOf(x) == LIVE && x != o) { forceInto(x, o); break; } }
        }
        if (level > e.level) {
            synchronized (entryLock) { Entry cu = entry; if (cu == null || level > cu.level) entry = new Entry(o, level); }
        }
    }

    /** HNSW neighbor-selection heuristic with the alpha robust prune; candidates ascending by distance to the base. */
    private int select(Scratch s, int[] cn, float[] cd, int n, int maxM, boolean fill, int[] out) {
        int m = 0;
        Arrays.fill(s.taken, 0, n, false);
        for (int i = 0; i < n && m < maxM; i++) {
            int c = cn[i]; boolean ok = true;
            for (int j = 0; j < m; j++) {
                float d = distNodes(c, out[j]);
                if ((d >= 0 ? d * alphaEff : d) <= cd[i]) { ok = false; break; }
            }
            if (ok) { out[m++] = c; s.taken[i] = true; }
        }
        if (fill) for (int i = 0; i < n && m < maxM; i++) if (!s.taken[i]) out[m++] = cn[i];
        return m;
    }

    private void addReverse(Scratch s, int x, int layer, int o) {
        Chunk cx = chunk(x); int ox = x & MASK;
        int maxDeg = layer == 0 ? maxDeg0 : M;
        synchronized (stripe(x)) {
            int[][] ls = (int[][]) VH_OUTER.getAcquire(cx.links, ox);
            if (ls == null || layer >= ls.length) return; // x was recycled under us: nothing to do
            int[] cur = (int[]) VH_LAYERS.getAcquire(ls, layer);
            if (contains(cur, o)) return;
            if (cur.length < maxDeg) {
                int[] nu = Arrays.copyOf(cur, cur.length + 1); nu[cur.length] = o;
                setList(cx, ox, layer, cur, nu);
                return;
            }
            int n = 0; s.ensureSel(maxDeg + 2);
            for (int i = 0; i <= cur.length; i++) {
                int v = i < cur.length ? cur[i] : o;
                int st = stateOf(v);
                if (st >= REMOVED || st == INIT && v != o) continue;
                float d = distNodes(x, v);
                int j = n++;
                while (j > 0 && s.cd[j - 1] > d) { s.cd[j] = s.cd[j - 1]; s.cn[j] = s.cn[j - 1]; j--; }
                s.cd[j] = d; s.cn[j] = v;
            }
            int m = select(s, s.cn, s.cd, n, maxDeg, true, s.sel);
            if (layer == 0) protect(s, cur, n, m);
            int[] nu = Arrays.copyOf(s.sel, m);
            setList(cx, ox, layer, cur, nu);
        }
    }

    /** keep an old neighbor that has no other in-edge by swapping it for a selected neighbor with in-degree > 1 */
    private void protect(Scratch s, int[] oldList, int n, int m) {
        for (int di = 0; di < oldList.length; di++) {
            int d = oldList[di];
            if (contains(s.sel, 0, m, d)) continue;
            if (stateOf(d) >= REMOVED || indegOf(d) > 1) continue;
            for (int j = m - 1; j >= 0; j--) {
                if (indegOf(s.sel[j]) > 1) { s.sel[j] = d; break; }
            }
        }
    }
    private static boolean contains(int[] a, int from, int to, int v) { for (int i = from; i < to; i++) if (a[i] == v) return true; return false; }

    /** make o reachable through x's layer-0 list (replace an entry whose target has other in-edges, else exceed the cap by one) */
    private void forceInto(int x, int o) {
        Chunk cx = chunk(x); int ox = x & MASK;
        synchronized (stripe(x)) {
            int[][] ls = (int[][]) VH_OUTER.getAcquire(cx.links, ox);
            if (ls == null) return;
            int[] cur = (int[]) VH_LAYERS.getAcquire(ls, 0);
            if (contains(cur, o)) return;
            int[] nu;
            if (cur.length < maxDeg0) { nu = Arrays.copyOf(cur, cur.length + 1); nu[cur.length] = o; }
            else {
                int r = -1;
                for (int j = cur.length - 1; j >= 0; j--) if (indegOf(cur[j]) > 1) { r = j; break; }
                if (r >= 0) { nu = cur.clone(); nu[r] = o; }
                else { nu = Arrays.copyOf(cur, cur.length + 1); nu[cur.length] = o; }
            }
            setList(cx, ox, 0, cur, nu);
        }
    }

    // ---------------------------------------------------------------- delete

    private void tombstone(int o) {
        Chunk c = chunk(o);
        if (VH_INT.compareAndSet(c.state, o & MASK, LIVE, DELETED)) { live.decrementAndGet(); tombstones.incrementAndGet(); }
    }

    @Override public boolean delete(long id) {
        Integer o = idMap.remove(id);
        if (o == null) return false;
        tombstone(o);
        return true;
    }

    // ---------------------------------------------------------------- search

    private int greedy(Scratch s, int cur, float curD, int layer, boolean q8) {
        for (boolean moved = true; moved; ) {
            moved = false;
            Chunk cc = chunk(cur);
            for (int x : list(cc, cur & MASK, layer)) {
                Chunk cx = chunk(x); int ox = x & MASK;
                int st = stateOf(cx, ox);
                if (st == INIT || st >= REMOVED) continue;
                float d = distC(s, cx, ox, q8);
                if (d < curD) { curD = d; cur = x; moved = true; cc = cx; }
            }
        }
        return cur;
    }

    /**
     * Beam search at one layer. Collected set: LIVE nodes that pass allow (ef best). Traversal goes through every other node.
     * Fills s.outN/outD ascending. budget > 0 only with a filter: visit limit, sets s.exhausted.
     */
    private void beam(Scratch s, int entryNode, float entryDist, int ef, int layer, LongPredicate allow, boolean q8, int budget) {
        s.newGen(); s.cand.clear(); s.res.clear(); s.exhausted = false; int visited = 0;
        Heap cand = s.cand, res = s.res;
        s.visit(entryNode);
        {
            Chunk ce = chunk(entryNode); int oe = entryNode & MASK;
            cand.push(entryDist, entryNode);
            if (stateOf(ce, oe) == LIVE && (allow == null || allow.test(ce.ids[oe]))) res.push(-entryDist, entryNode);
        }
        while (cand.n > 0) {
            float cdist = cand.topKey(); int cn = cand.topVal(); cand.pop();
            if (res.n >= ef && cdist > -res.topKey()) break;
            if (budget > 0 && visited > budget) { s.exhausted = true; break; }
            Chunk cc = chunk(cn);
            int[] nb = list(cc, cn & MASK, layer);
            for (int x : nb) {
                if (!s.visit(x)) continue;
                Chunk cx = chunk(x); int ox = x & MASK;
                int st = stateOf(cx, ox);
                if (st == INIT || st >= REMOVED) continue;
                visited++;
                float d = distC(s, cx, ox, q8);
                if (res.n < ef || d < -res.topKey()) {
                    cand.push(d, x);
                    if (st == LIVE && (allow == null || allow.test(cx.ids[ox]))) {
                        res.push(-d, x);
                        if (res.n > ef) res.pop();
                    }
                }
            }
        }
        s.visited = visited;
        int n = res.n; s.ensureOut(n); s.outCount = n;
        for (int i = n - 1; i >= 0; i--) { s.outD[i] = -res.topKey(); s.outN[i] = res.topVal(); res.pop(); }
    }

    @Override public Hit[] search(float[] q, int k, LongPredicate allow) { return search(q, k, allow, Math.max(k, p.efSearch())); }

    public Hit[] search(float[] q, int k, LongPredicate allow, int ef) { return search(q, k, allow, ef, filterScanBelow); }

    /** As {@link #search(float[], int, LongPredicate, int)} with the scan-first threshold of this call (see {@link #filterScanBelow(double)}). */
    public Hit[] search(float[] q, int k, LongPredicate allow, int ef, double scanBelow) {
        FilterPlan.check(scanBelow);
        if (k <= 0) return new Hit[0];
        Scratch s = enter();
        try {
            prepareQuery(s, q);
            Entry e = entry;
            if (e == null) return new Hit[0];
            ef = Math.max(ef, k);
            boolean q8 = p.sq8();
            int cur = e.node;
            float curD = distC(s, chunk(cur), cur & MASK, q8);
            for (int l = e.level; l >= 1; l--) { cur = greedy(s, cur, curD, l, q8); curD = distC(s, chunk(cur), cur & MASK, q8); }
            if (allow != null && scanBelow > 0
                    && FilterPlan.scanFirst(scanBelow, nextOrd.get(), o -> stateOf(chunk(o), o & MASK) == LIVE, o -> chunk(o).ids[o & MASK], allow)) {
                return scanExact(s, k, allow);
            }
            int budget = allow == null ? 0 : Math.max(50 * ef, nextOrd.get() / 20);
            beam(s, cur, curD, ef, 0, allow, q8, budget);
            if (allow != null && s.exhausted && s.outCount < ef) return scanExact(s, k, allow);
            return collect(s, k, q8);
        } finally { exit(s); }
    }

    private Hit[] collect(Scratch s, int k, boolean q8) {
        int n = s.outCount;
        if (q8) { // fp32 rerank of the collected candidates
            s.q = s.qbuf; s.qoff = 0;
            for (int i = 0; i < n; i++) {
                int o = s.outN[i]; float d = distFp(s.q, 0, chunk(o), o & MASK);
                int j = i;
                while (j > 0 && s.outD[j - 1] > d) { s.outD[j] = s.outD[j - 1]; s.outN[j] = s.outN[j - 1]; j--; }
                s.outD[j] = d; s.outN[j] = o;
            }
        }
        Hit[] out = new Hit[Math.min(k, n)]; int m = 0;
        for (int i = 0; i < n && m < k; i++) {
            int o = s.outN[i]; Chunk c = chunk(o); int off = o & MASK;
            if (stateOf(c, off) != LIVE) continue; // a delete that returned before this search started must be honoured
            long id = c.ids[off];
            boolean dup = false; for (int j = 0; j < m; j++) if (out[j].id() == id) { dup = true; break; }
            if (dup) continue;
            out[m++] = new Hit(id, score(s.outD[i]));
        }
        return m == out.length ? out : Arrays.copyOf(out, m);
    }
    private float score(float dist) { return mode == COS ? 1f - dist : -dist; }

    private Hit[] scanExact(Scratch s, int k, LongPredicate allow) {
        Heap h = s.res; h.clear();
        s.q = s.qbuf; s.qoff = 0;
        int hi = nextOrd.get();
        for (int o = 0; o < hi; o++) {
            Chunk c = chunk(o); int off = o & MASK;
            if (stateOf(c, off) != LIVE || !allow.test(c.ids[off])) continue;
            float d = distFp(s.q, 0, c, off);
            if (h.n < k) h.push(-d, o); else if (d < -h.topKey()) { h.pop(); h.push(-d, o); }
        }
        int n = h.n; s.ensureOut(n); s.outCount = n;
        for (int i = n - 1; i >= 0; i--) { s.outD[i] = -h.topKey(); s.outN[i] = h.topVal(); h.pop(); }
        return collect(s, k, false);
    }

    // ---------------------------------------------------------------- consolidate

    /** Bit i of b. An index past the end reads as not set: a pass sizes its bitsets from a snapshot of nextOrd, and a node allocated after the snapshot has no bit in them. */
    private static boolean bit(long[] b, int i) { int w = i >>> 6; return w < b.length && (b[w] >>> (i & 63) & 1L) != 0; }
    private static void setBit(long[] b, int i) { b[i >>> 6] |= 1L << (i & 63); }

    /**
     * Remove tombstoned nodes and repair their neighbors. Safe to run in a background thread while searches, adds and deletes continue:
     * lists are replaced copy-on-write, readers never lock, slots are recycled only after every operation that started before the
     * tombstones were unlinked has finished (this thread waits, at most 10 s; on timeout the recycle is deferred to the next call).
     * @return number of slots recycled
     */
    public int consolidate() {
        consolidateLock.lock();
        try {
            Scratch s = tl.get(); // heaps and stamps only; this thread is never inside an epoch while waiting
            int hi = nextOrd.get();
            long[] dead = new long[(hi + 63) / 64];
            int[] deadList = new int[Math.min(hi, 1 << 20)]; int nd = 0;
            for (int o = 0; o < hi; o++) {
                int st = stateOf(o);
                if (st == DELETED || st == REMOVED) { setBit(dead, o); if (nd == deadList.length) deadList = Arrays.copyOf(deadList, nd * 2); deadList[nd++] = o; }
            }
            if (nd == 0) { repairOrphans(s, hi); return 0; }
            // 1. repair every live node that points at a tombstone
            for (int o = 0; o < hi; o++) {
                Chunk c = chunk(o); int off = o & MASK;
                if (stateOf(c, off) != LIVE) continue;
                int levels = levelOf(o);
                for (int l = 0; l <= levels; l++) repairNode(s, o, c, off, l, dead);
            }
            // 2. new entry before any tombstone becomes unusable
            Entry e = entry;
            if (e != null && bit(dead, e.node)) {
                int best = -1, bl = -1;
                for (int o = 0; o < hi; o++) if (!bit(dead, o) && stateOf(o) == LIVE) { int lv = levelOf(o); if (lv > bl) { bl = lv; best = o; } }
                synchronized (entryLock) { Entry cu = entry; if (cu != null && bit(dead, cu.node)) entry = best < 0 ? null : new Entry(best, bl); }
            }
            // 3. mark removed
            for (int i = 0; i < nd; i++) { int o = deadList[i]; Chunk c = chunk(o); if (VH_INT.compareAndSet(c.state, o & MASK, DELETED, REMOVED)) tombstones.decrementAndGet(); }
            // 4. orphans (nodes no list reaches from the entry)
            repairOrphans(s, hi);
            // 5. grace period, background thread only
            long target = globalEpoch.incrementAndGet();
            long deadline = System.nanoTime() + 10_000_000_000L;
            boolean drained = false;
            while (System.nanoTime() < deadline) {
                boolean all = true;
                for (Scratch x : scratches) if (x.epoch < target) { all = false; break; }
                if (all) { drained = true; break; }
                try { Thread.sleep(1); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); break; }
            }
            if (!drained) { pendingRemoved = Arrays.copyOf(deadList, nd); return 0; }
            // 6. recycle
            for (int i = 0; i < nd; i++) {
                int o = deadList[i]; Chunk c = chunk(o); int off = o & MASK;
                if (stateOf(c, off) != REMOVED) continue;
                for (int v : list(c, off, 0)) addIndeg(v, -1);
                VH_OUTER.setRelease(c.links, off, EMPTY_LINKS);
                VH_INT.setRelease(c.state, off, FREE);
            }
            synchronized (freeLock) {
                if (freeStack.length < freeN + nd) freeStack = Arrays.copyOf(freeStack, Math.max(freeStack.length * 2, freeN + nd));
                for (int i = 0; i < nd; i++) { int o = deadList[i]; if (stateOf(o) == FREE) freeStack[freeN++] = o; }
            }
            pendingRemoved = EMPTY;
            return nd;
        } finally { consolidateLock.unlock(); }
    }

    private void repairNode(Scratch s, int x, Chunk cx, int ox, int layer, long[] dead) {
        int maxDeg = layer == 0 ? maxDeg0 : M;
        synchronized (stripe(x)) {
            int[][] ls = (int[][]) VH_OUTER.getAcquire(cx.links, ox);
            if (ls == null || layer >= ls.length || stateOf(cx, ox) != LIVE) return;
            int[] cur = (int[]) VH_LAYERS.getAcquire(ls, layer);
            boolean has = false;
            for (int v : cur) if (v < dead.length * 64 && bit(dead, v)) { has = true; break; }
            if (!has) return;
            s.newGen(); s.visit(x);
            int n = 0; int cap = 0;
            for (int v : cur) cap += 1 + (v < dead.length * 64 && bit(dead, v) ? list(chunk(v), v & MASK, layer).length : 0);
            s.ensureSel(cap + 2);
            for (int v : cur) {
                boolean dv = v < dead.length * 64 && bit(dead, v);
                if (!dv) { if (s.visit(v)) { s.cn[n] = v; n++; } continue; }
                for (int w : list(chunk(v), v & MASK, layer)) {
                    if (w < dead.length * 64 && bit(dead, w)) continue;
                    int st = stateOf(w);
                    if (st != LIVE) continue;
                    if (s.visit(w)) { s.cn[n] = w; n++; }
                }
            }
            for (int i = 0; i < n; i++) s.cd[i] = distNodes(x, s.cn[i]);
            // sort ascending (insertion sort on parallel arrays; n is small, a few hundred at most)
            for (int i = 1; i < n; i++) {
                float d = s.cd[i]; int v = s.cn[i]; int j = i;
                while (j > 0 && s.cd[j - 1] > d) { s.cd[j] = s.cd[j - 1]; s.cn[j] = s.cn[j - 1]; j--; }
                s.cd[j] = d; s.cn[j] = v;
            }
            int m;
            if (n <= maxDeg) { System.arraycopy(s.cn, 0, s.sel, 0, n); m = n; }
            else m = select(s, s.cn, s.cd, n, maxDeg, true, s.sel);
            setList(cx, ox, layer, cur, Arrays.copyOf(s.sel, m));
        }
    }

    private void repairOrphans(Scratch s, int hi) {
        Entry e = entry; if (e == null) return;
        int lim = Math.max(hi, e.node + 1); // the entry point can be a node allocated after the snapshot hi
        long[] seen = new long[(lim + 63) / 64];
        int[] queue = new int[lim]; int qh = 0, qt = 0;
        queue[qt++] = e.node; setBit(seen, e.node);
        while (qh < qt) {
            int u = queue[qh++]; Chunk c = chunk(u);
            int st = stateOf(c, u & MASK);
            if (st == INIT || st >= REMOVED) continue;
            for (int v : list(c, u & MASK, 0)) if (v < hi && !bit(seen, v)) { setBit(seen, v); queue[qt++] = v; }
        }
        for (int o = 0; o < hi; o++) {
            if (bit(seen, o)) continue;
            Chunk c = chunk(o); int off = o & MASK;
            if (stateOf(c, off) != LIVE) continue;
            s.q = c.vec; s.qoff = off * dim;
            Entry ce = entry; if (ce == null) return;
            float d0 = distFp(s.q, s.qoff, chunk(ce.node), ce.node & MASK);
            int cur = ce.node;
            for (int l = ce.level; l >= 1; l--) cur = greedy(s, cur, d0, l, false);
            d0 = distFp(s.q, s.qoff, chunk(cur), cur & MASK);
            beam(s, cur, d0, Math.min(p.efConstruction(), 64), 0, null, false, 0);
            int linked = 0;
            for (int i = 0; i < s.outCount && linked < 2; i++) { int x = s.outN[i]; if (x != o && bit(seen, x)) { forceInto(x, o); linked++; } }
            if (linked > 0) { orphansRepaired++; }
        }
    }
    private volatile int orphansRepaired;
    public int orphansRepaired() { return orphansRepaired; }

    // ---------------------------------------------------------------- seal, memory

    /** Snapshot the live nodes into an immutable compact form. Runs consolidate() first so the lists carry no tombstones. */
    public SealedGraph seal() {
        consolidate();
        int hi = nextOrd.get();
        int[] map = new int[hi]; int m = 0;
        for (int o = 0; o < hi; o++) { if (stateOf(o) == LIVE) map[o] = m++; else map[o] = -1; }
        float[] vec = new float[m * dim];
        byte[] codes = p.sq8() ? new byte[m * dim] : null;
        float[] qb = p.sq8() ? new float[m] : null, qd = p.sq8() ? new float[m] : null, qn = p.sq8() ? new float[m] : null;
        long[] ids = new long[m]; byte[] level = new byte[m];
        int[] off0 = new int[m + 1]; int[] edges0 = new int[m * maxDeg0 / 2 + 16]; int ne = 0;
        int[] upIdx = new int[m]; Arrays.fill(upIdx, -1);
        int[] upOff = new int[64]; int nuo = 0; int[] upEdges = new int[64]; int nue = 0;
        for (int o = 0; o < hi; o++) {
            int r = map[o]; if (r < 0) continue;
            Chunk c = chunk(o); int off = o & MASK;
            System.arraycopy(c.vec, off * dim, vec, r * dim, dim);
            if (codes != null) { System.arraycopy(c.codes, off * dim, codes, r * dim, dim); qb[r] = c.qbase[off]; qd[r] = c.qdelta[off]; qn[r] = c.qnorm[off]; }
            ids[r] = c.ids[off];
            int lv = levelOf(o); level[r] = (byte) lv;
            off0[r] = ne;
            for (int v : list(c, off, 0)) if (v < hi && map[v] >= 0) { if (ne == edges0.length) edges0 = Arrays.copyOf(edges0, ne * 2); edges0[ne++] = map[v]; }
            if (lv >= 1) {
                upIdx[r] = nuo;
                if (nuo + lv + 2 > upOff.length) upOff = Arrays.copyOf(upOff, Math.max(upOff.length * 2, nuo + lv + 2));
                upOff[nuo++] = nue; // boundary before layer 1
                for (int l = 1; l <= lv; l++) {
                    for (int v : list(c, off, l)) if (v < hi && map[v] >= 0) { if (nue == upEdges.length) upEdges = Arrays.copyOf(upEdges, nue * 2); upEdges[nue++] = map[v]; }
                    upOff[nuo++] = nue;
                }
            }
        }
        off0[m] = ne;
        Entry e = entry; int er = -1, ml = 0;
        if (e != null && e.node < hi && map[e.node] >= 0) { er = map[e.node]; ml = level[er]; }
        else for (int r = 0; r < m; r++) if (er < 0 || level[r] > ml) { er = r; ml = level[r]; }
        return new SealedGraph(dim, metric, p, m, vec, codes, qb, qd, qn, ids, level, off0, Arrays.copyOf(edges0, ne), upIdx,
                Arrays.copyOf(upOff, nuo), Arrays.copyOf(upEdges, nue), er, ml);
    }

    /** Re-inflate a sealed graph into a mutable index (same lists, same levels; tombstoned rows dropped). */
    public static GraphIndex fromSealed(SealedGraph sg) {
        GraphIndex g = new GraphIndex(sg.dim, sg.metric, sg.params);
        int n = sg.n; int[] map = new int[n]; int m = 0;
        for (int r = 0; r < n; r++) map[r] = sg.isDead(r) ? -1 : m++;
        for (int r = 0; r < n; r++) {
            if (map[r] < 0) continue;
            int o = g.allocOrdinal(); Chunk c = g.chunk(o); int off = o & MASK;
            System.arraycopy(sg.vec, r * g.dim, c.vec, off * g.dim, g.dim);
            c.ids[off] = sg.ids[r];
            if (g.p.sq8() && sg.codes != null) {
                System.arraycopy(sg.codes, r * g.dim, c.codes, off * g.dim, g.dim);
                c.qbase[off] = sg.qbase[r]; c.qdelta[off] = sg.qdelta[r]; c.qnorm[off] = sg.qnorm[r];
            } else if (g.p.sq8()) { float[] tmp = Arrays.copyOfRange(sg.vec, r * g.dim, (r + 1) * g.dim); g.writeNode(c, off, sg.ids[r], tmp); }
            int lv = sg.level[r]; int[][] ls = new int[lv + 1][];
            ls[0] = remap(sg.edges0, sg.off0[r], sg.off0[r + 1], map);
            for (int l = 1; l <= lv; l++) { int b = sg.upIdx[r]; ls[l] = remap(sg.upEdges, sg.upOff[b + l - 1], sg.upOff[b + l], map); }
            VH_OUTER.setRelease(c.links, off, ls);
            VH_INT.setRelease(c.state, off, LIVE);
            g.idMap.put(sg.ids[r], o);
        }
        for (int o = 0; o < m; o++) for (int v : list(g.chunk(o), o & MASK, 0)) g.addIndeg(v, 1);
        g.live.set(m); g.nextOrd.set(m); // allocOrdinal already advanced nextOrd; keep consistent
        if (sg.n > 0 && sg.entryRow >= 0 && map[sg.entryRow] >= 0) g.entry = new Entry(map[sg.entryRow], g.levelOf(map[sg.entryRow]));
        else if (m > 0) { int best = 0; for (int o = 1; o < m; o++) if (g.levelOf(o) > g.levelOf(best)) best = o; g.entry = new Entry(best, g.levelOf(best)); }
        return g;
    }
    private static int[] remap(int[] src, int from, int to, int[] map) {
        int[] t = new int[to - from]; int k = 0;
        for (int i = from; i < to; i++) { int v = map[src[i]]; if (v >= 0) t[k++] = v; }
        return k == t.length ? t : Arrays.copyOf(t, k);
    }

    /** approximate heap bytes: chunk arrays, link arrays (walked), id map (about 80 bytes per entry) */
    @Override public long bytesHeld() {
        long b = 0; Chunk[] cs = chunks; int hi = nextOrd.get();
        for (Chunk c : cs) {
            if (c == null) continue;
            b += 16L + (long) c.vec.length * 4 + (c.codes != null ? c.codes.length + 3L * CHUNK * 4 : 0) + (long) CHUNK * 8
                    + (long) CHUNK * 4 /* links refs, compressed oops */ + (long) CHUNK * 4 * 2;
        }
        for (int o = 0; o < hi; o++) {
            Chunk c = chunk(o); int[][] ls = (int[][]) VH_OUTER.getAcquire(c.links, o & MASK);
            if (ls == null || ls == EMPTY_LINKS) continue;
            b += 16 + 4L * ls.length;
            for (int[] l : ls) if (l != EMPTY) b += 16 + ((4L * l.length + 7) & ~7L);
        }
        b += idMap.size() * 80L + 8L * (freeStack.length);
        return b;
    }

    int liveCountForTest() { return live.get(); }
    /**
     * Exact membership of an id, for tests: the id-to-ordinal map holds the id and the node it points to is LIVE and stores that id.
     * It does not search the graph, so an approximate-search miss cannot change the answer.
     */
    boolean containsForTest(long id) {
        Integer o = idMap.get(id);
        if (o == null) return false;
        Chunk c = chunk(o); int off = o & MASK;
        return stateOf(c, off) == LIVE && c.ids[off] == id;
    }
    int ordinalsAllocated() { return nextOrd.get(); }
}
