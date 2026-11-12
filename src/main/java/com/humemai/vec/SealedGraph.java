package com.humemai.vec;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLongArray;
import java.util.function.LongPredicate;

/**
 * Immutable, compact form of a graph (the settled bulk): flat vector array, CSR neighbor lists (layer 0 in off0/edges0, layers above in
 * upIdx/upOff/upEdges), optional SQ8 codes. Searches are lock-free. The only mutation is a soft delete through a deletion bitset
 * (like a deletion vector); add() is unsupported. Produced by GraphIndex.seal(); GraphIndex.fromSealed() re-inflates it.
 */
@Internal
public final class SealedGraph implements VecIndex {
    final int dim; final Metric metric; final GraphIndex.Params params; final int n;
    final float[] vec; final byte[] codes; final float[] qbase, qdelta, qnorm; final long[] ids; final byte[] level;
    final int[] off0, edges0, upIdx, upOff, upEdges; final int entryRow, maxLevel;
    private final int mode;
    final long[] sortedIds; final int[] sortedRows; // ids in increasing order, and the row of each (the BYID section of a segment file)
    private final AtomicLongArray dead; private final AtomicInteger live;
    private final ThreadLocal<Sc> tl = ThreadLocal.withInitial(Sc::new);

    SealedGraph(int dim, Metric metric, GraphIndex.Params params, int n, float[] vec, byte[] codes, float[] qbase, float[] qdelta, float[] qnorm,
                long[] ids, byte[] level, int[] off0, int[] edges0, int[] upIdx, int[] upOff, int[] upEdges, int entryRow, int maxLevel) {
        this(dim, metric, params, n, vec, codes, qbase, qdelta, qnorm, ids, level, off0, edges0, upIdx, upOff, upEdges, entryRow, maxLevel, sortRows(ids, n));
    }

    /** the same graph with the id order given: {@code byId[i]} is the row of the i-th smallest id. A segment file stores it, so a load skips the sort. */
    SealedGraph(int dim, Metric metric, GraphIndex.Params params, int n, float[] vec, byte[] codes, float[] qbase, float[] qdelta, float[] qnorm,
                long[] ids, byte[] level, int[] off0, int[] edges0, int[] upIdx, int[] upOff, int[] upEdges, int entryRow, int maxLevel, int[] byId) {
        this.dim = dim; this.metric = metric; this.params = params; this.n = n; this.vec = vec; this.codes = codes;
        this.qbase = qbase; this.qdelta = qdelta; this.qnorm = qnorm; this.ids = ids; this.level = level;
        this.off0 = off0; this.edges0 = edges0; this.upIdx = upIdx; this.upOff = upOff; this.upEdges = upEdges;
        this.entryRow = entryRow; this.maxLevel = maxLevel;
        this.mode = metric == Metric.L2 ? GraphIndex.L2 : metric == Metric.COSINE ? GraphIndex.COS : GraphIndex.DOT;
        sortedRows = byId; sortedIds = new long[n];
        for (int i = 0; i < n; i++) sortedIds[i] = ids[byId[i]];
        dead = new AtomicLongArray((n + 63) / 64); live = new AtomicInteger(n);
    }

    private static int[] sortRows(long[] ids, int n) {
        Integer[] order = new Integer[n]; for (int i = 0; i < n; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Long.compare(ids[a], ids[b]));
        int[] rows = new int[n]; for (int i = 0; i < n; i++) rows[i] = order[i];
        return rows;
    }

    /**
     * Writes this graph as one segment file (see docs/SEGMENT-FORMAT.md): fp32 vectors, the graph, the stored id order, SQ8 codes when
     * present, a CRC32C per section. The file is new (an existing one is never overwritten) and flushed before this returns. A graph with
     * soft-deleted rows is refused: this version stores no tombstones inside a segment.
     */
    public void serialize(Path file) throws IOException { SegmentWriter.write(this, null, file); }

    /** Reads a file written by {@link #serialize}. Every section is checksummed; any damage throws {@link SegmentFormat.CorruptException}. */
    public static SealedGraph load(Path file) throws IOException { return SegmentReader.read(file, true); }

    static final class Sc {
        int[] stamp = new int[0]; int gen; final Heap cand = new Heap(), res = new Heap();
        float[] q; float sumq, qq; int[] outN = new int[256]; float[] outD = new float[256]; int outCount; boolean exhausted;
        void prep(int n) { if (stamp.length < n) stamp = new int[n]; if (++gen == 0) { Arrays.fill(stamp, 0); gen = 1; } }
    }

    /** number of rows including soft-deleted ones (row indexes are 0 until rowCount) */
    public int rowCount() { return n; }
    /** id of a row */
    public long idAt(int row) { return ids[row]; }
    /** true when the row was soft-deleted */
    public boolean isDeleted(int row) { return isDead(row); }
    /** copy of the stored fp32 vector of a row (unit length when the metric is COSINE, because vectors are normalized at add) */
    public float[] rawVector(int row) { return Arrays.copyOfRange(vec, row * dim, (row + 1) * dim); }
    /** copy a row vector into dst starting at dstOff, no allocation */
    public void copyVector(int row, float[] dst, int dstOff) { System.arraycopy(vec, row * dim, dst, dstOff, dim); }
    boolean isDead(int r) { return (dead.get(r >>> 6) >>> (r & 63) & 1L) != 0; }
    @Override public int dim() { return dim; }
    @Override public Metric metric() { return metric; }
    @Override public int size() { return live.get(); }
    @Override public void add(long id, float[] v) { throw new UnsupportedOperationException("sealed graph is immutable; add to a GraphIndex or a flat delta"); }

    private int rowOf(long id) { int i = Arrays.binarySearch(sortedIds, id); return i < 0 ? -1 : sortedRows[i]; }

    @Override public boolean delete(long id) {
        int r = rowOf(id); if (r < 0) return false;
        int w = r >>> 6; long bit = 1L << (r & 63);
        for (;;) {
            long cur = dead.get(w);
            if ((cur & bit) != 0) return false;
            if (dead.compareAndSet(w, cur, cur | bit)) { live.decrementAndGet(); return true; }
        }
    }

    private float distFp(float[] q, int r) {
        int vo = r * dim;
        switch (mode) {
            case GraphIndex.L2: return Kern.l2(q, 0, vec, vo, dim);
            case GraphIndex.COS: return 1f - Kern.dot(q, 0, vec, vo, dim);
            default: return -Kern.dot(q, 0, vec, vo, dim);
        }
    }
    private float dist(Sc s, int r, boolean q8) {
        if (!q8) return distFp(s.q, r);
        float dv = qbase[r] * s.sumq + qdelta[r] * Kern.dotQ8(s.q, codes, r * dim, dim);
        return mode == GraphIndex.L2 ? s.qq - 2f * dv + qnorm[r] : mode == GraphIndex.COS ? 1f - dv : -dv;
    }

    @Override public Hit[] search(float[] q, int k, LongPredicate allow) { return search(q, k, allow, Math.max(k, params.efSearch())); }

    private volatile double filterScanBelow;
    /** Scan-first threshold for filtered searches, as {@link GraphIndex#filterScanBelow(double)}. Default 0 = off. */
    public void filterScanBelow(double fraction) { FilterPlan.check(fraction); filterScanBelow = fraction; }
    public double filterScanBelow() { return filterScanBelow; }

    public Hit[] search(float[] q, int k, LongPredicate allow, int ef) { return search(q, k, allow, ef, filterScanBelow); }

    /** As {@link #search(float[], int, LongPredicate, int)} with the scan-first threshold of this call. */
    public Hit[] search(float[] q, int k, LongPredicate allow, int ef, double scanBelow) {
        FilterPlan.check(scanBelow);
        if (k <= 0 || n == 0) return new Hit[0];
        if (q.length != dim) throw new IllegalArgumentException("dim");
        Sc s = tl.get(); ef = Math.max(ef, k);
        float[] qq = q.clone();
        if (mode == GraphIndex.COS) { double nn = 0; for (float x : qq) nn += (double) x * x; if (nn > 0) { float inv = (float) (1 / Math.sqrt(nn)); for (int i = 0; i < dim; i++) qq[i] *= inv; } }
        s.q = qq; s.sumq = Kern.sum(qq, dim); s.qq = Kern.dot(qq, 0, qq, 0, dim);
        boolean q8 = codes != null && params.sq8();
        int cur = entryRow; float curD = dist(s, cur, q8);
        for (int l = maxLevel; l >= 1; l--) {
            for (boolean moved = true; moved; ) {
                moved = false;
                int b = upIdx[cur]; if (b < 0) break;
                for (int i = upOff[b + l - 1]; i < upOff[b + l]; i++) {
                    int x = upEdges[i]; float d = dist(s, x, q8);
                    if (d < curD) { curD = d; cur = x; moved = true; }
                }
            }
        }
        s.prep(n);
        if (allow != null && scanBelow > 0 && FilterPlan.scanFirst(scanBelow, n, r -> !isDead(r), r -> ids[r], allow)) return scanExact(s, k, allow);
        int budget = allow == null ? 0 : Math.max(50 * ef, n / 20);
        beam(s, cur, curD, ef, allow, q8, budget);
        if (allow != null && s.exhausted && s.outCount < ef) return scanExact(s, k, allow);
        return collect(s, k, q8);
    }

    private void beam(Sc s, int entryNode, float entryDist, int ef, LongPredicate allow, boolean q8, int budget) {
        Heap cand = s.cand, res = s.res; cand.clear(); res.clear(); s.exhausted = false; int visited = 0;
        s.stamp[entryNode] = s.gen;
        cand.push(entryDist, entryNode);
        if (!isDead(entryNode) && (allow == null || allow.test(ids[entryNode]))) res.push(-entryDist, entryNode);
        while (cand.n > 0) {
            float cd = cand.topKey(); int c = cand.topVal(); cand.pop();
            if (res.n >= ef && cd > -res.topKey()) break;
            if (budget > 0 && visited > budget) { s.exhausted = true; break; }
            for (int i = off0[c], e = off0[c + 1]; i < e; i++) {
                int x = edges0[i];
                if (s.stamp[x] == s.gen) continue;
                s.stamp[x] = s.gen; visited++;
                float d = dist(s, x, q8);
                if (res.n < ef || d < -res.topKey()) {
                    cand.push(d, x);
                    if (!isDead(x) && (allow == null || allow.test(ids[x]))) { res.push(-d, x); if (res.n > ef) res.pop(); }
                }
            }
        }
        int m = res.n; if (s.outN.length < m) { s.outN = new int[m * 2]; s.outD = new float[m * 2]; } s.outCount = m;
        for (int i = m - 1; i >= 0; i--) { s.outD[i] = -res.topKey(); s.outN[i] = res.topVal(); res.pop(); }
    }

    private Hit[] collect(Sc s, int k, boolean q8) {
        int m = s.outCount;
        if (q8) for (int i = 0; i < m; i++) {
            int r = s.outN[i]; float d = distFp(s.q, r); int j = i;
            while (j > 0 && s.outD[j - 1] > d) { s.outD[j] = s.outD[j - 1]; s.outN[j] = s.outN[j - 1]; j--; }
            s.outD[j] = d; s.outN[j] = r;
        }
        Hit[] out = new Hit[Math.min(k, m)]; int c = 0;
        for (int i = 0; i < m && c < k; i++) {
            int r = s.outN[i]; if (isDead(r)) continue;
            out[c++] = new Hit(ids[r], mode == GraphIndex.COS ? 1f - s.outD[i] : -s.outD[i]);
        }
        return c == out.length ? out : Arrays.copyOf(out, c);
    }

    private Hit[] scanExact(Sc s, int k, LongPredicate allow) {
        Heap h = s.res; h.clear();
        for (int r = 0; r < n; r++) {
            if (isDead(r) || !allow.test(ids[r])) continue;
            float d = distFp(s.q, r);
            if (h.n < k) h.push(-d, r); else if (d < -h.topKey()) { h.pop(); h.push(-d, r); }
        }
        int m = h.n; if (s.outN.length < m) { s.outN = new int[m * 2]; s.outD = new float[m * 2]; } s.outCount = m;
        for (int i = m - 1; i >= 0; i--) { s.outD[i] = -h.topKey(); s.outN[i] = h.topVal(); h.pop(); }
        return collect(s, k, false);
    }

    @Override public long bytesHeld() {
        long b = (long) vec.length * 4 + (codes != null ? codes.length + 12L * n : 0) + 8L * n + n + 4L * (off0.length + edges0.length + upIdx.length + upOff.length + upEdges.length)
                + 12L * n + 8L * ((n + 63) / 64);
        return b;
    }
}
