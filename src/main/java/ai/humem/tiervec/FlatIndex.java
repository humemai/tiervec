package ai.humem.tiervec;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.zip.CRC32;
import java.util.zip.CheckedInputStream;
import java.util.zip.CheckedOutputStream;
import java.util.function.LongPredicate;

/**
 * Flat tier: exact or quantized scan over every live vector, no graph.
 *
 * <p>Layout. Vectors live in chunks (the first chunks double from 64 to 4096 slots so a tiny index costs little, then 4096 per chunk).
 * A chunk holds a flat float[] (cap * dim fp32, normalized when the metric is COSINE), a long[] of ids, a live bitmap, and the optional
 * codes (SQ8: byte[] codes plus a float scale per vector, plus the squared norm for L2; BINARY: long[] sign bits). A slot is written once
 * and never reused, so a reader never sees a half-rewritten slot. Appends never copy old data; the only copy is the small array of chunk
 * references when it grows.
 *
 * <p>Concurrency. The read path takes no lock: it reads the current Store (volatile), a slot counter, and for each 64 slots one live word
 * with acquire semantics; a writer fills the slot and then sets the live bit with release semantics, which is the publish. Delete clears
 * the bit (CAS). Replace writes a NEW slot, sets it live, then clears the old bit; a search can see both for a moment, so the final
 * results are de-duplicated by id. Writers take a stripe monitor (64 stripes by id hash, each with its own id-to-slot map, so writers of
 * different ids do not contend except on the atomic slot counter) inside a shared read lock. {@link #compact()} and the one-time binary
 * calibration take the exclusive write lock, so they pause writers, never searches.
 *
 * <p>SQ8 scheme (simplest correct, data independent): per vector, scale = max|x| / 127 and code = round(x / scale) in [-127, 127]; the
 * query is either quantized the same way (kernel I8: dot = scale_v * scale_q * int8dot) or kept in float (kernels F8, F8X4, BLOCK:
 * dot = scale_v * sum q_j code_j; see {@link Sq8Kernel}); approximate L2^2 = |q|^2 - 2 dot + |v|^2 with |v|^2 stored exactly. No training, so churn cannot make it stale. Cost: a vector with one large coordinate loses resolution on
 * the small ones; the fp32 rerank repairs the top candidates.
 *
 * <p>BINARY scheme: bit j = 1 when x_j is above threshold_j. Thresholds are the per-dimension mean of the first {@value #CALIB} live
 * vectors, frozen then (a zero threshold would make all-positive data such as SIFT useless). Until calibration, searches are exact.
 * Hamming distance via Long.bitCount; rerank in fp32.
 *
 * <p>Quantized search: scan the codes (allow-list applied during the scan), keep the best rerankFactor*k candidates in a bounded heap,
 * rescore those exactly in fp32, return the best k. Scores returned are always exact fp32 scores. The exact path (Quant.NONE, or when
 * rerankFactor*k reaches the live count) scans fp32 and equals brute force exactly (ties broken by smaller id).
 */
public final class FlatIndex implements VecIndex {
    private static final VarHandle LIVE = MethodHandles.arrayElementVarHandle(long[].class);
    private static final Hit[] EMPTY = new Hit[0];
    static final int STRIPES = 64;
    /** live vectors needed before the binary thresholds are fixed */
    static final int CALIB = 512;
    private static final int GEO_CHUNKS = 7; // chunks 0..6 hold 64..4096 slots
    private static final int GEO_TOTAL = 64 * ((1 << GEO_CHUNKS) - 1); // 8128
    private static final int BIG = 4096;

    private final int dim;
    private final Metric metric;
    private final Quant quant;
    private final int rerank;
    private final int words;
    private final Sq8Kernel sq8;
    private final int sq8Mode; // scan mode of the SQ8 kernel: 1 I8, 3 F8, 4 F8X4, 5 BLOCK

    private volatile Store store = new Store();
    private final ReentrantReadWriteLock rw = new ReentrantReadWriteLock();
    private final LongIntMap[] maps = new LongIntMap[STRIPES];
    private final AtomicInteger live = new AtomicInteger();
    private float[] thr; // binary thresholds, written before 'calibrated'
    private volatile boolean calibrated;
    private volatile int slotsAtLastCompact;
    private final AtomicInteger compactions = new AtomicInteger();
    private double maxDeadFraction = 0.3;
    private int minDeadSlots = 1024;
    private double growthFactor = 2.0;
    private volatile ScheduledThreadPoolExecutor auto;
    private volatile Throwable autoError;

    public FlatIndex(int dim, Metric metric) {
        this(dim, metric, Quant.NONE, 0);
    }

    public FlatIndex(int dim, Metric metric, Quant quant) {
        this(dim, metric, quant, 0);
    }

    /** @param rerankFactor candidates kept = rerankFactor * k; 0 or less selects the default (SQ8 5, BINARY 10) */
    public FlatIndex(int dim, Metric metric, Quant quant, int rerankFactor) {
        this(dim, metric, quant, rerankFactor, Sq8Kernel.DEFAULT);
    }

    /**  sq8 how SQ8 scans score a vector (ignored for other quantizers); BLOCK changes the code layout */
    public FlatIndex(int dim, Metric metric, Quant quant, int rerankFactor, Sq8Kernel sq8) {
        if (dim < 1) throw new IllegalArgumentException("dim");
        this.dim = dim;
        this.metric = metric;
        this.quant = quant;
        this.rerank = rerankFactor > 0 ? rerankFactor : (quant == Quant.BINARY ? 10 : 5);
        this.words = (dim + 63) >>> 6;
        this.sq8 = sq8;
        this.sq8Mode = switch (sq8) { case I8 -> 1; case F8 -> 3; case F8X4 -> 4; case BLOCK -> 5; };
        for (int i = 0; i < STRIPES; i++) maps[i] = new LongIntMap();
        calibrated = quant != Quant.BINARY;
    }

    // ---- storage ----

    private static final class Chunk {
        final int start, cap;
        final long[] ids;
        final float[] f;
        final byte[] q;
        final float[] sc;
        final float[] nn;
        final long[] bits;
        final long[] liveBits;

        Chunk(int start, int cap, int dim, int words, Quant quant, Metric metric) {
            this.start = start;
            this.cap = cap;
            ids = new long[cap];
            f = new float[cap * dim];
            q = quant == Quant.SQ8 ? new byte[cap * dim] : null;
            sc = quant == Quant.SQ8 ? new float[cap] : null;
            nn = quant == Quant.SQ8 && metric == Metric.L2 ? new float[cap] : null;
            bits = quant == Quant.BINARY ? new long[cap * words] : null;
            liveBits = new long[(cap + 63) >>> 6];
        }

        long bytes() {
            long b = 64;
            b += 16 + ids.length * 8L + 16 + f.length * 4L + 16 + liveBits.length * 8L;
            if (q != null) b += 16 + q.length + 16 + sc.length * 4L;
            if (nn != null) b += 16 + nn.length * 4L;
            if (bits != null) b += 16 + bits.length * 8L;
            return b;
        }
    }

    private static final class Store {
        volatile Chunk[] chunks = new Chunk[16];
        final AtomicInteger next = new AtomicInteger();
    }

    static int chunkOf(int s) {
        if (s < GEO_TOTAL) return 31 - Integer.numberOfLeadingZeros((s >>> 6) + 1);
        return GEO_CHUNKS + ((s - GEO_TOTAL) >>> 12);
    }

    static int startOf(int ci) {
        return ci < GEO_CHUNKS ? 64 * ((1 << ci) - 1) : GEO_TOTAL + (ci - GEO_CHUNKS) * BIG;
    }

    static int capOf(int ci) {
        return ci < GEO_CHUNKS ? 64 << ci : BIG;
    }

    private Chunk ensureChunk(Store st, int ci) {
        Chunk[] cs = st.chunks;
        if (ci < cs.length && cs[ci] != null) return cs[ci];
        synchronized (st) {
            cs = st.chunks;
            if (ci >= cs.length) cs = Arrays.copyOf(cs, Math.max(ci + 1, cs.length * 2));
            if (cs[ci] == null) cs[ci] = new Chunk(startOf(ci), capOf(ci), dim, words, quant, metric);
            st.chunks = cs;
            return cs[ci];
        }
    }

    private static boolean isLive(Chunk c, int off) {
        return (((long) LIVE.getAcquire(c.liveBits, off >>> 6)) >>> (off & 63) & 1L) != 0;
    }

    private static void setLive(Chunk c, int off) {
        int w = off >>> 6;
        long bit = 1L << (off & 63);
        while (true) {
            long cur = (long) LIVE.getVolatile(c.liveBits, w);
            if (LIVE.compareAndSet(c.liveBits, w, cur, cur | bit)) return; // CAS is a release; neighbours in the word may race
        }
    }

    private static boolean clearLive(Chunk c, int off) {
        int w = off >>> 6;
        long bit = 1L << (off & 63);
        while (true) {
            long cur = (long) LIVE.getVolatile(c.liveBits, w);
            if ((cur & bit) == 0) return false;
            if (LIVE.compareAndSet(c.liveBits, w, cur, cur & ~bit)) return true;
        }
    }

    /** writes data of one slot (not yet live) from stored (already normalized for COSINE) values */
    private void writeSlot(Chunk c, int off, long id, float[] src, int so) {
        c.ids[off] = id;
        System.arraycopy(src, so, c.f, off * dim, dim);
        if (quant == Quant.SQ8) {
            if (sq8 == Sq8Kernel.BLOCK) encodeSq8(src, so, c.q, (off >>> 4) * 16 * dim + (off & 15), 16, c.sc, off);
            else encodeSq8(src, so, c.q, off * dim, 1, c.sc, off);
            if (c.nn != null) c.nn[off] = Kernels.dot(src, so, src, so, dim);
        } else if (quant == Quant.BINARY && calibrated) {
            encodeBin(src, so, thr, c.bits, off * words);
        }
    }

    /** code i goes to out[oo + i*stride] (stride 1: contiguous; stride 16: one lane of a dimension-major block) */
    private void encodeSq8(float[] src, int so, byte[] out, int oo, int stride, float[] scales, int si) {
        float m = 0f;
        for (int i = 0; i < dim; i++) m = Math.max(m, Math.abs(src[so + i]));
        if (m == 0f) {
            for (int i = 0; i < dim; i++) out[oo + i * stride] = 0;
            scales[si] = 0f;
            return;
        }
        float inv = 127f / m;
        for (int i = 0; i < dim; i++) {
            int v = Math.round(src[so + i] * inv);
            out[oo + i * stride] = (byte) (v > 127 ? 127 : Math.max(v, -127));
        }
        scales[si] = m / 127f;
    }

    private void encodeBin(float[] src, int so, float[] t, long[] out, int oo) {
        for (int w = 0; w < words; w++) {
            long b = 0;
            int hi = Math.min(dim, (w + 1) << 6);
            for (int j = w << 6; j < hi; j++) if (src[so + j] > t[j]) b |= 1L << (j & 63);
            out[oo + w] = b;
        }
    }

    // ---- VecIndex ----

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
        return live.get();
    }

    @Override
    public void add(long id, float[] v) {
        if (v.length != dim) throw new IllegalArgumentException("dim " + v.length + " != " + dim);
        addStored(id, metric == Metric.COSINE ? Kernels.normalized(v) : v);
    }

    /** v is already in stored form (normalized for COSINE) */
    private void addStored(long id, float[] sv) {
        int stripe = (Long.hashCode(id) * 0x9E3779B1) >>> 26; // top 6 bits: 64 stripes
        rw.readLock().lock();
        try {
            Store st = store;
            LongIntMap m = maps[stripe];
            synchronized (m) {
                int old = m.get(id);
                int slot = st.next.getAndIncrement();
                Chunk c = ensureChunk(st, chunkOf(slot));
                int off = slot - c.start;
                writeSlot(c, off, id, sv, 0);
                setLive(c, off); // publish
                m.put(id, slot);
                if (old >= 0) {
                    Chunk oc = st.chunks[chunkOf(old)];
                    clearLive(oc, old - oc.start);
                } else {
                    live.incrementAndGet();
                }
            }
        } finally {
            rw.readLock().unlock();
        }
        if (quant == Quant.BINARY && !calibrated && live.get() >= CALIB) calibrate();
    }

    @Override
    public boolean delete(long id) {
        int stripe = (Long.hashCode(id) * 0x9E3779B1) >>> 26;
        rw.readLock().lock();
        try {
            Store st = store;
            LongIntMap m = maps[stripe];
            synchronized (m) {
                int old = m.remove(id);
                if (old < 0) return false;
                Chunk oc = st.chunks[chunkOf(old)];
                clearLive(oc, old - oc.start);
                live.decrementAndGet();
                return true;
            }
        } finally {
            rw.readLock().unlock();
        }
    }

    private void calibrate() {
        rw.writeLock().lock();
        try {
            if (calibrated) return;
            Store st = store;
            int cnt = st.next.get();
            double[] sum = new double[dim];
            long n = 0;
            for (int ci = 0; ci < st.chunks.length; ci++) {
                Chunk c = st.chunks[ci];
                if (c == null || c.start >= cnt) break;
                for (int off = 0; off < c.cap && c.start + off < cnt; off++) {
                    if (!isLive(c, off)) continue;
                    n++;
                    for (int j = 0; j < dim; j++) sum[j] += c.f[off * dim + j];
                }
            }
            float[] t = new float[dim];
            if (n > 0) for (int j = 0; j < dim; j++) t[j] = (float) (sum[j] / n);
            thr = t;
            for (int ci = 0; ci < st.chunks.length; ci++) {
                Chunk c = st.chunks[ci];
                if (c == null || c.start >= cnt) break;
                for (int off = 0; off < c.cap && c.start + off < cnt; off++) {
                    if (isLive(c, off)) encodeBin(c.f, off * dim, t, c.bits, off * words);
                }
            }
            calibrated = true;
        } finally {
            rw.writeLock().unlock();
        }
    }

    /** number of slots ever written in the current store (live plus dead) */
    public int slotsUsed() {
        return store.next.get();
    }

    /** fraction of written slots that are dead (deleted or replaced) */
    public double deadFraction() {
        int s = slotsUsed();
        return s == 0 ? 0.0 : 1.0 - (double) live.get() / s;
    }

    /**
     * Repacks the live vectors into fresh chunks and swaps them in. Writers pause for the copy; searches never block: one that started
     * earlier finishes on the old chunks, a later one uses the new ones. Call when deadFraction() is high.
     */
    public void compact() {
        rw.writeLock().lock();
        try {
            Store old = store;
            Store ns = new Store();
            int cnt = old.next.get();
            for (LongIntMap m : maps) m.clear(); // refilled below; no other thread touches the maps under the write lock
            for (int ci = 0; ci < old.chunks.length; ci++) {
                Chunk c = old.chunks[ci];
                if (c == null || c.start >= cnt) break;
                for (int off = 0; off < c.cap && c.start + off < cnt; off++) {
                    if (!isLive(c, off)) continue;
                    int slot = ns.next.getAndIncrement();
                    Chunk nc = ensureChunk(ns, chunkOf(slot));
                    int noff = slot - nc.start;
                    writeSlot(nc, noff, c.ids[off], c.f, off * dim);
                    setLive(nc, noff);
                    long id = c.ids[off];
                    maps[(Long.hashCode(id) * 0x9E3779B1) >>> 26].put(id, slot);
                }
            }
            store = ns;
            slotsAtLastCompact = ns.next.get();
            compactions.incrementAndGet();
        } finally {
            rw.writeLock().unlock();
        }
    }

    // ---- automatic compaction ----

    /**
     * Policy for {@link #compactIfNeeded()}: compact when at least minDeadSlots slots are dead AND either the dead fraction reached
     * maxDeadFraction or the written slots grew to growthFactor times the slot count after the last compaction (or 4096, whichever is
     * larger). Defaults 0.3, 1024, 2.0.
     */
    public void setCompactionPolicy(double maxDeadFraction, int minDeadSlots, double growthFactor) {
        this.maxDeadFraction = maxDeadFraction;
        this.minDeadSlots = minDeadSlots;
        this.growthFactor = growthFactor;
    }

    /** number of compactions so far */
    public int compactions() {
        return compactions.get();
    }

    /** the last exception thrown inside the built-in daemon, or null */
    public Throwable lastCompactionError() {
        return autoError;
    }

    /**
     * Compacts when the policy says so; returns whether it did. Cheap when nothing is due (two atomic reads). The caller may call this
     * from its own scheduler; searches are never blocked, writers wait for the copy. Transient memory: the live data exists twice
     * while the copy runs.
     */
    public boolean compactIfNeeded() {
        int slots = slotsUsed();
        int dead = slots - live.get();
        if (dead < minDeadSlots) return false;
        boolean frac = dead >= maxDeadFraction * slots;
        boolean growth = slots >= growthFactor * Math.max(slotsAtLastCompact, 4096);
        if (!frac && !growth) return false;
        compact();
        return true;
    }

    /** starts a daemon thread that calls compactIfNeeded() every periodMillis; stopped by close() or stopAutoCompaction() */
    public synchronized void startAutoCompaction(long periodMillis) {
        if (auto != null) return;
        ScheduledThreadPoolExecutor ex = new ScheduledThreadPoolExecutor(1, r -> {
            Thread t = new Thread(r, "flat-compactor");
            t.setDaemon(true);
            return t;
        });
        ex.setRemoveOnCancelPolicy(true);
        ex.scheduleWithFixedDelay(() -> {
            try {
                compactIfNeeded();
            } catch (Throwable t) {
                autoError = t;
            }
        }, periodMillis, periodMillis, TimeUnit.MILLISECONDS);
        auto = ex;
    }

    public synchronized void stopAutoCompaction() {
        ScheduledThreadPoolExecutor ex = auto;
        auto = null;
        if (ex != null) ex.shutdownNow();
    }

    @Override
    public void close() {
        stopAutoCompaction();
    }

    // ---- serialize and load ----

    private static final long MAGIC = 0x484D464C41543031L; // "HMFLAT01"
    private static final int FORMAT_VERSION = 1;

    /**
     * Writes the live vectors. Format (big-endian header, little-endian float payload): magic "HMFLAT01", version int, dim int, metric
     * ordinal int, quant ordinal int, rerankFactor int, sq8 kernel ordinal int, calibrated byte, count int, [dim floats of binary
     * thresholds when quant is BINARY and calibrated], then count records of (id long, dim float32 in STORED form, i.e. normalized for
     * COSINE), then a CRC-32 long over every byte before it. Only live vectors are written (the file is a compacted snapshot, so no
     * live bitmap is needed). Codes are NOT stored: SQ8 codes are a deterministic function of the vector and are recomputed on load
     * (about as cheap as reading them), binary codes are recomputed from the stored thresholds, which keeps the file at fp32 size and
     * independent of the scan kernel. Writers pause for the duration (exclusive lock, so the snapshot is consistent); searches do not.
     */
    public void serialize(OutputStream os) throws IOException {
        rw.writeLock().lock();
        try {
            BufferedOutputStream bos = new BufferedOutputStream(os, 1 << 16);
            CRC32 crc = new CRC32();
            DataOutputStream out = new DataOutputStream(new CheckedOutputStream(bos, crc));
            int n = live.get();
            out.writeLong(MAGIC);
            out.writeInt(FORMAT_VERSION);
            out.writeInt(dim);
            out.writeInt(metric.ordinal());
            out.writeInt(quant.ordinal());
            out.writeInt(rerank);
            out.writeInt(sq8.ordinal());
            out.writeByte(calibrated && quant == Quant.BINARY ? 1 : 0);
            out.writeInt(n);
            ByteBuffer bb = ByteBuffer.allocate(dim * 4).order(ByteOrder.LITTLE_ENDIAN);
            if (quant == Quant.BINARY && calibrated) {
                bb.clear();
                bb.asFloatBuffer().put(thr, 0, dim);
                out.write(bb.array());
            }
            Store st = store;
            int cnt = st.next.get();
            int written = 0;
            for (int ci = 0; ci < st.chunks.length; ci++) {
                Chunk c = st.chunks[ci];
                if (c == null || c.start >= cnt) break;
                for (int off = 0; off < c.cap && c.start + off < cnt; off++) {
                    if (!isLive(c, off)) continue;
                    out.writeLong(c.ids[off]);
                    bb.clear();
                    bb.asFloatBuffer().put(c.f, off * dim, dim);
                    out.write(bb.array());
                    written++;
                }
            }
            if (written != n) throw new IllegalStateException("live count " + n + " but " + written + " live slots");
            out.flush();
            long sum = crc.getValue();
            new DataOutputStream(bos).writeLong(sum);
            bos.flush();
        } finally {
            rw.writeLock().unlock();
        }
    }

    /** Reads what {@link #serialize} wrote. Throws IOException on a bad magic, version, header value, truncation or checksum mismatch. */
    public static FlatIndex load(InputStream is) throws IOException {
        BufferedInputStream bis = new BufferedInputStream(is, 1 << 16);
        CRC32 crc = new CRC32();
        DataInputStream in = new DataInputStream(new CheckedInputStream(bis, crc));
        if (in.readLong() != MAGIC) throw new IOException("not a FlatIndex file (bad magic)");
        int version = in.readInt();
        if (version != FORMAT_VERSION) throw new IOException("unsupported FlatIndex format version " + version);
        int dim = in.readInt(), mo = in.readInt(), qo = in.readInt(), rerank = in.readInt(), ko = in.readInt();
        int cal = in.readUnsignedByte();
        int n = in.readInt();
        if (dim < 1 || dim > (1 << 20) || mo < 0 || mo >= Metric.values().length || qo < 0 || qo >= Quant.values().length || rerank < 1
                || ko < 0 || ko >= Sq8Kernel.values().length || cal > 1 || n < 0) {
            throw new IOException("bad FlatIndex header");
        }
        FlatIndex idx = new FlatIndex(dim, Metric.values()[mo], Quant.values()[qo], rerank, Sq8Kernel.values()[ko]);
        byte[] raw = new byte[dim * 4];
        ByteBuffer bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        if (cal == 1) {
            in.readFully(raw);
            float[] t = new float[dim];
            bb.clear();
            bb.asFloatBuffer().get(t);
            idx.thr = t;
            idx.calibrated = true;
        }
        for (int i = 0; i < n; i++) {
            long id = in.readLong();
            in.readFully(raw);
            float[] v = new float[dim];
            bb.clear();
            bb.asFloatBuffer().get(v);
            idx.addStored(id, v);
        }
        if (idx.size() != n) throw new IOException("duplicate ids in file: " + n + " records, " + idx.size() + " distinct");
        long expected = crc.getValue();
        long stored;
        try {
            stored = new DataInputStream(bis).readLong();
        } catch (EOFException e) {
            throw new IOException("truncated FlatIndex file (no checksum)", e);
        }
        if (stored != expected) throw new IOException("FlatIndex checksum mismatch");
        return idx;
    }

    @Override
    public long bytesHeld() {
        long b = 256 + STRIPES * 16L;
        Store st = store;
        Chunk[] cs = st.chunks;
        b += 16 + cs.length * 8L;
        for (Chunk c : cs) if (c != null) b += c.bytes();
        for (LongIntMap m : maps) {
            synchronized (m) {
                b += m.bytes();
            }
        }
        if (thr != null) b += 16 + thr.length * 4L;
        return b;
    }

    // ---- search ----

    @Override
    public Hit[] search(float[] q, int k, LongPredicate allow) {
        if (q.length != dim) throw new IllegalArgumentException("dim " + q.length + " != " + dim);
        if (k <= 0) return EMPTY;
        Store st = store;
        int cnt = st.next.get();
        if (cnt == 0) return EMPTY;
        float[] qq = metric == Metric.COSINE ? Kernels.normalized(q) : q;
        boolean coded = quant != Quant.NONE && calibrated;
        int liveN = live.get();
        int want = Math.min(k, cnt);
        for (int attempt = 0; ; attempt++) {
            TopK h;
            long cand = (long) rerank * want;
            if (coded && cand < liveN) {
                TopK cs = new TopK((int) cand);
                scan(st, cnt, qq, allow, cs, quant == Quant.SQ8 ? sq8Mode : 2);
                h = new TopK(want);
                rerankInto(st, qq, cs, h);
            } else {
                h = new TopK(want);
                scan(st, cnt, qq, allow, h, 0);
            }
            Hit[] res = h.drainSorted();
            Hit[] d = dedupe(res);
            if (d.length >= k) return d.length == k ? d : Arrays.copyOf(d, k);
            if (d.length == res.length || res.length < want || attempt >= 4) return d;
            want = Math.min(cnt, want + (res.length - d.length) + 1);
        }
    }

    /** a replace in flight can show an id twice (old and new slot); keep the better one. Rare, so the common case is one sort of ids. */
    private static Hit[] dedupe(Hit[] r) {
        if (r.length < 2) return r;
        long[] ids = new long[r.length];
        for (int i = 0; i < r.length; i++) ids[i] = r[i].id();
        Arrays.sort(ids);
        boolean dup = false;
        for (int i = 1; i < ids.length; i++) if (ids[i] == ids[i - 1]) { dup = true; break; }
        if (!dup) return r;
        Hit[] out = new Hit[r.length];
        int n = 0;
        outer:
        for (Hit h : r) { // r is best first, so the first occurrence is the best
            for (int j = 0; j < n; j++) if (out[j].id() == h.id()) continue outer;
            out[n++] = h;
        }
        return Arrays.copyOf(out, n);
    }

    private void rerankInto(Store st, float[] qq, TopK cand, TopK out) {
        Chunk[] cs = st.chunks;
        for (int i = 0; i < cand.n; i++) {
            int slot = cand.slot[i];
            Chunk c = cs[chunkOf(slot)];
            int off = slot - c.start;
            if (!isLive(c, off)) continue;
            float s = metric == Metric.L2 ? -Kernels.l2sq(qq, 0, c.f, off * dim, dim) : Kernels.dot(qq, 0, c.f, off * dim, dim);
            out.offer(s, c.ids[off], slot);
        }
    }

    /** mode 0: exact fp32, 1: SQ8 codes, 2: binary codes. Scores are higher-is-better in every mode. */
    private void scan(Store st, int cnt, float[] qq, LongPredicate allow, TopK h, int mode) {
        byte[] qc = null;
        float qs = 0f, qn2 = 0f;
        long[] qb = null;
        if (mode == 1) {
            qc = new byte[dim];
            float[] one = new float[1];
            encodeSq8(qq, 0, qc, 0, 1, one, 0);
            qs = one[0];
            qn2 = Kernels.dot(qq, 0, qq, 0, dim);
        } else if (mode == 3 || mode == 4 || mode == 5) {
            qn2 = Kernels.dot(qq, 0, qq, 0, dim);
        } else if (mode == 2) {
            qb = new long[words];
            encodeBin(qq, 0, thr, qb, 0);
        }
        final boolean l2 = metric == Metric.L2;
        final int d = dim;
        Chunk[] cs = st.chunks;
        final float[] tmp = mode == 5 ? new float[16] : null;
        for (int ci = 0; ci < cs.length; ci++) {
            Chunk c = cs[ci];
            if (c == null) {
                if (startOf(ci) >= cnt) break;
                continue;
            }
            if (c.start >= cnt) break;
            int lim = Math.min(c.cap, cnt - c.start);
            int nw = (lim + 63) >>> 6;
            if (mode == 5) {
                for (int w = 0; w < nw; w++) {
                    long bits = (long) LIVE.getAcquire(c.liveBits, w);
                    int rem = lim - (w << 6);
                    if (rem < 64) bits &= (1L << rem) - 1;
                    for (int nib = 0; nib < 4 && bits != 0; nib++) {
                        int nb = (int) ((bits >>> (nib << 4)) & 0xFFFF);
                        if (nb == 0) continue;
                        int blockOff = (w << 6) + (nib << 4);
                        Kernels.block16(qq, 0, c.q, (blockOff >>> 4) * 16 * d, d, tmp);
                        while (nb != 0) {
                            int lane = Integer.numberOfTrailingZeros(nb);
                            nb &= nb - 1;
                            int off = blockOff + lane;
                            long id = c.ids[off];
                            if (allow != null && !allow.test(id)) continue;
                            float a = tmp[lane] * c.sc[off];
                            h.offer(l2 ? -(qn2 - 2f * a + c.nn[off]) : a, id, c.start + off);
                        }
                    }
                }
                continue;
            }
            for (int w = 0; w < nw; w++) {
                long bits = (long) LIVE.getAcquire(c.liveBits, w);
                int rem = lim - (w << 6);
                if (rem < 64) bits &= (1L << rem) - 1;
                while (bits != 0) {
                    int off = (w << 6) + Long.numberOfTrailingZeros(bits);
                    bits &= bits - 1;
                    long id = c.ids[off];
                    if (allow != null && !allow.test(id)) continue;
                    float s;
                    if (mode == 0) {
                        s = l2 ? -Kernels.l2sq(qq, 0, c.f, off * d, d) : Kernels.dot(qq, 0, c.f, off * d, d);
                    } else if (mode == 1) {
                        float a = Kernels.dotI8(c.q, off * d, qc, 0, d) * c.sc[off] * qs;
                        s = l2 ? -(qn2 - 2f * a + c.nn[off]) : a;
                    } else if (mode == 3 || mode == 4) {
                        float a = (mode == 3 ? Kernels.dotF8(qq, 0, c.q, off * d, d) : Kernels.dotF8x4(qq, 0, c.q, off * d, d)) * c.sc[off];
                        s = l2 ? -(qn2 - 2f * a + c.nn[off]) : a;
                    } else {
                        s = -Kernels.hamming(qb, 0, c.bits, off * words, words);
                    }
                    h.offer(s, id, c.start + off);
                }
            }
        }
    }
}
