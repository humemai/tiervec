package com.humemai.vec;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Threaded test of the IVF index: 8 writers (add, replace, delete on their own id ranges), 8 readers, one thread that calls retrain() over
 * and over (so the model is swapped many times under load), one that calls compact(). Pass = zero exceptions, no reader sees a deleted id,
 * results sorted and duplicate free, every writer finds its own insert right after it (read-your-writes, every 10th op), every writer finds
 * its last 100 inserts at the end, and after the stop and one more retrain: size equals the sum of the writers' models, the list sizes add
 * up to size, and every id of every model is found by a search for its own vector. Args: seconds (default 10), quant (NONE or SQ8, default SQ8).
 */
public final class IvfConcurrencyTest {
    static final int DIM = 100; // not a multiple of 8
    static final int MAXN = 60000;
    static final int W = 8, R = 8;
    static final int LAT_CAP = 100000;

    static IvfIndex idx;
    static float[][] centers;
    static final ConcurrentLinkedQueue<String> errors = new ConcurrentLinkedQueue<>();
    static final AtomicBoolean stop = new AtomicBoolean(), retraining = new AtomicBoolean();
    static final ConcurrentHashMap<Long, Long> deleted = new ConcurrentHashMap<>();
    static final AtomicLong seq = new AtomicLong(), writes = new AtomicLong(), reads = new AtomicLong();
    static final AtomicInteger retrains = new AtomicInteger(), lastChecked = new AtomicInteger(), lastMissing = new AtomicInteger();
    @SuppressWarnings("unchecked")
    static final Map<Long, float[]>[] MODELS = new Map[W];
    static final long[][] LAT_NORMAL = new long[R][LAT_CAP], LAT_DURING = new long[R][LAT_CAP];
    static final int[] NORMAL_N = new int[R], DURING_N = new int[R];

    public static void main(String[] args) throws Exception {
        int seconds = args.length > 0 ? Integer.parseInt(args[0]) : 10;
        Quant quant = args.length > 1 ? Quant.valueOf(args[1]) : Quant.SQ8;
        IvfIndex.Params p = new IvfIndex.Params();
        p.quant = quant;
        p.kmeansIters = 4;
        p.trainSample = 10000;
        p.threads = 2;
        p.trainAt = 2000;
        idx = new IvfIndex(DIM, Metric.COSINE, p);
        centers = IvfT.centers(32, DIM, 7);
        System.out.println("IvfConcurrencyTest seconds=" + seconds + " quant=" + quant + " kernels=" + (Kernels.VECTOR ? "vector" : "scalar"));
        ArrayList<Thread> threads = new ArrayList<>();
        for (int w = 0; w < W; w++) {
            final int wi = w;
            MODELS[wi] = new HashMap<>();
            threads.add(new Thread(() -> writer(wi), "writer-" + w));
        }
        for (int r = 0; r < R; r++) {
            final int ri = r;
            threads.add(new Thread(() -> reader(ri), "reader-" + r));
        }
        threads.add(new Thread(IvfConcurrencyTest::retrainer, "retrainer"));
        threads.add(new Thread(IvfConcurrencyTest::compactor, "compactor"));
        long t0 = System.nanoTime();
        for (Thread t : threads) t.start();
        Thread.sleep(seconds * 1000L);
        stop.set(true);
        for (Thread t : threads) {
            t.join(180000);
            if (t.isAlive()) errors.add("thread " + t.getName() + " did not stop");
        }
        System.out.println("INFO ran " + (System.nanoTime() - t0) / 1_000_000 + " ms: writes " + writes.get() + ", reads " + reads.get() + ", retrains during the run " + retrains.get() + ", rebuilds " + idx.rebuildCount() + ", size " + idx.size() + ", nlist " + idx.nlist());

        IvfT.check(errors.isEmpty(), "no errors during the run (" + errors.size() + ")");
        int shown = 0;
        for (String e : errors) {
            if (shown++ < 15) System.out.println("  ERROR " + e);
        }
        IvfT.check(retrains.get() >= 1, "at least one retrain completed under load, got " + retrains.get());
        // Proof that writers and readers really overlapped (some of each, plus the retrain above), not a throughput floor: CI VMs differ by 10x in speed
        // and this test is about correctness under concurrency, not speed. 249 reads in 5 s was seen on a GitHub-hosted Windows runner.
        IvfT.check(writes.get() > 100 && reads.get() > 50, "writers and readers both ran, writes " + writes.get() + ", reads " + reads.get());
        IvfT.check(lastChecked.get() > 0, "the last-100-inserts check ran, checked " + lastChecked.get());
        IvfT.check(lastMissing.get() == 0, "every writer found its last 100 inserts, missing " + lastMissing.get());

        long t1 = System.nanoTime();
        idx.retrain(); // one more, now without load
        System.out.println("INFO final retrain " + (System.nanoTime() - t1) / 1_000_000 + " ms");
        long expected = 0;
        int missing = 0;
        for (int w = 0; w < W; w++) {
            expected += MODELS[w].size();
            for (Map.Entry<Long, float[]> e : MODELS[w].entrySet()) {
                boolean ok = false;
                for (Hit h : idx.search(e.getValue(), 3, null)) if (h.id() == e.getKey()) ok = true;
                if (!ok) missing++;
            }
        }
        IvfT.check(idx.size() == expected, "size equals the model: index " + idx.size() + ", model " + expected);
        long sum = 0;
        for (int s : idx.listSizes()) sum += s;
        IvfT.check(sum == idx.size(), "list sizes add up to size: " + sum + " vs " + idx.size());
        IvfT.check(missing == 0, "every id of every model is found by its own vector, missing " + missing + " of " + expected);
        IvfT.check(idx.lastError() == null, "no maintenance error " + idx.lastError());
        percentiles("search latency (us) outside a retrain", LAT_NORMAL, NORMAL_N);
        percentiles("search latency (us) during a retrain", LAT_DURING, DURING_N);
        idx.close();
        IvfT.done("IvfConcurrencyTest");
    }

    static void percentiles(String label, long[][] lat, int[] cnt) {
        int total = 0;
        for (int c : cnt) total += c;
        long[] all = new long[total];
        int k = 0;
        for (int r = 0; r < R; r++) {
            System.arraycopy(lat[r], 0, all, k, cnt[r]);
            k += cnt[r];
        }
        Arrays.sort(all);
        if (total == 0) {
            System.out.println("INFO " + label + ": no samples");
            return;
        }
        System.out.println("INFO " + label + ": n " + total + ", p50 " + all[total / 2] / 1000.0 + ", p99 " + all[(int) Math.min(total - 1L, (long) (total * 0.99))] / 1000.0 + ", max " + all[total - 1] / 1000.0);
    }

    static boolean found(long id, float[] v) {
        for (Hit h : idx.search(v, 5, null)) if (h.id() == id) return true;
        return false;
    }

    static void writer(int w) {
        Random rnd = new Random(1000 + w);
        Map<Long, float[]> model = MODELS[w];
        ArrayList<Long> live = new ArrayList<>();
        ArrayDeque<Long> last = new ArrayDeque<>();
        long next = (long) w * 100_000_000L;
        long ops = 0;
        try {
            while (!stop.get()) {
                double r = rnd.nextDouble();
                boolean full = idx.size() > MAXN;
                if (live.isEmpty() || (!full && r < 0.6)) {
                    long id = next++;
                    float[] v = IvfT.point(centers, rnd, 1.0);
                    idx.add(id, v);
                    model.put(id, v);
                    live.add(id);
                    last.addLast(id);
                    if (last.size() > 100) last.pollFirst();
                    if (ops % 10 == 0 && !found(id, v)) errors.add("writer " + w + ": own insert " + id + " not found right after add");
                } else if (r < 0.75 && !full) {
                    long id = live.get(rnd.nextInt(live.size()));
                    float[] v = IvfT.point(centers, rnd, 1.0);
                    idx.add(id, v);
                    model.put(id, v);
                    last.addLast(id);
                    if (last.size() > 100) last.pollFirst();
                    if (ops % 10 == 0 && !found(id, v)) errors.add("writer " + w + ": replaced id " + id + " not found right after add");
                } else {
                    int j = rnd.nextInt(live.size());
                    long id = live.get(j);
                    live.set(j, live.get(live.size() - 1));
                    live.remove(live.size() - 1);
                    if (!idx.delete(id)) errors.add("writer " + w + ": delete of a live id returned false " + id);
                    model.remove(id);
                    deleted.put(id, seq.incrementAndGet());
                }
                ops++;
                writes.incrementAndGet();
            }
            int checked = 0, missing = 0;
            for (long id : last) {
                float[] v = model.get(id);
                if (v == null) continue;
                checked++;
                if (!found(id, v)) missing++;
            }
            lastChecked.addAndGet(checked);
            lastMissing.addAndGet(missing);
        } catch (Throwable t) {
            errors.add("writer " + w + ": " + t);
            t.printStackTrace(System.out);
        }
    }

    static void reader(int ri) {
        Random rnd = new Random(5000 + ri);
        long iter = 0;
        try {
            while (!stop.get()) {
                float[] q = IvfT.point(centers, rnd, 1.0);
                boolean filtered = iter % 7 == 6;
                long s0 = seq.get();
                boolean dur = retraining.get();
                long t0 = System.nanoTime();
                Hit[] h = filtered ? idx.search(q, 10, id -> (id & 1L) == 0) : idx.search(q, 10, null);
                long dt = System.nanoTime() - t0;
                if (!filtered && !dur && !retraining.get()) {
                    if (NORMAL_N[ri] < LAT_CAP) LAT_NORMAL[ri][NORMAL_N[ri]++] = dt;
                } else if (!filtered && dur && retraining.get()) {
                    if (DURING_N[ri] < LAT_CAP) LAT_DURING[ri][DURING_N[ri]++] = dt;
                }
                if (h.length > 10) errors.add("reader " + ri + ": more than k hits");
                for (int i = 0; i < h.length; i++) {
                    if (!Float.isFinite(h[i].score())) errors.add("reader " + ri + ": non-finite score");
                    if (i > 0 && h[i - 1].score() < h[i].score()) errors.add("reader " + ri + ": results not sorted");
                    for (int j = 0; j < i; j++) if (h[j].id() == h[i].id()) errors.add("reader " + ri + ": duplicate id " + h[i].id());
                    if (filtered && (h[i].id() & 1L) != 0) errors.add("reader " + ri + ": filter violated by " + h[i].id());
                    Long d = deleted.get(h[i].id());
                    if (d != null && d <= s0) errors.add("reader " + ri + ": returned deleted id " + h[i].id());
                }
                iter++;
                reads.incrementAndGet();
            }
        } catch (Throwable t) {
            errors.add("reader " + ri + ": " + t);
            t.printStackTrace(System.out);
        }
    }

    static void retrainer() {
        try {
            Thread.sleep(1000);
            while (!stop.get()) {
                retraining.set(true);
                try {
                    idx.retrain();
                    retrains.incrementAndGet();
                } finally {
                    retraining.set(false);
                }
                Thread.sleep(300);
            }
        } catch (Throwable t) {
            errors.add("retrainer: " + t);
            t.printStackTrace(System.out);
        }
    }

    static void compactor() {
        try {
            while (!stop.get()) {
                idx.compact();
                Thread.sleep(100);
            }
        } catch (Throwable t) {
            errors.add("compactor: " + t);
            t.printStackTrace(System.out);
        }
    }
}
