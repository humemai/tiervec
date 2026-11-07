package com.humemai.vec;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.LockSupport;

/**
 * Rule G2: 8 writer threads plus 8 reader threads for DURATION seconds (default 60), with consolidate() running in a background thread
 * every few seconds. Pass criteria: zero exceptions; read-your-writes checks pass; deleted ids never returned by a search that started
 * after the delete returned; every writer finds its own last 100 inserts at the end; size() equals inserts minus deletes (no lost or
 * phantom vectors). Reports reader p50/p99/p999 with and without a consolidation running.
 * Bounded by time AND by op counts (at most 10000 inserts per writer, so at most 80000 vectors).
 */
public final class GraphConcurrencyTest {
    static final int DIM = 128, WRITERS = 8, READERS = 8, MAX_PER_WRITER = 10000, LAT_CAP = 400_000;
    static final ConcurrentLinkedQueue<Throwable> errors = new ConcurrentLinkedQueue<>();
    static final ConcurrentHashMap<Long, Long> deletedAt = new ConcurrentHashMap<>();
    static final AtomicLong inserts = new AtomicLong(), deletes = new AtomicLong(), rywChecks = new AtomicLong(), rywMiss = new AtomicLong(),
            phantoms = new AtomicLong(), queries = new AtomicLong(), orderViol = new AtomicLong(), dupViol = new AtomicLong(), filterViol = new AtomicLong();
    static volatile boolean stop;
    static volatile int phase; // 0 idle, 1 consolidating
    static final long[][] lat = new long[2][];
    static final AtomicInteger[] latN = { new AtomicInteger(), new AtomicInteger() };

    public static void main(String[] args) throws Exception {
        int seconds = args.length > 0 ? Integer.parseInt(args[0]) : 60;
        boolean sq8 = args.length > 1 && args[1].equals("sq8");
        System.out.println("jdk " + System.getProperty("java.version") + " seconds=" + seconds + " sq8=" + sq8);
        GraphT.Gen gen = new GraphT.Gen(DIM, 31, 60, 0.4f);
        GraphIndex idx = new GraphIndex(DIM, Metric.COSINE, GraphIndex.Params.defaults().withSq8(sq8));
        lat[0] = new long[LAT_CAP]; lat[1] = new long[LAT_CAP];
        long start = System.nanoTime(), end = start + seconds * 1_000_000_000L;
        List<Thread> threads = new ArrayList<>();
        long[][] lastIds = new long[WRITERS][100];
        int[] wInserted = new int[WRITERS];
        for (int w = 0; w < WRITERS; w++) {
            final int wi = w;
            threads.add(new Thread(() -> {
                try {
                    long base = (long) wi * 1_000_000L; int seq = 0;
                    while (!stop && System.nanoTime() < end && seq < MAX_PER_WRITER) {
                        double t = (System.nanoTime() - start) / 1e9;
                        if (seq > 3000 + 90 * t) { LockSupport.parkNanos(1_000_000L); continue; }
                        long id = base + seq; float[] v = gen.vec(id);
                        idx.add(id, v); inserts.incrementAndGet();
                        lastIds[wi][seq % 100] = id; seq++; wInserted[wi] = seq;
                        if (seq % 20 == 0) {
                            rywChecks.incrementAndGet();
                            Hit[] h = idx.search(v, 10, null, 100); boolean f = false; for (Hit x : h) if (x.id() == id) f = true;
                            if (!f) rywMiss.incrementAndGet();
                        }
                        if (seq % 5 == 0 && seq > 400) { // churn: delete an id older than the last 300
                            long victim = base + (seq - 300 - (seq / 5) % 50);
                            if (idx.delete(victim)) { deletes.incrementAndGet(); deletedAt.put(victim, System.nanoTime()); }
                        }
                    }
                } catch (Throwable e) { errors.add(e); }
            }, "writer-" + w));
        }
        for (int r = 0; r < READERS; r++) {
            final int ri = r;
            threads.add(new Thread(() -> {
                try {
                    SplittableRandom rnd = new SplittableRandom(1000 + ri);
                    long n = 0;
                    while (!stop && System.nanoTime() < end + 2_000_000_000L && n < 3_000_000) {
                        n++;
                        float[] q = gen.vec(5_000_000L + rnd.nextInt(100000));
                        boolean filtered = n % 10 == 0;
                        long t0 = System.nanoTime(); int ph = phase;
                        Hit[] h = filtered ? idx.search(q, 10, id -> id % 7 == 0, 100) : idx.search(q, 10, null, 100);
                        long dt = System.nanoTime() - t0;
                        if (ph == phase) { int slot = latN[ph].getAndIncrement(); if (slot < LAT_CAP) lat[ph][slot] = dt; }
                        queries.incrementAndGet();
                        for (int i = 0; i < h.length; i++) {
                            Long da = deletedAt.get(h[i].id());
                            if (da != null && da < t0) phantoms.incrementAndGet();
                            if (i > 0 && h[i].score() > h[i - 1].score() + 1e-6f) orderViol.incrementAndGet();
                            for (int j = 0; j < i; j++) if (h[j].id() == h[i].id()) dupViol.incrementAndGet();
                            if (filtered && h[i].id() % 7 != 0) filterViol.incrementAndGet();
                        }
                    }
                } catch (Throwable e) { errors.add(e); }
            }, "reader-" + r));
        }
        final List<String> consolidations = new ArrayList<>();
        Thread cons = new Thread(() -> {
            try {
                long next = start + 8_000_000_000L;
                for (int round = 0; round < 12 && !stop; round++) {
                    while (System.nanoTime() < next && !stop && System.nanoTime() < end) LockSupport.parkNanos(20_000_000L);
                    if (stop || System.nanoTime() >= end) break;
                    phase = 1; long a = System.nanoTime();
                    int rec = idx.consolidate();
                    double ms = (System.nanoTime() - a) / 1e6; phase = 0;
                    consolidations.add(String.format("t=%.1fs recycled=%d took=%.0f ms tombstonesLeft=%d size=%d", (a - start) / 1e9, rec, ms, idx.tombstoneCount(), idx.size()));
                    next = System.nanoTime() + 7_000_000_000L;
                }
            } catch (Throwable e) { errors.add(e); }
        }, "consolidator");
        threads.add(cons);
        for (Thread t : threads) t.start();
        for (Thread t : threads) t.join((seconds + 60) * 1000L);
        stop = true;
        for (Thread t : threads) if (t.isAlive()) { errors.add(new IllegalStateException("thread did not finish: " + t.getName())); }
        double elapsed = (System.nanoTime() - start) / 1e9;

        System.out.printf("elapsed %.1fs inserts=%d deletes=%d queries=%d (%.0f q/s) size=%d tombstones=%d ordinals=%d%n", elapsed, inserts.get(), deletes.get(),
                queries.get(), queries.get() / elapsed, idx.size(), idx.tombstoneCount(), idx.ordinalsAllocated());
        for (String c : consolidations) System.out.println("  consolidate " + c);
        GraphT.check(errors.isEmpty(), "zero exceptions, got " + errors.size());
        for (Throwable e : errors) { System.out.println("  " + e); e.printStackTrace(System.out); }
        GraphT.check(rywMiss.get() == 0, "read-your-writes: " + rywMiss.get() + " misses of " + rywChecks.get());
        GraphT.check(phantoms.get() == 0, "deleted ids returned after delete: " + phantoms.get());
        GraphT.check(orderViol.get() == 0 && dupViol.get() == 0 && filterViol.get() == 0, "order/dup/filter violations " + orderViol + "/" + dupViol + "/" + filterViol);
        GraphT.check(idx.size() == inserts.get() - deletes.get(), "size " + idx.size() + " == inserts - deletes " + (inserts.get() - deletes.get()));
        GraphT.check(consolidations.size() >= 1 || seconds < 10, "at least one consolidation ran: " + consolidations.size());

        verify(idx, gen, lastIds, "final (before extra consolidate)");
        int rec = idx.consolidate();
        System.out.println("  final consolidate recycled=" + rec + " tombstones=" + idx.tombstoneCount());
        verify(idx, gen, lastIds, "after final consolidate");
        GraphT.check(idx.size() == inserts.get() - deletes.get(), "size after final consolidate");

        for (int ph = 0; ph < 2; ph++) {
            int n = Math.min(latN[ph].get(), LAT_CAP); long[] s = GraphT.sorted(lat[ph], n);
            System.out.printf("  reader latency %-14s n=%d p50=%d us p99=%d us p99.9=%d us max=%d us%n", ph == 0 ? "no-consolidate" : "consolidating", n,
                    GraphT.pct(s, .5) / 1000, GraphT.pct(s, .99) / 1000, GraphT.pct(s, .999) / 1000, n == 0 ? 0 : s[n - 1] / 1000);
        }
        System.out.printf("RESULT checks=%d failures=%d%n", GraphT.checks, GraphT.failures);
        if (GraphT.exitOnFailure) System.exit(GraphT.failures == 0 ? 0 : 1);
    }

    static void verify(GraphIndex idx, GraphT.Gen gen, long[][] lastIds, String label) {
        int miss = 0, total = 0;
        for (long[] ids : lastIds) for (long id : ids) {
            if (id == 0 && ids[0] == 0 && false) continue;
            if (deletedAt.containsKey(id)) continue;
            total++;
            Hit[] h = idx.search(gen.vec(id), 10, null, 200); boolean f = false; for (Hit x : h) if (x.id() == id) f = true;
            if (!f) miss++;
        }
        GraphT.check(miss == 0, label + ": writers find their last inserts, missing " + miss + " of " + total);
        int ph = 0, checked = 0;
        for (var e : deletedAt.entrySet()) {
            if (checked++ >= 3000) break;
            for (Hit x : idx.search(gen.vec(e.getKey()), 10, null, 100)) if (x.id() == e.getKey()) ph++;
        }
        GraphT.check(ph == 0, label + ": deleted ids returned " + ph + " (checked " + Math.min(checked, 3000) + ")");
    }
}
