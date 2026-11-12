package ai.humem.tiervec;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Query latency before, during and after a background consolidate() (single query thread, warm), after deleting 40 percent of a
 * 50000-vector index. Descriptive numbers; also checks that results during consolidation never include a deleted id.
 */
public final class GraphLatencyTest {
    static final int DIM = 128;

    public static void main(String[] args) throws Exception {
        int n = args.length > 0 ? Integer.parseInt(args[0]) : 50000;
        GraphT.Gen g = new GraphT.Gen(DIM, 41, 80, 0.4f); float[][] d = new float[n][];
        for (int i = 0; i < n; i++) d[i] = g.vec(i);
        GraphIndex idx = new GraphIndex(DIM, Metric.COSINE);
        long t0 = System.nanoTime();
        AtomicInteger next = new AtomicInteger(); List<Thread> ts = new ArrayList<>();
        for (int t = 0; t < 8; t++) { Thread th = new Thread(() -> { int i; while ((i = next.getAndIncrement()) < n) idx.add(i, d[i]); }); ts.add(th); th.start(); }
        for (Thread th : ts) th.join();
        System.out.printf("built n=%d with 8 threads in %.1fs%n", n, (System.nanoTime() - t0) / 1e9);
        boolean[] del = new boolean[n];
        for (int i = 0; i < n; i++) if (i % 5 < 2) { idx.delete(i); del[i] = true; }
        int nq = 3000; float[][] qs = new float[nq][]; for (int i = 0; i < nq; i++) qs[i] = g.vec(9_000_000L + i);
        for (int i = 0; i < 1000; i++) idx.search(qs[i], 10, null, 100); // warm
        long[] before = run(idx, qs, nq, del, null);
        AtomicBoolean done = new AtomicBoolean(); int[] rec = new int[1]; double[] ms = new double[1];
        Thread c = new Thread(() -> { long a = System.nanoTime(); rec[0] = idx.consolidate(); ms[0] = (System.nanoTime() - a) / 1e6; done.set(true); });
        long[] during = new long[400_000]; int dn = 0; boolean leak = false;
        c.start();
        while (!done.get() && dn < during.length) {
            float[] q = qs[dn % nq]; long a = System.nanoTime(); Hit[] h = idx.search(q, 10, null, 100); during[dn++] = System.nanoTime() - a;
            for (Hit x : h) if (del[(int) x.id()]) leak = true;
        }
        c.join();
        long[] after = run(idx, qs, nq, del, null);
        report("before", before, before.length); report("during", during, dn); report("after", after, after.length);
        System.out.printf("consolidate recycled=%d in %.0f ms; queries during=%d; deleted id leaked=%b; size=%d bytes=%.1f MB%n", rec[0], ms[0], dn, leak, idx.size(), idx.bytesHeld() / 1e6);
        GraphT.check(!leak, "no deleted id returned during consolidate");
        GraphT.check(rec[0] == n * 2 / 5, "recycled all tombstones " + rec[0]);
        System.out.printf("RESULT checks=%d failures=%d%n", GraphT.checks, GraphT.failures);
        if (GraphT.exitOnFailure) System.exit(GraphT.failures == 0 ? 0 : 1);
    }

    static long[] run(GraphIndex idx, float[][] qs, int nq, boolean[] del, Object unused) {
        long[] l = new long[nq];
        for (int i = 0; i < nq; i++) { long a = System.nanoTime(); idx.search(qs[i], 10, null, 100); l[i] = System.nanoTime() - a; }
        return l;
    }
    static void report(String name, long[] l, int n) {
        long[] s = GraphT.sorted(l, n);
        System.out.printf("  %-7s n=%-7d p50=%d us p99=%d us p99.9=%d us max=%d us%n", name, n, GraphT.pct(s, .5) / 1000, GraphT.pct(s, .99) / 1000, GraphT.pct(s, .999) / 1000, s.length == 0 ? 0 : s[s.length - 1] / 1000);
    }
}
