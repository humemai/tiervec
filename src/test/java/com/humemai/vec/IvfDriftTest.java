package com.humemai.vec;

/**
 * Update-heavy drift test: insert n vectors from distribution A (lists trained on exactly those), add n more from a shifted distribution B
 * (other cluster centers) WITHOUT retraining, report recall and latency before and after retrain(). Shows what drift costs and that
 * needsRetrain() notices it. Arg: n (default 20000).
 */
public final class IvfDriftTest {
    static final int DIM = 128;

    public static void main(String[] args) {
        int n = args.length > 0 ? Integer.parseInt(args[0]) : 20000;
        int nq = 200;
        System.out.println("IvfDriftTest n=" + n + " kernels=" + (Kernels.VECTOR ? "vector" : "scalar"));
        float[][] a = IvfT.clustered(n + nq, DIM, 64, 1.0, 1, 2, 0f);
        float[][] b = IvfT.clustered(n + nq, DIM, 64, 1.0, 3, 4, 0.1f);
        Metric m = Metric.COSINE;
        long[] ids = IvfT.seqIds(2 * n);
        float[][] st = new float[2 * n][];
        float[][] sa = IvfT.stored(m, a, 0, n), sb = IvfT.stored(m, b, 0, n);
        for (int i = 0; i < n; i++) {
            st[i] = sa[i];
            st[n + i] = sb[i];
        }
        float[][] qa = new float[nq][], qb = new float[nq][];
        for (int i = 0; i < nq; i++) {
            qa[i] = a[n + i];
            qb[i] = b[n + i];
        }
        IvfIndex.Params p = new IvfIndex.Params();
        p.trainAt = n; // trained when the last A vector arrives
        IvfIndex ix = new IvfIndex(DIM, m, p);
        for (int i = 0; i < n; i++) ix.add(i, a[i]);
        IvfT.check(ix.isTrained(), "trained at n");
        IvfT.check(ix.nlist() == (int) Math.round(4 * Math.sqrt(n)), "nlist before, got " + ix.nlist());
        boolean[] onlyA = new boolean[2 * n];
        for (int i = 0; i < n; i++) onlyA[i] = true;
        Hit[][] ta0 = IvfT.truths(m, ids, st, onlyA, qa, 10, null);
        double[] base = IvfT.eval(ix, qa, ta0, 0, 10);
        System.out.println("DRIFT before the shift (A only): nlist " + ix.nlist() + " recall@10 on A queries " + String.format("%.4f", base[0]) + " " + String.format("%.1f", base[1]) + " us/query; drift ratio " + String.format("%.3f", ix.driftRatio()) + ", skew " + String.format("%.2f", ix.listSkew()) + ", needsRetrain " + ix.needsRetrain());
        IvfT.check(base[0] >= 0.9, "baseline recall " + base[0]);
        IvfT.check(!ix.needsRetrain(), "no retrain needed on the training distribution (drift " + ix.driftRatio() + ", skew " + ix.listSkew() + ")");

        for (int i = 0; i < n; i++) ix.add(n + i, b[i]);
        IvfT.check(ix.size() == 2 * n, "size after the shifted inserts");
        Hit[][] ta = IvfT.truths(m, ids, st, null, qa, 10, null);
        Hit[][] tb = IvfT.truths(m, ids, st, null, qb, 10, null);
        int[] nps = {8, 16, 35, 64};
        double[][] before = report("after the shift, NO retrain", ix, qa, qb, ta, tb, nps);
        boolean need = ix.needsRetrain();
        IvfT.check(need, "needsRetrain notices the drift (drift " + ix.driftRatio() + ", skew " + ix.listSkew() + ")");
        IvfT.check(ix.driftRatio() > 1.5, "drift ratio above 1.5, got " + ix.driftRatio());

        long t0 = System.nanoTime();
        IvfT.check(ix.retrainIfNeeded(), "retrainIfNeeded ran");
        System.out.println("INFO retrain of " + 2 * n + " vectors: " + (System.nanoTime() - t0) / 1_000_000 + " ms");
        IvfT.check(ix.nlist() == (int) Math.round(4 * Math.sqrt(2 * n)), "nlist after, got " + ix.nlist());
        IvfT.check(ix.size() == 2 * n, "size after retrain");
        double[][] after = report("after retrain()", ix, qa, qb, ta, tb, nps);
        IvfT.check(!ix.needsRetrain(), "no retrain needed after retrain (drift " + ix.driftRatio() + ", skew " + ix.listSkew() + ")");
        IvfT.check(after[2][0] >= 0.9 && after[2][2] >= 0.9, "recall at nprobe 35 after retrain: A " + after[2][0] + ", B " + after[2][2]);
        IvfT.check(after[2][0] >= before[2][0] - 0.02 && after[2][2] >= before[2][2] - 0.02, "retrain does not lose recall at nprobe 35");
        IvfT.check(ix.lastError() == null, "no error");
        IvfT.done("IvfDriftTest");
    }

    /** returns rows per nprobe: {recall A, us A, recall B, us B} */
    static double[][] report(String label, IvfIndex ix, float[][] qa, float[][] qb, Hit[][] ta, Hit[][] tb, int[] nps) {
        int[] sz = ix.listSizes();
        int max = 0;
        for (int s : sz) max = Math.max(max, s);
        System.out.println("DRIFT " + label + ": nlist " + ix.nlist() + ", size " + ix.size() + ", largest list " + max + ", mean list " + String.format("%.1f", (double) ix.size() / ix.nlist()) + ", drift ratio " + String.format("%.3f", ix.driftRatio()) + ", skew " + String.format("%.2f", ix.listSkew()) + ", needsRetrain " + ix.needsRetrain());
        double[][] rows = new double[nps.length][];
        for (int i = 0; i < nps.length; i++) {
            double[] x = IvfT.eval(ix, qa, ta, nps[i], 10);
            double[] y = IvfT.eval(ix, qb, tb, nps[i], 10);
            rows[i] = new double[] {x[0], x[1], y[0], y[1]};
            System.out.println("DRIFT   nprobe " + nps[i] + ": A queries recall@10 " + String.format("%.4f", x[0]) + " " + String.format("%.1f", x[1]) + " us | B queries recall@10 " + String.format("%.4f", y[0]) + " " + String.format("%.1f", y[1]) + " us");
        }
        return rows;
    }
}
