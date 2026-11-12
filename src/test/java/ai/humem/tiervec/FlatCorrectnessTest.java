package ai.humem.tiervec;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.function.LongPredicate;

/** Single-threaded correctness: kernels, exact path vs brute force, delete, replace, filter, odd dims, quantized recall, compaction. */
public final class FlatCorrectnessTest {
    static float[] rnd(Random r, int dim) {
        float[] v = new float[dim];
        for (int i = 0; i < dim; i++) v[i] = (float) r.nextGaussian();
        return v;
    }

    static float[] unit(Random r, int dim) {
        return Kernels.normalized(rnd(r, dim));
    }

    /** brute-force reference using the same kernels on the same stored form; order: score desc, id asc */
    static Hit[] brute(Map<Long, float[]> model, Metric m, float[] q, int k, LongPredicate allow) {
        float[] qq = m == Metric.COSINE ? Kernels.normalized(q) : q;
        List<Hit> all = new ArrayList<>();
        for (Map.Entry<Long, float[]> e : model.entrySet()) {
            if (allow != null && !allow.test(e.getKey())) continue;
            float[] v = m == Metric.COSINE ? Kernels.normalized(e.getValue()) : e.getValue();
            float s = m == Metric.L2 ? -Kernels.l2sq(qq, 0, v, 0, v.length) : Kernels.dot(qq, 0, v, 0, v.length);
            all.add(new Hit(e.getKey(), s));
        }
        all.sort((a, b) -> a.score() != b.score() ? Float.compare(b.score(), a.score()) : Long.compare(a.id(), b.id()));
        return all.subList(0, Math.min(k, all.size())).toArray(new Hit[0]);
    }

    static boolean same(Hit[] a, Hit[] b) {
        if (a.length != b.length) return false;
        for (int i = 0; i < a.length; i++) if (a[i].id() != b[i].id() || a[i].score() != b[i].score()) return false;
        return true;
    }

    static double scalarScore(Metric m, float[] a, float[] b) {
        double dot = 0, na = 0, nb = 0, l2 = 0;
        for (int i = 0; i < a.length; i++) {
            dot += (double) a[i] * b[i];
            na += (double) a[i] * a[i];
            nb += (double) b[i] * b[i];
            double d = (double) a[i] - b[i];
            l2 += d * d;
        }
        return switch (m) {
            case DOT -> dot;
            case L2 -> -l2;
            case COSINE -> na == 0 || nb == 0 ? 0 : dot / Math.sqrt(na * nb);
        };
    }

    public static void main(String[] args) {
        int p0 = T.pass, f0 = T.fail;
        Random r = new Random(42);

        // 1. kernels vs scalar double, lengths around lane widths
        int[] lens = {1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 33, 63, 64, 65, 100, 127, 128, 129, 131, 255, 300, 768, 1000, 1536};
        for (int n : lens) {
            float[] a = rnd(r, n), b = rnd(r, n);
            double d = 0, l = 0;
            for (int i = 0; i < n; i++) {
                d += (double) a[i] * b[i];
                l += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
            }
            T.check(Math.abs(Kernels.dot(a, 0, b, 0, n) - d) < 1e-3 * Math.sqrt(n), "dot n=" + n);
            T.check(Math.abs(Kernels.l2sq(a, 0, b, 0, n) - l) < 1e-3 * n, "l2 n=" + n);
            // with offsets
            float[] a2 = new float[n + 5], b2 = new float[n + 9];
            System.arraycopy(a, 0, a2, 5, n);
            System.arraycopy(b, 0, b2, 9, n);
            T.check(Kernels.dot(a2, 5, b2, 9, n) == Kernels.dot(a, 0, b, 0, n), "dot offsets bitwise n=" + n);
            byte[] x = new byte[n + 3], y = new byte[n + 1];
            int ref = 0;
            for (int i = 0; i < n; i++) {
                x[3 + i] = (byte) (r.nextInt(255) - 127);
                y[1 + i] = (byte) (r.nextInt(255) - 127);
                ref += x[3 + i] * y[1 + i];
            }
            T.check(Kernels.dotI8(x, 3, y, 1, n) == ref, "dotI8 n=" + n);
            // extremes
            Arrays.fill(x, (byte) -127);
            Arrays.fill(y, (byte) -127);
            T.check(Kernels.dotI8(x, 3, y, 1, n) == 127 * 127 * n, "dotI8 extreme n=" + n);
        }
        long[] ba = {0xFFFFFFFFFFFFFFFFL, 0L, 5L}, bb = {0L, 0L, 3L};
        T.check(Kernels.hamming(ba, 0, bb, 0, 3) == 64 + 2, "hamming");
        T.section("kernels", p0, f0);

        // 2. exact path equals brute force: churn, filter, odd dims, all metrics
        p0 = T.pass; f0 = T.fail;
        for (int dim : new int[] {1, 5, 100, 128, 131}) {
            for (Metric m : Metric.values()) {
                FlatIndex idx = new FlatIndex(dim, m);
                Map<Long, float[]> model = new HashMap<>();
                for (int i = 0; i < 3000; i++) {
                    long id = 1000 + i;
                    float[] v = rnd(r, dim);
                    idx.add(id, v);
                    model.put(id, v);
                }
                Set<Long> deleted = new HashSet<>();
                for (int i = 0; i < 600; i++) { // deletes
                    long id = 1000 + r.nextInt(3000);
                    boolean existed = model.remove(id) != null;
                    T.check(idx.delete(id) == existed, "delete return dim=" + dim);
                    if (existed) deleted.add(id);
                }
                for (int i = 0; i < 300; i++) { // replaces of live ids
                    List<Long> keys = new ArrayList<>(model.keySet());
                    long id = keys.get(r.nextInt(keys.size()));
                    float[] v = rnd(r, dim);
                    idx.add(id, v);
                    model.put(id, v);
                }
                for (int i = 0; i < 100; i++) { // new ids after deletes
                    long id = 10000 + i;
                    float[] v = rnd(r, dim);
                    idx.add(id, v);
                    model.put(id, v);
                }
                T.check(idx.size() == model.size(), "size dim=" + dim + " " + m);
                boolean allEq = true, noDeleted = true, scoresOk = true;
                LongPredicate filter = id -> id % 3 != 0;
                for (int qi = 0; qi < 20; qi++) {
                    float[] q = rnd(r, dim);
                    for (int k : new int[] {1, 10, 50}) {
                        Hit[] got = idx.search(q, k, null);
                        allEq &= same(got, brute(model, m, q, k, null));
                        for (Hit h : got) noDeleted &= !deleted.contains(h.id()) || model.containsKey(h.id());
                        Hit[] gf = idx.search(q, k, filter);
                        allEq &= same(gf, brute(model, m, q, k, filter));
                        for (Hit h : gf) allEq &= filter.test(h.id());
                        for (Hit h : got) scoresOk &= Math.abs(scalarScore(m, q, model.get(h.id())) - h.score()) < 1e-3 * (1 + Math.abs(h.score()));
                    }
                }
                T.check(allEq, "exact==brute dim=" + dim + " " + m);
                T.check(noDeleted, "deleted not returned dim=" + dim + " " + m);
                T.check(scoresOk, "scores close to scalar double dim=" + dim + " " + m);
                // k larger than live, and allow that matches nothing
                T.check(idx.search(rnd(r, dim), 100000, null).length == model.size(), "k>n returns all dim=" + dim);
                T.check(idx.search(rnd(r, dim), 10, id -> false).length == 0, "empty filter");
                T.check(new FlatIndex(dim, m).search(rnd(r, dim), 10, null).length == 0, "empty index");
            }
        }
        T.section("exact path vs brute force, deletes, replaces, filter, dims 1/5/100/128/131", p0, f0);

        // 3. replace and delete semantics
        p0 = T.pass; f0 = T.fail;
        {
            FlatIndex idx = new FlatIndex(8, Metric.DOT);
            float[] a = {1, 0, 0, 0, 0, 0, 0, 0}, b = {0, 1, 0, 0, 0, 0, 0, 0};
            idx.add(7, a);
            idx.add(7, b); // replace
            T.check(idx.size() == 1, "replace keeps size 1");
            Hit[] h = idx.search(a, 5, null);
            T.check(h.length == 1 && h[0].score() == 0f, "replaced vector is the new one");
            h = idx.search(b, 5, null);
            T.check(h.length == 1 && h[0].id() == 7 && h[0].score() == 1f, "new vector found");
            T.check(idx.delete(7) && !idx.delete(7) && idx.size() == 0, "delete twice");
            T.check(idx.search(b, 5, null).length == 0, "deleted never returned");
            idx.add(7, a); // re-add after delete
            T.check(idx.size() == 1 && idx.search(a, 1, null)[0].id() == 7, "re-add after delete");
            float[] z = new float[8];
            FlatIndex ci = new FlatIndex(8, Metric.COSINE);
            ci.add(1, z); // zero vector must not produce NaN
            ci.add(2, a);
            Hit[] zh = ci.search(a, 2, null);
            T.check(zh.length == 2 && zh[0].id() == 2 && !Float.isNaN(zh[1].score()), "zero vector cosine");
            try {
                idx.add(1, new float[3]);
                T.check(false, "wrong dim must throw");
            } catch (IllegalArgumentException e) {
                T.check(true, "wrong dim throws");
            }
        }
        T.section("replace/delete semantics", p0, f0);

        // 4. quantized: returned scores are exact, recall, filter, tiny index, pre-calibration, L2
        p0 = T.pass; f0 = T.fail;
        for (Quant qt : new Quant[] {Quant.SQ8, Quant.BINARY}) {
            for (Sq8Kernel kv : qt == Quant.SQ8 ? Sq8Kernel.values() : new Sq8Kernel[] {Sq8Kernel.DEFAULT})
            for (Metric m : Metric.values()) {
                int dim = 131; // not a multiple of 8, 16 or 64
                int n = 6000;
                FlatIndex idx = new FlatIndex(dim, m, qt, 0, kv);
                FlatIndex ex = new FlatIndex(dim, m);
                Map<Long, float[]> model = new HashMap<>();
                // clustered data (more like embeddings than isotropic noise) with a positive offset to exercise the binary thresholds
                float[][] centers = new float[20][];
                for (int c = 0; c < 20; c++) centers[c] = rnd(r, dim);
                for (int i = 0; i < n; i++) {
                    float[] v = centers[r.nextInt(20)].clone();
                    for (int j = 0; j < dim; j++) v[j] += 0.6f * (float) r.nextGaussian() + 0.5f;
                    idx.add(i, v);
                    ex.add(i, v);
                    model.put((long) i, v);
                }
                for (int i = 0; i < 500; i++) { // churn
                    long id = r.nextInt(n);
                    if (model.remove(id) != null) {
                        idx.delete(id);
                        ex.delete(id);
                    }
                }
                double rec = 0;
                int nq = 40;
                boolean exactScores = true, filtOk = true;
                LongPredicate filter = id -> id % 2 == 0;
                for (int qi = 0; qi < nq; qi++) {
                    float[] q = centers[r.nextInt(20)].clone();
                    for (int j = 0; j < dim; j++) q[j] += 0.6f * (float) r.nextGaussian() + 0.5f;
                    Hit[] g = idx.search(q, 10, null), e = ex.search(q, 10, null);
                    Set<Long> es = new HashSet<>();
                    for (Hit h : e) es.add(h.id());
                    int hit = 0;
                    for (Hit h : g) if (es.contains(h.id())) hit++;
                    rec += hit / 10.0;
                    for (Hit h : g) {
                        float[] qq = m == Metric.COSINE ? Kernels.normalized(q) : q;
                        float[] v = m == Metric.COSINE ? Kernels.normalized(model.get(h.id())) : model.get(h.id());
                        float s = m == Metric.L2 ? -Kernels.l2sq(qq, 0, v, 0, dim) : Kernels.dot(qq, 0, v, 0, dim);
                        exactScores &= s == h.score();
                    }
                    for (Hit h : idx.search(q, 10, filter)) filtOk &= filter.test(h.id());
                    T.check(idx.search(q, 10, filter).length == 10, "filter fills k " + qt);
                }
                rec /= nq;
                System.out.println("  recall@10 " + qt + (qt == Quant.SQ8 ? "/" + kv : "") + " " + m + " n=" + (n - 500) + " dim=" + dim + " rerank=default: " + String.format("%.3f", rec));
                T.check(exactScores, "rerank scores are exact fp32 " + qt + " " + m);
                T.check(filtOk, "allow-list respected " + qt + " " + m);
                T.check(rec >= (qt == Quant.SQ8 ? 0.95 : (m == Metric.COSINE ? 0.6 : 0.5)), "recall sanity " + qt + " " + m + " got " + rec);
            }
            // before calibration (n < 512): exact answers
            FlatIndex small = new FlatIndex(40, Metric.COSINE, qt);
            Map<Long, float[]> sm = new HashMap<>();
            for (int i = 0; i < 300; i++) {
                float[] v = rnd(r, 40);
                small.add(i, v);
                sm.put((long) i, v);
            }
            float[] q = rnd(r, 40);
            T.check(same(small.search(q, 10, null), brute(sm, Metric.COSINE, q, 10, null)), "small index exact " + qt);
        }
        T.section("quantized SQ8/BINARY: exact rerank scores, recall, filter, small index", p0, f0);

        // 5. compaction
        p0 = T.pass; f0 = T.fail;
        for (Quant qt : Quant.values()) {
            int dim = 100;
            FlatIndex idx = new FlatIndex(dim, Metric.L2, qt, 1000); // rerank huge: quantized path degenerates to exact, so results are comparable
            Map<Long, float[]> model = new HashMap<>();
            for (int i = 0; i < 20000; i++) {
                float[] v = rnd(r, dim);
                idx.add(i, v);
                model.put((long) i, v);
            }
            for (int i = 0; i < 20000; i++) if (i % 4 != 0) {
                idx.delete(i);
                model.remove((long) i);
            }
            long before = idx.bytesHeld();
            double dead = idx.deadFraction();
            float[] q = rnd(r, dim);
            Hit[] ref = brute(model, Metric.L2, q, 20, null);
            T.check(same(idx.search(q, 20, null), ref), "pre-compact exact " + qt);
            idx.compact();
            T.check(idx.size() == model.size() && idx.slotsUsed() == model.size(), "compact size " + qt);
            T.check(same(idx.search(q, 20, null), ref), "post-compact exact " + qt);
            T.check(idx.bytesHeld() < before * 0.5, "bytesHeld shrinks " + qt + " " + before + " -> " + idx.bytesHeld() + " dead=" + dead);
            idx.add(5, rnd(r, dim)); // an id that was deleted, adds fine after compaction
            T.check(idx.delete(4) && !idx.delete(4), "post-compact delete of a kept id, twice");
            T.check(!idx.delete(8 + 1), "post-compact delete of a dropped id");
            T.check(idx.size() == model.size(), "post-compact size (added id 5, deleted id 4)");
        }
        T.section("compaction", p0, f0);

        // 6. bytesHeld sanity
        p0 = T.pass; f0 = T.fail;
        {
            FlatIndex a = new FlatIndex(768, Metric.COSINE), s = new FlatIndex(768, Metric.COSINE, Quant.SQ8), b = new FlatIndex(768, Metric.COSINE, Quant.BINARY);
            for (int i = 0; i < 10000; i++) {
                float[] v = rnd(r, 768);
                a.add(i, v);
                s.add(i, v);
                b.add(i, v);
            }
            long fp32 = 10000L * 768 * 4;
            System.out.println("  bytesHeld n=10000 dim=768: NONE " + a.bytesHeld() + " SQ8 " + s.bytesHeld() + " BINARY " + b.bytesHeld() + " (raw fp32 " + fp32 + ")");
            T.check(a.bytesHeld() >= fp32 && a.bytesHeld() < fp32 * 1.5, "NONE ~ fp32");
            T.check(s.bytesHeld() > a.bytesHeld() + 10000L * 768 && b.bytesHeld() > a.bytesHeld(), "codes counted");
        }
        T.section("bytesHeld", p0, f0);
        extra(r);
        T.done("FlatCorrectnessTest");
    }

    /** kernels against their scalar twins, new SQ8 kernels, compaction policy, serialize/load */
    static void extra(Random r) {
        int p0 = T.pass, f0 = T.fail;
        System.out.println("  Kernels.VECTOR=" + Kernels.VECTOR + (Kernels.VECTOR ? " lanes=" + VKernels.FL : " (" + Kernels.scalarReason + ")"));
        T.check(Kernels.VECTOR || Kernels.scalarReason != null, "scalar mode always has a reason");
        T.check(!(Kernels.VECTOR && Boolean.getBoolean("ai.humem.tiervec.scalar")), "the scalar flag is honoured");
        T.check(Kernels.VECTOR || Boolean.getBoolean("ai.humem.tiervec.scalar"), "Vector API must work in the non-scalar run: " + Kernels.scalarReason);
        for (int n : new int[] {1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 128, 131, 255, 768, 1536}) {
            float[] a = rnd(r, n), b = rnd(r, n);
            double tol = 1e-3 * Math.sqrt(n) + 1e-4;
            T.check(Math.abs(Kernels.dot(a, 0, b, 0, n) - Kernels.dotScalar(a, 0, b, 0, n)) < tol, "dot vs scalar twin n=" + n);
            T.check(Math.abs(Kernels.l2sq(a, 0, b, 0, n) - Kernels.l2sqScalar(a, 0, b, 0, n)) < tol * 4, "l2 vs scalar twin n=" + n);
            byte[] x = new byte[n + 3], y = new byte[n + 1];
            for (int i = 0; i < x.length; i++) x[i] = (byte) (r.nextInt(255) - 127);
            for (int i = 0; i < y.length; i++) y[i] = (byte) (r.nextInt(255) - 127);
            T.check(Kernels.dotI8(x, 3, y, 1, n) == Kernels.dotI8Scalar(x, 3, y, 1, n), "dotI8 vs scalar twin n=" + n);
            double ref = 0;
            for (int i = 0; i < n; i++) ref += (double) a[i] * x[3 + i];
            float f8 = Kernels.dotF8(a, 0, x, 3, n), f84 = Kernels.dotF8x4(a, 0, x, 3, n), f8s = Kernels.dotF8Scalar(a, 0, x, 3, n);
            double t8 = 1e-5 * 127 * Math.sqrt(n) * 40 + 1e-3;
            T.check(Math.abs(f8 - ref) < t8 && Math.abs(f84 - ref) < t8 && Math.abs(f8s - ref) < t8, "dotF8, dotF8x4, scalar vs double n=" + n);
            // block16: 16 vectors, dimension-major
            byte[] blk = new byte[n * 16 + 5];
            for (int i = 5; i < blk.length; i++) blk[i] = (byte) (r.nextInt(255) - 127);
            float[] out = new float[16], outS = new float[16];
            Kernels.block16(a, 0, blk, 5, n, out);
            Kernels.block16Scalar(a, 0, blk, 5, n, outS);
            boolean ok = true;
            for (int lane = 0; lane < 16; lane++) {
                double d = 0;
                for (int j = 0; j < n; j++) d += (double) a[j] * blk[5 + j * 16 + lane];
                ok &= Math.abs(out[lane] - d) < t8 && Math.abs(outS[lane] - d) < t8;
            }
            T.check(ok, "block16 vs double n=" + n);
        }
        T.section("kernel twins and SQ8 kernels", p0, f0);

        // compaction policy
        p0 = T.pass; f0 = T.fail;
        {
            FlatIndex idx = new FlatIndex(32, Metric.DOT);
            for (int i = 0; i < 5000; i++) idx.add(i, rnd(r, 32));
            T.check(!idx.compactIfNeeded() && idx.compactions() == 0, "no dead slots, no compaction");
            for (int i = 0; i < 500; i++) idx.delete(i);
            T.check(!idx.compactIfNeeded(), "500 dead < minDeadSlots");
            for (int i = 500; i < 2000; i++) idx.delete(i); // 2000 dead of 5000 = 0.4 >= 0.3
            T.check(idx.compactIfNeeded() && idx.compactions() == 1 && idx.slotsUsed() == 3000 && idx.size() == 3000, "dead fraction triggers");
            T.check(!idx.compactIfNeeded(), "nothing due right after");
            // built-in daemon under churn keeps the slot count bounded and the answers right
            FlatIndex d = new FlatIndex(32, Metric.COSINE);
            d.setCompactionPolicy(0.3, 256, 2.0);
            d.startAutoCompaction(10);
            Map<Long, float[]> model = new HashMap<>();
            long end = System.nanoTime() + 2_000_000_000L;
            long next = 0;
            int maxSlots = 0;
            while (System.nanoTime() < end) {
                long id = next++;
                float[] v = rnd(r, 32);
                d.add(id, v);
                model.put(id, v);
                if (id >= 1000) {
                    d.delete(id - 1000);
                    model.remove(id - 1000);
                }
                maxSlots = Math.max(maxSlots, d.slotsUsed());
            }
            T.check(d.lastCompactionError() == null, "daemon raised no error");
            T.check(d.compactions() > 0, "daemon compacted (" + d.compactions() + ")");
            T.check(maxSlots < 100000, "slots stay bounded under churn (without the daemon this run writes about a million), max " + maxSlots + " for 1000 live");
            d.close();
            float[] q = rnd(r, 32);
            T.check(same(d.search(q, 10, null), brute(model, Metric.COSINE, q, 10, null)), "exact after daemon compactions");
        }
        T.section("compaction policy and daemon", p0, f0);

        // serialize and load
        p0 = T.pass; f0 = T.fail;
        for (Quant qt : Quant.values()) {
            for (Sq8Kernel kv : qt == Quant.SQ8 ? Sq8Kernel.values() : new Sq8Kernel[] {Sq8Kernel.DEFAULT}) {
                for (Metric m : new Metric[] {Metric.COSINE, Metric.L2}) {
                    int dim = 70;
                    FlatIndex a = new FlatIndex(dim, m, qt, 0, kv);
                    for (int i = 0; i < 2500; i++) a.add(i * 7L + 1, rnd(r, dim));
                    for (int i = 0; i < 600; i++) a.delete(r.nextInt(2500) * 7L + 1);
                    for (int i = 0; i < 100; i++) a.add(r.nextInt(2500) * 7L + 1, rnd(r, dim)); // replaces and re-adds
                    java.io.ByteArrayOutputStream bos = new java.io.ByteArrayOutputStream();
                    FlatIndex b = null;
                    try {
                        a.serialize(bos);
                        byte[] bytes = bos.toByteArray();
                        b = FlatIndex.load(new java.io.ByteArrayInputStream(bytes));
                        String tag = qt + "/" + kv + "/" + m;
                        T.check(b.size() == a.size() && b.dim() == dim && b.metric() == m, "load size, dim, metric " + tag);
                        boolean same = true;
                        for (int i = 0; i < 20; i++) {
                            float[] q = rnd(r, dim);
                            same &= same(a.search(q, 10, null), b.search(q, 10, null));
                            LongPredicate f = id -> id % 2 == 0;
                            same &= same(a.search(q, 10, f), b.search(q, 10, f));
                        }
                        T.check(same, "loaded index answers identically " + tag);
                        T.check(bytes.length == 8 + 4 * 6 + 1 + 4 + (qt == Quant.BINARY ? dim * 4 : 0) + a.size() * (8 + dim * 4) + 8, "file size " + tag + " " + bytes.length);
                        if (kv == Sq8Kernel.DEFAULT && m == Metric.COSINE) {
                            byte[] bad = bytes.clone();
                            bad[bad.length / 2] ^= 0x10;
                            T.check(loadFails(bad), "flipped payload byte is rejected " + qt);
                            T.check(loadFails(Arrays.copyOf(bytes, bytes.length - 3)), "truncated file is rejected " + qt);
                            T.check(loadFails(Arrays.copyOf(bytes, 20)), "short file is rejected " + qt);
                            byte[] badMagic = bytes.clone();
                            badMagic[0] ^= 1;
                            T.check(loadFails(badMagic), "bad magic is rejected " + qt);
                            byte[] badCrc = bytes.clone();
                            badCrc[badCrc.length - 1] ^= 1;
                            T.check(loadFails(badCrc), "bad checksum is rejected " + qt);
                        }
                    } catch (Exception e) {
                        T.check(false, "serialize/load " + qt + "/" + kv + "/" + m + " threw " + e);
                    }
                }
            }
        }
        // an empty index round-trips
        try {
            java.io.ByteArrayOutputStream bos = new java.io.ByteArrayOutputStream();
            new FlatIndex(9, Metric.DOT).serialize(bos);
            FlatIndex e = FlatIndex.load(new java.io.ByteArrayInputStream(bos.toByteArray()));
            T.check(e.size() == 0 && e.search(new float[9], 3, null).length == 0, "empty index round-trips");
        } catch (Exception e) {
            T.check(false, "empty round trip threw " + e);
        }
        T.section("serialize and load", p0, f0);
    }

    static boolean loadFails(byte[] bytes) {
        try {
            FlatIndex.load(new java.io.ByteArrayInputStream(bytes));
            return false;
        } catch (java.io.IOException e) {
            return true;
        } catch (RuntimeException e) {
            return false; // a corrupt file must surface as IOException, not as an unchecked exception
        }
    }
}
