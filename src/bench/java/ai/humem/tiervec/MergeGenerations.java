package ai.humem.tiervec;

import java.io.EOFException;
import java.io.IOException;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.TreeMap;
import java.util.zip.CRC32;

/**
 * Many merges of one tier in a row, to compare the two merge modes ({@link TieredIndex.MergeMode}, issue #16; docs/BENCHMARKS.md, "Successive
 * merges"). Maven does not compile it: it is compiled with the library sources, like {@link Bench}.
 *
 * <p>The run draws rows of a prepared real data set ({@link RealData}, {@code --data}) in an order that depends on {@code --seed} only, builds the
 * first {@code --base} of them with one {@code rebuildFrom}, and then runs {@code --generations} generations. A generation deletes
 * {@code --delete-pct} percent of the rows that are live at its start (chosen by the seed), adds {@code --add-pct} percent of the base count from
 * the rows that were not used yet, seals the tail and calls {@code mergeNow()}, which is timed. The tail limit and the segment cap are out of
 * reach and the dead share of a segment stays below {@code deadFraction}, so the worker of the tier never seals or merges by itself, and the timed
 * merge is the one that the harness calls. The writes depend on the seed and the options, not on the merge mode.
 *
 * <p>After every merge the run measures recall against the exact answers over the live rows, which the harness keeps up to date with its own scalar
 * dot product, and after the last generation it times the searches. The rows are held outside the heap, so the heap holds the tier, the queries and
 * the exact answers. A garbage collection is forced before and after each timed merge, and the line of a generation has the uptime of the JVM
 * around the collections and the merge, so a garbage collection log with the {@code uptimemillis} decoration can be matched to the merge.
 *
 * <p>Options are {@code --key value} pairs, see {@link #defaults()}. The exit code is 1 when a check fails (an exception, a merge that did not take
 * the path of its mode, a wrong live count, a deleted or malformed answer, a recall below {@code --min-recall}), and 2 for a wrong option.
 */
public final class MergeGenerations {
    private MergeGenerations() {}

    /** out of reach, so the worker never seals the tail */
    static final int TAIL_LIMIT = 100_000_000;
    /** out of reach, so the count rule never merges */
    static final int MAX_SEGMENTS = 1000;
    /** below {@code deadFraction} (0.3), so that no merge of this program falls back to a rebuild because of hidden rows */
    static final double MAX_DELETE_PCT = 25;
    /** how many exact answers are kept for each query; deletes shorten a list, and a list shorter than k is computed again */
    static final int KEEP = 100;
    static final int BLOCK = 64;

    static Map<String, String> defaults() {
        Map<String, String> d = new LinkedHashMap<>();
        d.put("data", "");              // required: a prepared real data set, see RealData
        d.put("base", "100000");        // rows of the first segment
        d.put("add-pct", "10");         // rows added per generation, percent of the base
        d.put("delete-pct", "0");       // rows deleted per generation, percent of the rows live at its start (at most 25)
        d.put("generations", "10");
        d.put("merge-mode", "rebuild"); // rebuild or insert
        d.put("seed", "42");            // the order of the rows and the choice of the deletes
        d.put("queries", "200");
        d.put("k", "10");
        d.put("ef", "100");
        d.put("efc", "200");
        d.put("m", "16");
        d.put("build-threads", "2");    // threads of the first build, of a seal and of a merge
        d.put("reps", "5");             // timed passes over the queries at the end, after one warm-up pass
        d.put("threads", "0");          // threads of the exact scans; 0 is availableProcessors
        d.put("min-recall", "0");       // floor of the recall of every generation; 0 is no floor
        d.put("host-label", "");        // label of the machine in the file; empty is the host name
        d.put("repo", ".");             // directory for git rev-parse
        d.put("out", "");               // result file; empty is standard output only
        return d;
    }

    static Map<String, String> options(String[] a) {
        Map<String, String> m = defaults();
        for (int i = 0; i < a.length; i += 2) {
            if (!a[i].startsWith("--") || !m.containsKey(a[i].substring(2)) || i + 1 >= a.length)
                throw new IllegalArgumentException("bad option " + a[i] + " (options are --key value pairs: " + defaults().keySet() + ")");
            m.put(a[i].substring(2), a[i + 1]);
        }
        return m;
    }

    public static void main(String[] args) throws Exception {
        Run run;
        try {
            run = new Run(options(args));
        } catch (IllegalArgumentException e) {
            System.err.println(e.getMessage());
            System.exit(2);
            return;
        }
        System.exit(run.go());
    }

    /** the first n entries of a random order of the rows [0, count); an entry does not depend on n */
    static int[] drawOrder(int count, int n, long seed) {
        int[] p = new int[count];
        for (int i = 0; i < count; i++) p[i] = i;
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = 0; i < n; i++) swap(p, i, i + r.nextInt(count - i));
        return Arrays.copyOf(p, n);
    }

    static void swap(int[] a, int i, int j) {
        int t = a[i];
        a[i] = a[j];
        a[j] = t;
    }

    static void crc(CRC32 c, int v) {
        for (int shift = 0; shift < 32; shift += 8) c.update(v >>> shift);
    }

    static long uptimeMs() {
        return ManagementFactory.getRuntimeMXBean().getUptime();
    }

    /** the rows that a run uses, outside the heap: row s of this object is the row {@code order[s]} of the file */
    static final class Rows {
        private final FloatBuffer values;
        private final int dim;

        private Rows(FloatBuffer values, int dim) {
            this.values = values;
            this.dim = dim;
        }

        static Rows read(Path file, int[] order, int dim) throws IOException {
            final int rowBytes = dim * 4;
            ByteBuffer all = ByteBuffer.allocateDirect(Math.multiplyExact(order.length, rowBytes));
            try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
                for (int s = 0; s < order.length; s++) {
                    ByteBuffer dst = all.duplicate();
                    dst.limit((s + 1) * rowBytes);
                    dst.position(s * rowBytes);
                    long at = (long) order[s] * rowBytes;
                    while (dst.hasRemaining()) {
                        if (ch.read(dst, at + dst.position() - (long) s * rowBytes) < 0) throw new EOFException(file + " ends early at row " + order[s]);
                    }
                }
            }
            Rows rows = new Rows(all.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer(), dim);
            float[] x = new float[dim];
            for (int s = 0; s < order.length; s++) {
                rows.get(s, x);
                double norm = 0;
                for (float f : x) norm += (double) f * f;
                if (!(Math.abs(Math.sqrt(norm) - 1.0) <= 1e-3))
                    throw new IllegalArgumentException(file + ": row " + order[s] + " has length " + Math.sqrt(norm) + ", not 1 (scripts/prep_real_dataset.py normalises the vectors)");
            }
            return rows;
        }

        void get(int s, float[] into) {
            values.get(s * dim, into);
        }

        float[] row(int s) {
            float[] x = new float[dim];
            get(s, x);
            return x;
        }
    }

    /**
     * The exact best rows of one query, by score (higher first) and then by id (lower first). The list is always a prefix of that order over the
     * live rows: a delete removes an entry, and an added row enters only when it is better than the last entry that was there before the add.
     */
    static final class Best {
        final float[] score;
        final int[] id;
        int n;

        Best(int cap) {
            score = new float[cap];
            id = new int[cap];
        }

        static boolean better(float s, int i, float otherScore, int otherId) {
            return s > otherScore || (s == otherScore && i < otherId);
        }

        void offer(float s, int i) {
            if (n == score.length) {
                if (!better(s, i, score[n - 1], id[n - 1])) return;
                n--;
            }
            int p = n;
            while (p > 0 && better(s, i, score[p - 1], id[p - 1])) {
                score[p] = score[p - 1];
                id[p] = id[p - 1];
                p--;
            }
            score[p] = s;
            id[p] = i;
            n++;
        }

        void drop(BitSet gone) {
            int w = 0;
            for (int r = 0; r < n; r++) {
                if (gone.get(id[r])) continue;
                score[w] = score[r];
                id[w] = id[r];
                w++;
            }
            n = w;
        }

        long[] top(int k) {
            long[] t = new long[Math.min(k, n)];
            for (int i = 0; i < t.length; i++) t[i] = id[i];
            return t;
        }
    }

    private static final class Run {
        final Map<String, String> o;
        final RealData real;
        final TieredIndex.MergeMode mode;
        final int base, add, generations, nq, k, ef, reps, threads, buildThreads, rows;
        final double deletePct, minRecall;
        final long seed;
        final List<String> failures = new ArrayList<>();
        final BitSet gone = new BitSet();
        Bench.Out out;
        Rows data;
        float[][] qs;
        Best[] best;
        TieredIndex tier;
        int[] live; // the live rows are the first liveCount entries
        int liveCount;
        int rescans; // queries whose exact list was computed again during the generations

        Run(Map<String, String> o) throws IOException {
            this.o = o;
            if (o.get("data").isEmpty()) throw new IllegalArgumentException("--data is required: the directory of a prepared real data set (docs/BENCHMARKS.md, \"Real data\")");
            real = RealData.open(Path.of(o.get("data")));
            String m = o.get("merge-mode");
            if (!m.equals("rebuild") && !m.equals("insert")) throw new IllegalArgumentException("--merge-mode is rebuild or insert, not " + m);
            mode = m.equals("insert") ? TieredIndex.MergeMode.INSERT : TieredIndex.MergeMode.REBUILD;
            base = num("base");
            generations = num("generations");
            nq = num("queries");
            k = num("k");
            ef = num("ef");
            reps = num("reps");
            buildThreads = num("build-threads");
            threads = num("threads") > 0 ? num("threads") : Runtime.getRuntime().availableProcessors();
            seed = Long.parseLong(o.get("seed"));
            deletePct = Double.parseDouble(o.get("delete-pct"));
            minRecall = Double.parseDouble(o.get("min-recall"));
            add = (int) Math.round(base * Double.parseDouble(o.get("add-pct")) / 100);
            if (base < TieredIndex.Config.defaults().smallLimit() || generations < 1 || add < 1 || reps < 1 || ef < 1 || buildThreads < 1 || k < 1 || k > KEEP)
                throw new IllegalArgumentException("--base is at least smallLimit, --generations, --reps, --ef and --build-threads at least 1, --add-pct gives at least one row, and --k is 1 to " + KEEP);
            if (!(deletePct >= 0 && deletePct <= MAX_DELETE_PCT)) throw new IllegalArgumentException("--delete-pct is 0 to " + MAX_DELETE_PCT + ", not " + deletePct);
            if (nq < 1 || nq > real.queries) throw new IllegalArgumentException("--queries is 1 to " + real.queries + " (what the data set holds), not " + nq);
            long need = base + (long) generations * add;
            if (need > real.count) throw new IllegalArgumentException("the run needs " + need + " rows, the data set holds " + real.count);
            if (need * real.dim * 4 > Integer.MAX_VALUE) throw new IllegalArgumentException("the rows of the run do not fit one buffer of 2 GiB");
            rows = (int) need;
        }

        private int num(String key) {
            return Integer.parseInt(o.get(key));
        }

        private void fail(String why) {
            failures.add(why);
            out.line("FAIL " + why);
        }

        private void check(boolean ok, int g, String why) {
            if (!ok) fail("generation " + g + ": " + why);
        }

        int go() throws Exception {
            out = new Bench.Out(o.get("out"));
            long t0 = System.nanoTime();
            try {
                header();
                load();
                build();
                for (int g = 1; g <= generations; g++) generation(g);
                truthCheck();
                search();
                out.line(Bench.f("MEMORY end rssPeakMB=%.0f", Bench.statusKb("VmHWM") / 1024.0));
            } catch (Throwable e) {
                fail("exception: " + e);
                e.printStackTrace();
            } finally {
                if (tier != null) tier.close();
            }
            for (String why : failures) out.line("GEN-FAILURE " + why);
            out.line(Bench.f("GEN-SUMMARY status=%s failures=%d wallSeconds=%.1f", failures.isEmpty() ? "OK" : "FAIL", failures.size(), (System.nanoTime() - t0) / 1e9));
            out.line("GEN-END");
            out.close();
            return failures.isEmpty() ? 0 : 1;
        }

        private void header() {
            out.line("GEN-RESULT format=1 harness=ai.humem.tiervec.MergeGenerations");
            String label = o.get("host-label").isEmpty() ? Bench.firstLine("/proc/sys/kernel/hostname") : o.get("host-label");
            out.line("RUN started=" + Instant.now() + " host=" + label);
            String sha = Bench.sh("git", "-C", o.get("repo"), "rev-parse", "HEAD");
            out.line("COMMIT sha=" + (sha == null ? "unknown" : sha));
            String allowed = "unknown";
            for (String l : Bench.readLines("/proc/self/status")) if (l.startsWith("Cpus_allowed_list:")) allowed = l.substring(18).trim();
            String first = allowed.split("[-,]")[0];
            int cpu = first.matches("[0-9]+") ? Integer.parseInt(first) : 0;
            out.line("CPU model=\"" + Bench.cpuinfoOf(cpu)[0] + "\" cpusAllowedList=" + allowed + " availableProcessors=" + Runtime.getRuntime().availableProcessors());
            out.line("CPUFREQ governor=" + Bench.firstLine("/sys/devices/system/cpu/cpu" + cpu + "/cpufreq/scaling_governor") + " noTurbo="
                    + Bench.firstLine("/sys/devices/system/cpu/intel_pstate/no_turbo"));
            out.line("RAM totalKiB=" + Bench.meminfoKb("MemTotal") + " availableKiB=" + Bench.meminfoKb("MemAvailable"));
            out.line("JDK version=" + System.getProperty("java.version") + " vendor=" + System.getProperty("java.vendor") + " runtime=" + System.getProperty("java.runtime.version"));
            StringBuilder gcs = new StringBuilder();
            for (GarbageCollectorMXBean b : ManagementFactory.getGarbageCollectorMXBeans()) gcs.append(gcs.length() == 0 ? "" : ",").append(b.getName());
            out.line("JVM maxHeapMiB=" + Runtime.getRuntime().maxMemory() / (1 << 20) + " collectors=" + gcs + " args=" + ManagementFactory.getRuntimeMXBean().getInputArguments());
            out.line("VECTORAPI module=" + ModuleLayer.boot().findModule("jdk.incubator.vector").isPresent() + " kernels=" + (Kernels.VECTOR ? "vector" : "scalar")
                    + " lanes=" + Kern.lanes() + " scalarProperty=" + System.getProperty("ai.humem.tiervec.scalar"));
            StringBuilder ps = new StringBuilder();
            for (Map.Entry<String, String> e : new TreeMap<>(o).entrySet()) ps.append(ps.length() == 0 ? "" : " ").append(e.getKey()).append('=').append(e.getValue());
            out.line("PARAMS " + ps);
            out.line("DATASET " + real.describe());
        }

        private void load() throws IOException {
            long t0 = System.nanoTime();
            int[] order = drawOrder(real.count, rows, seed);
            data = Rows.read(real.dir.resolve(RealData.VECTORS), order, real.dim);
            qs = real.readQueries(nq);
            CRC32 orderCrc = new CRC32();
            for (int row : order) crc(orderCrc, row);
            CRC32 firstCrc = new CRC32();
            for (int s = 0; s < Math.min(1000, rows); s++) for (float x : data.row(s)) crc(firstCrc, Float.floatToIntBits(x));
            out.line(Bench.f("DATA rows=%d base=%d addPerGeneration=%d queries=%d dim=%d orderCrc=%08x first1000Crc=%08x loadSeconds=%.1f", rows, base, add, nq, real.dim,
                    orderCrc.getValue(), firstCrc.getValue(), (System.nanoTime() - t0) / 1e9));
        }

        private void build() throws Exception {
            GraphIndex.Params gp0 = GraphIndex.Params.defaults().withM(num("m")).withEfConstruction(num("efc"));
            GraphIndex.Params gp = new GraphIndex.Params(gp0.M(), gp0.efConstruction(), ef, gp0.alpha(), false, gp0.seed());
            TieredIndex.Config d = TieredIndex.Config.defaults();
            TieredIndex.Config cfg = new TieredIndex.Config(TAIL_LIMIT, 0L, d.smallLimit(), MAX_SEGMENTS, d.mergeFan(), d.deadFraction(), 0, d.hardTailFactor(), Quant.NONE, gp,
                    buildThreads, false, 0.0, mode);
            out.line("CONFIG " + cfg);
            tier = new TieredIndex(real.dim, Metric.COSINE, cfg);
            List<TieredIndex.IdVector> src = new AbstractList<>() {
                @Override
                public TieredIndex.IdVector get(int i) {
                    return new TieredIndex.IdVector(i, data.row(i));
                }

                @Override
                public int size() {
                    return base;
                }
            };
            long t0 = System.nanoTime();
            tier.rebuildFrom(src);
            double seconds = (System.nanoTime() - t0) / 1e9;
            live = new int[rows];
            for (int i = 0; i < base; i++) live[i] = i;
            liveCount = base;
            TieredIndex.Stats s = tier.stats();
            check(s.segments() == 1 && s.tailSize() == 0 && tier.size() == base, 0, "after the first build: segments " + s.segments() + ", tail " + s.tailSize() + ", size " + tier.size());
            out.line(Bench.f("BASE rows=%d seconds=%.3f segments=%d", base, seconds, s.segments()));
            best = newLists();
            fromScratch(allQueries());
        }

        private void generation(int g) throws Exception {
            final int before = liveCount;
            final int deletes = (int) Math.round(before * deletePct / 100);
            SplittableRandom r = new SplittableRandom(Bench.mix(seed) + g);
            for (int i = 0; i < deletes; i++) swap(live, i, i + r.nextInt(liveCount - i));
            for (int i = 0; i < deletes; i++) {
                check(tier.delete(live[i]), g, "the delete of the live id " + live[i] + " returned false");
                gone.set(live[i]);
            }
            System.arraycopy(live, deletes, live, 0, liveCount - deletes);
            liveCount -= deletes;
            final int from = base + (g - 1) * add;
            for (int id = from; id < from + add; id++) {
                tier.add(id, data.row(id));
                live[liveCount++] = id;
            }
            long a = System.nanoTime();
            tier.sealNow();
            double sealSeconds = (System.nanoTime() - a) / 1e9;
            TieredIndex.Stats s0 = tier.stats();
            int[] sizes = s0.segmentSizes();
            check(s0.tailSize() == 0 && sizes.length == 2 && sizes[0] == before && sizes[1] == add, g,
                    "before the merge: tail " + s0.tailSize() + ", segments " + Arrays.toString(sizes) + ", expected [" + before + ", " + add + "]");

            long gcBefore = uptimeMs();
            System.gc();
            Bench.Gc c0 = Bench.Gc.take();
            long start = uptimeMs();
            long t0 = System.nanoTime();
            tier.mergeNow();
            long t1 = System.nanoTime();
            long end = uptimeMs();
            Bench.Gc c1 = Bench.Gc.take();
            System.gc();
            long gcAfter = uptimeMs();

            TieredIndex.Stats s1 = tier.stats();
            long built = s1.mergedVectors() - s0.mergedVectors();
            long expected = mode == TieredIndex.MergeMode.INSERT ? add : liveCount;
            check(s1.merges() - s0.merges() == 1 && s1.bigMerges() - s0.bigMerges() == 1, g, "mergeNow ran " + (s1.merges() - s0.merges()) + " merges");
            check(s1.segments() == 1 && s1.segmentSizes()[0] == liveCount, g, "after the merge: segments " + Arrays.toString(s1.segmentSizes()) + ", live " + liveCount);
            check(built == expected, g, "the merge built " + built + " rows, " + mode + " builds " + expected);
            check(tier.size() == liveCount, g, "size " + tier.size() + ", expected " + liveCount);
            check(s1.failures() == 0 && tier.lastError() == null && tier.health() == TieredIndex.Health.OK, g, "failures " + s1.failures() + ", " + tier.lastError());

            updateTruth(from, from + add);
            double sum = 0, min = 1;
            int bad = 0;
            for (int q = 0; q < nq; q++) {
                Hit[] h = tier.search(qs[q], k, null, ef);
                bad += badAnswers(h, from + add);
                double rec = Bench.recall(h, best[q].top(k));
                sum += rec;
                min = Math.min(min, rec);
            }
            check(bad == 0, g, bad + " bad answers");
            check(sum / nq >= minRecall, g, Bench.f("recall %.5f is below the floor %.5f", sum / nq, minRecall));
            out.line(Bench.f("GEN g=%d deleted=%d added=%d live=%d rowsBefore=%d built=%d sealSeconds=%.3f mergeSeconds=%.3f uptimeGcBeforeMs=%d uptimeStartMs=%d uptimeEndMs=%d"
                            + " uptimeGcAfterMs=%d gcCount=%d gcMillis=%d bytesHeld=%d recall=%.5f recallMin=%.2f", g, deletes, add, liveCount, before + add, built, sealSeconds,
                    (t1 - t0) / 1e9, gcBefore, start, end, gcAfter, c1.count() - c0.count(), c1.millis() - c0.millis(), s1.bytesHeld(), sum / nq, min));
        }

        /** malformed answers, ids that no add has used, and deleted ids */
        private int badAnswers(Hit[] h, int used) {
            int bad = Bench.malformed(h, k) == null ? 0 : 1;
            for (Hit x : h) if (x.id() < 0 || x.id() >= used || gone.get((int) x.id())) bad++;
            return bad;
        }

        private Best[] newLists() {
            Best[] b = new Best[nq];
            for (int q = 0; q < nq; q++) b[q] = new Best(KEEP);
            return b;
        }

        private int[] allQueries() {
            int[] all = new int[nq];
            for (int q = 0; q < nq; q++) all[q] = q;
            return all;
        }

        /** the exact lists of these queries over the live rows, from nothing */
        private void fromScratch(int[] queries) throws Exception {
            for (int q : queries) best[q].n = 0;
            scan(live, liveCount, queries, false);
        }

        /** after the writes of one generation: the deleted ids leave the lists, then the added rows [from, to) enter where they belong */
        private void updateTruth(int from, int to) throws Exception {
            int[] added = new int[to - from];
            for (int i = 0; i < added.length; i++) added[i] = from + i;
            for (Best b : best) b.drop(gone);
            scan(added, added.length, Arrays.stream(allQueries()).filter(q -> best[q].n > 0).toArray(), true);
            int[] again = Arrays.stream(allQueries()).filter(q -> best[q].n < k).toArray();
            rescans += again.length;
            fromScratch(again);
        }

        /** offers the first count rows of ids to the lists of the queries; with aboveLast only a row that is better than the last entry its list had */
        private void scan(int[] ids, int count, int[] queries, boolean aboveLast) throws Exception {
            if (count == 0 || queries.length == 0) return;
            final int dim = real.dim;
            Bench.parallel(threads, queries.length, (t, lo, hi) -> {
                float[][] block = new float[BLOCK][dim];
                float[] lastScore = new float[hi - lo];
                int[] lastId = new int[hi - lo];
                if (aboveLast) {
                    for (int x = lo; x < hi; x++) {
                        Best b = best[queries[x]];
                        lastScore[x - lo] = b.score[b.n - 1];
                        lastId[x - lo] = b.id[b.n - 1];
                    }
                }
                for (int s = 0; s < count; s += BLOCK) {
                    int m = Math.min(BLOCK, count - s);
                    for (int i = 0; i < m; i++) data.get(ids[s + i], block[i]);
                    for (int x = lo; x < hi; x++) {
                        int q = queries[x];
                        for (int i = 0; i < m; i++) {
                            float sc = Bench.dot(qs[q], block[i], dim);
                            if (!aboveLast || Best.better(sc, ids[s + i], lastScore[x - lo], lastId[x - lo])) best[q].offer(sc, ids[s + i]);
                        }
                    }
                }
            });
        }

        /** the lists that were kept up to date against a scan of all live rows */
        private void truthCheck() throws Exception {
            Best[] kept = best;
            best = newLists();
            fromScratch(allQueries());
            int wrong = 0;
            for (int q = 0; q < nq; q++) if (!Arrays.equals(kept[q].top(k), best[q].top(k))) wrong++;
            out.line(Bench.f("TRUTH-CHECK mismatchedQueries=%d rescans=%d queries=%d", wrong, rescans, nq));
            if (wrong > 0) fail("the exact answers that were kept up to date differ from a full scan for " + wrong + " queries");
        }

        private void search() {
            System.gc();
            long[] lat = new long[reps * nq];
            double sum = 0, total = 0;
            int n = 0, bad = 0;
            for (int pass = 0; pass <= reps; pass++) {
                for (int q = 0; q < nq; q++) {
                    long a = System.nanoTime();
                    Hit[] h = tier.search(qs[q], k, null, ef);
                    long b = System.nanoTime();
                    if (pass == 0) {
                        sum += Bench.recall(h, best[q].top(k));
                        bad += badAnswers(h, rows);
                    } else {
                        lat[n++] = b - a;
                        total += b - a;
                    }
                }
            }
            Arrays.sort(lat);
            if (bad > 0) fail("final search: " + bad + " bad answers");
            out.line(Bench.f("SEARCH-FINAL k=%d ef=%d live=%d segments=%d recall=%.5f p50us=%.1f p99us=%.1f meanUs=%.1f samples=%d", k, ef, liveCount, tier.segmentCount(), sum / nq,
                    Bench.pct(lat, .5) / 1e3, Bench.pct(lat, .99) / 1e3, total / n / 1e3, n));
        }
    }
}
