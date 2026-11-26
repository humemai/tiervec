package ai.humem.tiervec;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.LockSupport;
import java.util.function.BooleanSupplier;
import java.util.function.LongConsumer;
import java.util.function.LongSupplier;
import java.util.zip.CRC32;
import java.util.AbstractList;

/**
 * A paced stream of writes into one tier while paced searches run, to choose a merge policy (issue #4; docs/BENCHMARKS.md, "Paced writes").
 * Maven does not compile it: it is compiled with the library sources, like {@link Bench}.
 *
 * <p>The run draws rows of a prepared real data set ({@link RealData}, cosine or l2) in an order that depends on {@code --seed} only and builds the
 * first {@code --base} of them with one {@code rebuildFrom}. It searches the idle index at a fixed rate for {@code --idle-seconds}, then one writer
 * adds the next {@code --inserts} rows, one at {@code --rate} per second (a late add goes at once), while one searcher goes on at
 * {@code --search-rate} per second. When the writer is done it waits until no seal or merge is due, searches the idle index again, and measures the
 * layout that the policy left: recall and latency at {@code --ef} and at matched recall ({@code --targets}), against exact answers that it computes
 * with its own scalar code. With {@code --sealed true} there is no stream: all {@code base + inserts} rows go into one {@code rebuildFrom}, the
 * single sealed graph that a policy is compared with. With {@code --split-beam both} the paced searches alternate between the full beam and the split
 * beam ({@code TieredIndex.reconfigure}), and both are measured at the end.
 *
 * <p>The tail is never sealed by age, so the layout does not change while it is measured. The write amplification of the stream is the change of the
 * rows that merges built divided by the change of the adds, between the end of the first and of the last merge that rewrote the largest segment
 * while the writer ran, when there are at least {@value #WINDOW_MERGES} such merges (so at least three lie between them). The busy share is the share
 * of the stream during which a seal or a merge ran, from the task log of the tier.
 *
 * <p>Options are {@code --key value} pairs, see {@link #defaults()}. The exit code is 1 when a check fails (an exception, a wrong live count, a
 * failure of the worker, a malformed answer, an idle timeout, a recall below {@code --min-recall}), and 2 for a wrong option.
 */
public final class PacedWrites {
    private PacedWrites() {}

    static final int WINDOW_MERGES = 5;

    static Map<String, String> defaults() {
        Map<String, String> d = new LinkedHashMap<>();
        d.put("data", "");                  // required: a prepared real data set, see RealData
        d.put("base", "100000");            // rows of the first segment
        d.put("inserts", "100000");         // rows of the paced stream
        d.put("rate", "100");               // adds per second of the writer
        d.put("search-rate", "100");        // searches per second of the paced searcher
        d.put("idle-seconds", "60");        // paced searches of the idle index before and after the stream
        d.put("drain-timeout-s", "1800");   // the wait after the stream until no seal or merge is due
        d.put("seed", "42");                // the order of the rows
        d.put("queries", "1000");
        d.put("k", "10");
        d.put("ef", "100");
        d.put("targets", "0.95,0.99");      // recall targets of the matched latency at the end
        d.put("ef-grid", "10,12,14,16,18,20,23,26,30,35,40,46,53,60,70,80,92,106,122,140,160,185,212,245,280,320,370,425,490,560,640,800,1000");
        d.put("reps", "5");                 // timed passes over the queries per beam width at the end, after one warm-up pass
        d.put("efc", "200");
        d.put("m", "16");
        d.put("tail-limit", "2000");
        d.put("small-limit", "1000");
        d.put("max-segments", "8");
        d.put("merge-fan", "4");
        d.put("merge-trigger-ratio", "0");  // 0 is off
        d.put("split-beam", "both");        // false, true, or both
        d.put("build-threads", "0");        // threads of one seal or merge build; 0 is the default of TieredIndex.Config
        d.put("sealed", "false");           // true: no stream, one sealed graph over base + inserts rows
        d.put("threads", "0");              // threads of the exact scan; 0 is availableProcessors
        d.put("min-recall", "0");           // floor of recall at --ef at the end; 0 is no floor
        d.put("host-label", "");
        d.put("repo", ".");
        d.put("out", "");
        return d;
    }

    public static void main(String[] args) throws Exception {
        Run run;
        try {
            Map<String, String> o = defaults();
            for (int i = 0; i < args.length; i += 2) {
                if (!args[i].startsWith("--") || !o.containsKey(args[i].substring(2)) || i + 1 >= args.length)
                    throw new IllegalArgumentException("bad option " + args[i] + " (options are --key value pairs: " + defaults().keySet() + ")");
                o.put(args[i].substring(2), args[i + 1]);
            }
            run = new Run(o);
        } catch (IllegalArgumentException e) {
            System.err.println(e.getMessage());
            System.exit(2);
            return;
        }
        System.exit(run.go());
    }

    static void parkUntil(long deadline) {
        long left;
        while ((left = deadline - System.nanoTime()) > 0) LockSupport.parkNanos(left);
    }

    /** the score of the metric with the harness's own scalar code, higher is better: the dot product, or minus the squared distance */
    static float score(Metric metric, float[] a, float[] b) {
        if (metric != Metric.L2) return Bench.dot(a, b, a.length);
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int i = 0;
        for (; i + 4 <= a.length; i += 4) {
            float d0 = a[i] - b[i], d1 = a[i + 1] - b[i + 1], d2 = a[i + 2] - b[i + 2], d3 = a[i + 3] - b[i + 3];
            s0 += d0 * d0;
            s1 += d1 * d1;
            s2 += d2 * d2;
            s3 += d3 * d3;
        }
        for (; i < a.length; i++) {
            float d = a[i] - b[i];
            s0 += d * d;
        }
        return -((s0 + s1) + (s2 + s3));
    }

    /** the paced searches of one phase: per search its start, its end and whether the split beam was on */
    static final class Samples {
        final Bench.LongList start = new Bench.LongList(), end = new Bench.LongList(), split = new Bench.LongList();
        int bad;
    }

    private static final class Run {
        final Map<String, String> o;
        final RealData real;
        final Metric metric;
        final int base, inserts, rows, nq, k, ef, reps, threads;
        final double rate, searchRate, idleSeconds, minRecall;
        final boolean sealed;
        final boolean[] splits;
        final List<String> failures = new ArrayList<>();
        final AtomicLong added = new AtomicLong(), mergedRows = new AtomicLong();
        final ConcurrentLinkedQueue<long[]> bigMerges = new ConcurrentLinkedQueue<>(); // {System.nanoTime, adds, merged rows}
        Bench.Out out;
        MergeGenerations.Rows data;
        float[][] qs;
        float[] kth; // per query: the score of the k-th exact answer
        TieredIndex tier;

        Run(Map<String, String> o) throws Exception {
            this.o = o;
            if (o.get("data").isEmpty()) throw new IllegalArgumentException("--data is required: the directory of a prepared real data set (docs/BENCHMARKS.md, \"Real data\")");
            real = RealData.open(Path.of(o.get("data")));
            metric = real.metric();
            base = num("base");
            inserts = num("inserts");
            nq = num("queries");
            k = num("k");
            ef = num("ef");
            reps = num("reps");
            threads = num("threads") > 0 ? num("threads") : Runtime.getRuntime().availableProcessors();
            rate = Double.parseDouble(o.get("rate"));
            searchRate = Double.parseDouble(o.get("search-rate"));
            idleSeconds = Double.parseDouble(o.get("idle-seconds"));
            minRecall = Double.parseDouble(o.get("min-recall"));
            String s = o.get("sealed");
            if (!s.equals("true") && !s.equals("false")) throw new IllegalArgumentException("--sealed is true or false, not " + s);
            sealed = s.equals("true");
            String sb = o.get("split-beam");
            if (sb.equals("both")) splits = new boolean[] {false, true};
            else if (sb.equals("true") || sb.equals("false")) splits = new boolean[] {sb.equals("true")};
            else throw new IllegalArgumentException("--split-beam is false, true or both, not " + sb);
            if (base < num("small-limit") || inserts < 0 || !(rate > 0) || !(searchRate > 0) || !(idleSeconds >= 0) || reps < 1 || k < 1 || ef < k)
                throw new IllegalArgumentException("--base is at least --small-limit, --inserts and --idle-seconds at least 0, --rate and --search-rate above 0, --reps and --k at least 1, and --ef at least --k");
            long need = (long) base + inserts;
            if (need > real.count) throw new IllegalArgumentException("the run needs " + need + " rows, the data set holds " + real.count);
            if (need * real.dim * 4 > Integer.MAX_VALUE) throw new IllegalArgumentException("the rows of the run do not fit one buffer of 2 GiB");
            if (nq < 1 || nq > real.queries) throw new IllegalArgumentException("--queries is 1 to " + real.queries + " (what the data set holds), not " + nq);
            rows = (int) need;
        }

        private int num(String key) {
            return Integer.parseInt(o.get(key));
        }

        private void fail(String why) {
            failures.add(why);
            out.line("FAIL " + why);
        }

        private void check(boolean ok, String why) {
            if (!ok) fail(why);
        }

        int go() throws Exception {
            out = new Bench.Out(o.get("out"));
            long t0 = System.nanoTime();
            try {
                out.line("PW-RESULT format=1 harness=ai.humem.tiervec.PacedWrites");
                MergeGenerations.machine(out, o, real);
                load();
                build();
                if (!sealed) {
                    idle("idle-before");
                    stream();
                }
                idle(sealed ? "idle" : "idle-after");
                finalState();
                System.gc();
                Runtime rt = Runtime.getRuntime();
                out.line(Bench.f("MEMORY end rssPeakMB=%.0f heapUsedMB=%.0f bytesHeld=%d", Bench.statusKb("VmHWM") / 1024.0, (rt.totalMemory() - rt.freeMemory()) / 1e6,
                        tier.bytesHeld()));
            } catch (Throwable e) {
                fail("exception: " + e);
                e.printStackTrace();
            } finally {
                if (tier != null) tier.close();
            }
            for (String why : failures) out.line("PW-FAILURE " + why);
            out.line(Bench.f("PW-SUMMARY status=%s failures=%d wallSeconds=%.1f", failures.isEmpty() ? "OK" : "FAIL", failures.size(), (System.nanoTime() - t0) / 1e9));
            out.line("PW-END");
            out.close();
            return failures.isEmpty() ? 0 : 1;
        }

        private void load() throws Exception {
            long t0 = System.nanoTime();
            int[] order = MergeGenerations.drawOrder(real.count, rows, Long.parseLong(o.get("seed")));
            data = MergeGenerations.Rows.read(real.dir.resolve(RealData.VECTORS), order, real.dim, metric == Metric.COSINE);
            qs = real.readQueries(nq);
            CRC32 orderCrc = new CRC32();
            for (int row : order) MergeGenerations.crc(orderCrc, row);
            CRC32 firstCrc = new CRC32();
            for (int s = 0; s < Math.min(1000, rows); s++) for (float x : data.row(s)) MergeGenerations.crc(firstCrc, Float.floatToIntBits(x));
            out.line(Bench.f("DATA rows=%d base=%d inserts=%d queries=%d dim=%d metric=%s orderCrc=%08x first1000Crc=%08x loadSeconds=%.1f", rows, base, inserts, nq, real.dim,
                    metric, orderCrc.getValue(), firstCrc.getValue(), (System.nanoTime() - t0) / 1e9));
        }

        private void build() {
            GraphIndex.Params gp0 = GraphIndex.Params.defaults().withM(num("m")).withEfConstruction(num("efc"));
            GraphIndex.Params gp = new GraphIndex.Params(gp0.M(), gp0.efConstruction(), ef, gp0.alpha(), false, gp0.seed());
            TieredIndex.Config d = TieredIndex.Config.defaults();
            int buildThreads = num("build-threads") > 0 ? num("build-threads") : d.buildThreads();
            TieredIndex.Config cfg = new TieredIndex.Config(num("tail-limit"), 0L, num("small-limit"), num("max-segments"), num("merge-fan"), d.deadFraction(),
                    d.mergeRatePerSec(), d.hardTailFactor(), Quant.NONE, gp, buildThreads, splits[0], Double.parseDouble(o.get("merge-trigger-ratio")), TieredIndex.MergeMode.REBUILD);
            out.line("CONFIG " + cfg);
            tier = new TieredIndex(real.dim, metric, cfg);
            final int first = sealed ? rows : base;
            List<TieredIndex.IdVector> src = new AbstractList<>() {
                @Override
                public TieredIndex.IdVector get(int i) {
                    return new TieredIndex.IdVector(i, data.row(i));
                }

                @Override
                public int size() {
                    return first;
                }
            };
            long t0 = System.nanoTime();
            tier.rebuildFrom(src);
            double seconds = (System.nanoTime() - t0) / 1e9;
            TieredIndex.Stats s = tier.stats();
            check(s.segments() == 1 && s.tailSize() == 0 && tier.size() == first, "after the first build: segments " + s.segments() + ", tail " + s.tailSize() + ", size " + tier.size());
            out.line(Bench.f("BASE rows=%d seconds=%.3f segments=%d buildThreads=%d", first, seconds, s.segments(), buildThreads));
        }

        private void split(boolean on) {
            if (tier.config().splitBeam() != on) tier.reconfigure(Map.of("splitBeam", String.valueOf(on)));
        }

        /** paced searches from now on, while running: search j starts at t0 + j / searchRate (at once when late) and takes the next split beam setting */
        private Samples paced(BooleanSupplier running) {
            Samples s = new Samples();
            long t0 = System.nanoTime();
            for (long j = 0; running.getAsBoolean(); j++) {
                parkUntil(t0 + (long) (j * 1e9 / searchRate));
                boolean on = splits[(int) (j % splits.length)];
                split(on);
                float[] q = qs[(int) ((j / splits.length) % nq)];
                long a = System.nanoTime();
                Hit[] h = tier.search(q, k, null, ef);
                long b = System.nanoTime();
                s.start.add(a);
                s.end.add(b);
                s.split.add(on ? 1 : 0);
                if (Bench.malformed(h, k) != null) s.bad++;
                for (Hit x : h) if (x.id() < 0 || x.id() >= rows) s.bad++;
            }
            return s;
        }

        private void idle(String phase) {
            long until = System.nanoTime() + (long) (idleSeconds * 1e9);
            Samples s = paced(() -> System.nanoTime() < until);
            check(s.bad == 0, phase + ": " + s.bad + " bad answers");
            int[] cls = new int[s.start.n];
            lat(phase, s, cls, 0);
        }

        /** one LAT line per split beam setting, over the samples of the class */
        private void lat(String phase, Samples s, int[] cls, int want) {
            for (boolean on : splits) {
                Bench.LongList v = new Bench.LongList();
                for (int i = 0; i < s.start.n; i++) if (cls[i] == want && (s.split.a[i] == 1) == on) v.add(s.end.a[i] - s.start.a[i]);
                long[] x = Arrays.copyOf(v.a, v.n);
                Arrays.sort(x);
                out.line(Bench.f("LAT phase=%s split=%b samples=%d p50us=%.1f p99us=%.1f p999us=%.1f maxUs=%.1f", phase, on, x.length, Bench.pct(x, .5) / 1e3,
                        Bench.pct(x, .99) / 1e3, Bench.pct(x, .999) / 1e3, x.length == 0 ? 0 : x[x.length - 1] / 1e3));
            }
        }

        private void stream() throws Exception {
            tier.bindMetrics(new MetricSink() {
                @Override
                public LongConsumer counter(String name) {
                    // called on the thread of the event, the worker for a merge; the counters of a merge arrive in the order merges, mergedVectors, bigMerges
                    if (name.equals("tiervec.mergedVectors")) return n -> mergedRows.addAndGet(n);
                    if (name.equals("tiervec.bigMerges")) return n -> bigMerges.add(new long[] {System.nanoTime(), added.get(), mergedRows.get()});
                    return n -> { };
                }

                @Override
                public void gauge(String name, LongSupplier value) {
                }
            });
            final TieredIndex.Stats s0 = tier.stats();
            final Throwable[] error = new Throwable[1];
            final long[] lag = new long[1];
            final long[] writerEnd = new long[1];
            final long t0 = System.nanoTime();
            Thread writer = new Thread(() -> {
                try {
                    for (int i = 0; i < inserts; i++) {
                        long due = t0 + (long) (i * 1e9 / rate);
                        parkUntil(due);
                        lag[0] = Math.max(lag[0], System.nanoTime() - due);
                        tier.add(base + i, data.row(base + i));
                        added.incrementAndGet();
                    }
                } catch (Throwable e) {
                    error[0] = e;
                } finally {
                    writerEnd[0] = System.nanoTime();
                }
            }, "paced-writer");
            writer.start();
            Samples s = paced(writer::isAlive);
            writer.join();
            if (error[0] != null) throw new IllegalStateException("the writer failed: " + error[0], error[0]);
            final long t1 = writerEnd[0];
            final TieredIndex.Stats s1 = tier.stats();
            boolean idle = tier.awaitIdle(Long.parseLong(o.get("drain-timeout-s")) * 1000L);
            final double drain = (System.nanoTime() - t1) / 1e9;
            final TieredIndex.Stats s2 = tier.stats();
            check(idle, "the tier was not idle within the drain timeout");
            check(s.bad == 0, "stream: " + s.bad + " bad answers");
            check(s2.taskDropped() == 0, "the task log dropped " + s2.taskDropped() + " tasks");

            // the share of the stream during which a seal or a merge ran
            int[] kinds = s2.taskKinds();
            long[] from = s2.taskStartNs(), to = s2.taskEndNs();
            long sealNs = 0, mergeNs = 0;
            for (int i = 0; i < kinds.length; i++) {
                long overlap = Math.max(0, Math.min(to[i], t1) - Math.max(from[i], t0));
                if (kinds[i] == 1) sealNs += overlap;
                else mergeNs += overlap;
            }
            double seconds = (t1 - t0) / 1e9;
            out.line(Bench.f("STREAM inserts=%d rate=%.1f seconds=%.1f achievedRate=%.1f maxLagMs=%.0f backpressureWaits=%d seals=%d merges=%d bigMerges=%d busyShare=%.4f"
                            + " sealSeconds=%.1f mergeSeconds=%.1f sealedVectors=%d mergedVectors=%d searches=%d", inserts, rate, seconds, inserts / seconds, lag[0] / 1e6,
                    s1.backpressureWaits() - s0.backpressureWaits(), s1.seals() - s0.seals(), s1.merges() - s0.merges(), s1.bigMerges() - s0.bigMerges(),
                    (sealNs + mergeNs) / 1e9 / seconds, sealNs / 1e9, mergeNs / 1e9, s1.sealedVectors() - s0.sealedVectors(), s1.mergedVectors() - s0.mergedVectors(), s.start.n));

            // the write amplification between the ends of the first and the last merge into the largest segment while the writer ran
            List<long[]> big = new ArrayList<>();
            for (long[] e : bigMerges) if (e[0] <= t1) big.add(e);
            for (int i = 0; i < big.size(); i++)
                out.line(Bench.f("BIG i=%d atSeconds=%.1f adds=%d mergedVectors=%d", i + 1, (big.get(i)[0] - t0) / 1e9, big.get(i)[1], big.get(i)[2]));
            String amp = "NA";
            long a = 0, b = 0;
            if (big.size() >= WINDOW_MERGES) {
                long[] first = big.get(0), last = big.get(big.size() - 1);
                a = first[1];
                b = last[1];
                if (b > a) amp = Bench.f("%.4f", (double) (last[2] - first[2]) / (b - a));
            }
            out.line(Bench.f("WINDOW bigMerges=%d writeAmp=%s fromAdds=%d toAdds=%d", big.size(), amp, a, b));
            out.line(Bench.f("WHOLE adds=%d mergedVectorsAfterDrain=%d mergeWriteAmp=%.4f", inserts, s2.mergedVectors() - s0.mergedVectors(),
                    inserts == 0 ? 0.0 : (double) (s2.mergedVectors() - s0.mergedVectors()) / inserts));

            // the latency of the paced searches by what ran while they did
            int[] cls = new int[s.start.n];
            int p = 0;
            for (int i = 0; i < s.start.n; i++) {
                long x = s.start.a[i], y = s.end.a[i];
                while (p < kinds.length && to[p] <= x) p++;
                for (int t = p; t < kinds.length && from[t] < y; t++) cls[i] = Math.max(cls[i], kinds[t]); // 2 (a merge) wins over 1 (a seal)
            }
            lat("stream-quiet", s, cls, 0);
            lat("stream-seal", s, cls, 1);
            lat("stream-merge", s, cls, 2);
            out.line(Bench.f("DRAIN seconds=%.1f idle=%b merges=%d", drain, idle, s2.merges() - s1.merges()));
        }

        private void finalState() throws Exception {
            TieredIndex.Stats s = tier.stats();
            check(tier.size() == rows, "size " + tier.size() + ", expected " + rows);
            check(s.failures() == 0 && tier.lastError() == null && tier.health() == TieredIndex.Health.OK, "failures " + s.failures() + ", " + tier.lastError());
            out.line(Bench.f("FINAL live=%d segments=%d segmentSizes=%s tail=%d bytesHeld=%d", tier.size(), s.segments(), Arrays.toString(s.segmentSizes()).replace(" ", ""),
                    s.tailSize(), s.bytesHeld()));
            exact();
            System.gc();
            for (boolean on : splits) {
                split(on);
                double rec = recallAt(ef, on);
                double[] t = timed(ef);
                check(rec >= minRecall, Bench.f("split %b: recall %.4f is below the floor %.4f", on, rec, minRecall));
                out.line(Bench.f("SEARCH split=%b k=%d ef=%d recall=%.5f p50us=%.3f p99us=%.3f meanUs=%.3f samples=%d", on, k, ef, rec, t[0], t[1], t[2], reps * nq));
                matched(on);
            }
        }

        /** the score of the k-th best exact answer of every query, over all rows of the index */
        private void exact() throws Exception {
            kth = new float[nq];
            long t0 = System.nanoTime();
            Bench.parallel(threads, nq, (t, lo, hi) -> {
                MergeGenerations.Best[] best = new MergeGenerations.Best[hi - lo];
                for (int q = lo; q < hi; q++) best[q - lo] = new MergeGenerations.Best(k);
                float[] x = new float[real.dim];
                for (int row = 0; row < rows; row++) {
                    data.get(row, x);
                    for (int q = lo; q < hi; q++) best[q - lo].offer(score(metric, qs[q], x), row);
                }
                for (int q = lo; q < hi; q++) kth[q] = best[q - lo].score[best[q - lo].n - 1];
            });
            out.line(Bench.f("EXACT queries=%d rows=%d seconds=%.1f threads=%d", nq, rows, (System.nanoTime() - t0) / 1e9, threads));
        }

        /** one pass over the queries: the mean share of the k answers whose exact score is at least the k-th exact score (a tie counts) */
        private double recallAt(int efv, boolean on) {
            double sum = 0;
            int bad = 0;
            float[] x = new float[real.dim];
            for (int q = 0; q < nq; q++) {
                Hit[] h = tier.search(qs[q], k, null, efv);
                if (Bench.malformed(h, k) != null) bad++;
                int good = 0;
                for (Hit hit : h) {
                    if (hit.id() < 0 || hit.id() >= rows) {
                        bad++;
                        continue;
                    }
                    data.get((int) hit.id(), x);
                    if (score(metric, qs[q], x) >= kth[q]) good++;
                }
                sum += (double) Math.min(good, k) / k;
            }
            check(bad == 0, "split " + on + " ef " + efv + ": " + bad + " bad answers");
            return sum / nq;
        }

        /** one warm-up pass and the timed passes at one beam width: p50, p99 and the mean in microseconds */
        private double[] timed(int efv) {
            for (int q = 0; q < nq; q++) tier.search(qs[q], k, null, efv);
            long[] lat = new long[reps * nq];
            long sum = 0;
            for (int i = 0; i < lat.length; i++) {
                float[] q = qs[i % nq];
                long a = System.nanoTime();
                tier.search(q, k, null, efv);
                lat[i] = System.nanoTime() - a;
                sum += lat[i];
            }
            Arrays.sort(lat);
            return new double[] {Bench.pct(lat, .5) / 1000.0, Bench.pct(lat, .99) / 1000.0, sum / 1000.0 / lat.length};
        }

        /**
         * Latency at matched recall, as the MATCHED block of {@link Bench}: the beam widths from the smallest until the highest target is reached, then
         * for each target the first width that reaches it and the one before are timed and the latency is interpolated linearly in recall.
         */
        private void matched(boolean on) {
            double[] targets = Arrays.stream(o.get("targets").split(",")).filter(v -> !v.isBlank()).mapToDouble(v -> Double.parseDouble(v.trim())).toArray();
            if (targets.length == 0) return;
            int[] grid = Arrays.stream(o.get("ef-grid").split(",")).mapToInt(v -> Integer.parseInt(v.trim())).filter(v -> v >= k).sorted().distinct().toArray();
            double top = Arrays.stream(targets).max().getAsDouble();
            double[] rec = new double[grid.length];
            int swept = 0;
            while (swept < grid.length) {
                rec[swept] = recallAt(grid[swept], on);
                out.line(Bench.f("MATCH-SWEEP split=%b ef=%d recall=%.5f", on, grid[swept], rec[swept]));
                swept++;
                if (rec[swept - 1] >= top) break;
            }
            int[] reach = new int[targets.length];
            TreeSet<Integer> need = new TreeSet<>();
            for (int t = 0; t < targets.length; t++) {
                reach[t] = -1;
                for (int i = 0; i < swept && reach[t] < 0; i++) if (rec[i] >= targets[t]) reach[t] = i;
                if (reach[t] >= 0) need.add(reach[t]);
                if (reach[t] > 0) need.add(reach[t] - 1);
            }
            Map<Integer, double[]> at = new TreeMap<>();
            for (int i : need) {
                double[] r = timed(grid[i]);
                at.put(i, r);
                out.line(Bench.f("MATCH-TIMED split=%b ef=%d recall=%.5f p50us=%.3f p99us=%.3f meanUs=%.3f samples=%d", on, grid[i], rec[i], r[0], r[1], r[2], reps * nq));
            }
            for (int t = 0; t < targets.length; t++) {
                int hi = reach[t];
                if (hi < 0) {
                    double best = 0;
                    for (int i = 0; i < swept; i++) best = Math.max(best, rec[i]);
                    out.line(Bench.f("MATCHED split=%b target=%s status=unreached maxEf=%d maxRecall=%.5f", on, targets[t], grid[swept - 1], best));
                    continue;
                }
                int lo = Math.max(0, hi - 1);
                double w = hi == lo ? 1 : (targets[t] - rec[lo]) / (rec[hi] - rec[lo]);
                double[] a = at.get(lo), b = at.get(hi);
                double p50 = a[0] + w * (b[0] - a[0]), p99 = a[1] + w * (b[1] - a[1]), mean = a[2] + w * (b[2] - a[2]);
                out.line(Bench.f("MATCHED split=%b target=%s status=%s efLo=%d efHi=%d recallLo=%.5f recallHi=%.5f p50us=%.3f p99us=%.3f meanUs=%.3f", on, targets[t],
                        hi == lo ? "floor" : "ok", grid[lo], grid[hi], rec[lo], rec[hi], p50, p99, mean));
            }
        }
    }
}
