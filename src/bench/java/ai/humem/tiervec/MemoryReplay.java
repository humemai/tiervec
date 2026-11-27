package ai.humem.tiervec;

import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;

/**
 * Replays a runbook into one arm and counts (docs/AGENT-MEMORY-BENCHMARK.md, issue #159). Nothing here is timed.
 *
 * <p>Every search of the runbook and of a checkpoint is checked against the live rows. After every {@code probeEvery}-th write, and after
 * every batch, a probe searches with a filter that allows only the written ids. A batch is also watched while it is applied: on an arm with
 * atomic batches a second thread searches during the call, and on the others the replay searches between the writes.
 */
final class MemoryReplay {
    static final int K = 10;

    private MemoryReplay() {}

    /** the neighbours that the estimate of the intrinsic dimension takes, or all rows but one when a collection is smaller */
    static final int DIMENSION_NEIGHBOURS = 100;

    /**
     * The exact answers of every query of every checkpoint of a runbook, computed once and used for every arm, and how hard those queries are.
     * Both come from the rows alone, so they say nothing about any arm.
     */
    static final class Answers {
        /** by collection, checkpoint, and query */
        private final List<List<LiveRows.Top[]>> tops = new ArrayList<>();
        final int checkpoints;
        /** the queries of all checkpoints, and the mean over them of {@link #intrinsicDimension()} and of {@link #contrast()} */
        long queries;
        private double dimensionSum, contrastSum;

        /** the mean estimate of the local intrinsic dimension at the queries ({@link LiveRows#intrinsicDimension}) */
        double intrinsicDimension() { return dimensionSum / queries; }

        /**
         * The mean relative contrast of the queries: the mean distance to the live rows over the distance to the 10th nearest. Close to 1: the
         * neighbours of a query are hardly closer than any other row.
         */
        double contrast() { return contrastSum / queries; }

        private void measure(LiveRows live, float[] q) {
            double[] d = live.distances(q);
            double mean = 0;
            for (double x : d) mean += x;
            mean /= d.length;
            queries++;
            dimensionSum += LiveRows.intrinsicDimension(d, Math.min(DIMENSION_NEIGHBOURS, d.length - 1));
            contrastSum += mean / d[Math.min(K, d.length) - 1];
        }

        Answers(Runbook runbook) {
            for (List<Runbook.Op> ops : runbook.collections) {
                LiveRows live = new LiveRows();
                List<LiveRows.Top[]> mine = new ArrayList<>();
                for (Runbook.Op op : ops) {
                    if (op.kind() == Runbook.Kind.CHECKPOINT) {
                        LiveRows.Top[] t = new LiveRows.Top[op.queries().length];
                        for (int i = 0; i < t.length; i++) {
                            t[i] = live.top(op.queries()[i], K);
                            measure(live, op.queries()[i]);
                        }
                        mine.add(t);
                    } else if (op.kind() == Runbook.Kind.BATCH) {
                        op.writes().forEach(live::apply);
                    } else if (op.kind() != Runbook.Kind.SEARCH) {
                        live.apply(op);
                    }
                }
                tops.add(mine);
            }
            checkpoints = tops.get(0).size();
        }
    }

    /** what one arm did with one runbook, summed over the collections */
    static final class Result {
        final Violations violations = new Violations();
        final List<String> failures = new ArrayList<>();
        boolean atomicBatches;
        /** rows added or replaced (creates, updates, and reuses), deletes, batches, searches of the runbook, and probes */
        long added, deleted, batches, searches, probes;
        /** by checkpoint: the sum of recall over the queries, and their number */
        double[] recallSum;
        long[] recallCount;
        long rowsBuilt, rowsRebuilt, bytesHeld, liveRows;
        /**
         * Distance computations, see {@link DistanceCount} (0 in a build that does not count): in the searches of the checkpoints, the
         * quantized ones among them, and in the writes with their maintenance. Probes are in neither.
         */
        long searchDistances, searchQuantized, writeDistances;
        /** the searches of the checkpoints */
        long checkpointSearches;
        /** the heap that closing and dropping the arms gave back after a full collection; small collections put it below the noise */
        long heapBytes;

        double recall(int checkpoint) { return recallCount[checkpoint] == 0 ? 0 : recallSum[checkpoint] / recallCount[checkpoint]; }

        double finalRecall() { return recall(recallSum.length - 1); }

        double lowestRecall() {
            double min = 1;
            for (int i = 0; i < recallSum.length; i++) min = Math.min(min, recall(i));
            return min;
        }

        /** the mean recall of the checkpoints after the one that follows the load, so under churn; the first one's when there is no other */
        double churnMeanRecall() {
            if (recallSum.length == 1) return recall(0);
            double sum = 0;
            for (int i = 1; i < recallSum.length; i++) sum += recall(i);
            return sum / (recallSum.length - 1);
        }
    }

    /**
     * Replays every collection of the runbook into an arm of its own from {@code open}. The arms stay open until the whole runbook is done,
     * as the collections of a host do, and are closed before the method returns.
     */
    static Result run(Runbook runbook, Answers answers, Supplier<MemoryArm> open, int probeEvery) {
        Result r = new Result();
        r.recallSum = new double[answers.checkpoints];
        r.recallCount = new long[answers.checkpoints];
        List<MemoryArm> arms = new ArrayList<>();
        try {
            replayAll(runbook, answers, open, probeEvery, r, arms);
            long held = usedHeap();
            arms.forEach(MemoryArm::close);
            arms.clear();
            r.heapBytes = held - usedHeap();
        } finally {
            arms.forEach(MemoryArm::close);
        }
        return r;
    }

    /** its own method, so that no local variable keeps an arm reachable when the heap is measured */
    private static void replayAll(Runbook runbook, Answers answers, Supplier<MemoryArm> open, int probeEvery, Result r, List<MemoryArm> arms) {
        for (int c = 0; c < runbook.collections.size(); c++) {
            MemoryArm arm = open.get();
            arms.add(arm);
            r.atomicBatches = arm.atomicBatches();
            new Replayed(arm, r, answers.tops.get(c), probeEvery, "collection " + c).replay(runbook.collections.get(c));
            if (arm.failure() != null) r.failures.add("collection " + c + ": the maintenance failed: " + arm.failure());
            r.rowsBuilt += arm.rowsBuilt();
            r.rowsRebuilt += arm.rowsRebuilt();
            r.bytesHeld += arm.bytesHeld();
        }
    }

    private static long usedHeap() {
        for (int i = 0; i < 3; i++) System.gc();
        return ManagementFactory.getMemoryMXBean().getHeapMemoryUsage().getUsed();
    }

    /** the replay of one collection */
    private static final class Replayed {
        private final MemoryArm arm;
        private final Result r;
        private final List<LiveRows.Top[]> tops;
        private final int probeEvery;
        private final String name;
        private final LiveRows live = new LiveRows();
        private long singleWrites;
        private int checkpoint;
        private String where;

        Replayed(MemoryArm arm, Result r, List<LiveRows.Top[]> tops, int probeEvery, String name) {
            this.arm = arm;
            this.r = r;
            this.tops = tops;
            this.probeEvery = probeEvery;
            this.name = name;
        }

        void replay(List<Runbook.Op> ops) {
            for (int i = 0; i < ops.size(); i++) {
                Runbook.Op op = ops.get(i);
                where = name + " operation " + i + " (" + op.kind() + ")";
                switch (op.kind()) {
                    case SEARCH -> {
                        r.violations.searchAnswer(live, op.vec(), arm.search(op.vec(), K, null), K, where);
                        r.searches++;
                    }
                    case CHECKPOINT -> checkpoint(op.queries());
                    case BATCH -> batch(op.writes());
                    default -> single(op);
                }
            }
            r.liveRows += live.size();
        }

        private void checkpoint(float[][] queries) {
            maintain();
            for (int i = 0; i < queries.length; i++) {
                float[] q = queries[i];
                long full = DistanceCount.fullSoFar(), quantized = DistanceCount.quantizedSoFar();
                Hit[] hits = arm.search(q, K, null);
                quantized = DistanceCount.quantizedSoFar() - quantized;
                r.searchDistances += DistanceCount.fullSoFar() - full + quantized;
                r.searchQuantized += quantized;
                r.checkpointSearches++;
                r.violations.searchAnswer(live, q, hits, K, where + " query " + i);
                r.recallSum[checkpoint] += live.recall(tops.get(checkpoint)[i], q, hits);
                r.recallCount[checkpoint]++;
            }
            checkpoint++;
        }

        private void write(Runbook.Op w) {
            long before = DistanceCount.soFar();
            if (w.kind() == Runbook.Kind.DELETE) {
                arm.delete(w.id());
                r.deleted++;
            } else {
                arm.put(w.id(), w.vec());
                r.added++;
            }
            r.writeDistances += DistanceCount.soFar() - before;
        }

        private void maintain() {
            long before = DistanceCount.soFar();
            arm.maintain();
            r.writeDistances += DistanceCount.soFar() - before;
        }

        private void single(Runbook.Op w) {
            float[] old = live.vector(w.id());
            write(w);
            maintain();
            live.apply(w);
            // a delete is probed with the vector that the id had, so that an index which kept it returns it
            if (singleWrites++ % probeEvery == 0) probe(w.vec() != null ? w.vec() : old, new long[] {w.id()});
        }

        private void probe(float[] q, long[] ids) {
            r.violations.probeAnswer(live, q, ids, arm.search(q, ids.length, Violations.only(ids)), where + " probe");
            r.probes++;
        }

        private void batch(List<Runbook.Op> writes) {
            int n = writes.size();
            long[] ids = new long[n];
            float[][] before = new float[n][], after = new float[n][];
            for (int i = 0; i < n; i++) {
                Runbook.Op w = writes.get(i);
                ids[i] = w.id();
                before[i] = live.vector(w.id());
                after[i] = w.vec();
            }
            float[] q = after[0] != null ? after[0] : before[0];
            boolean part = false;
            if (arm.atomicBatches()) {
                part = watched(q, ids, before, after, writes);
                for (Runbook.Op w : writes) {
                    if (w.kind() == Runbook.Kind.DELETE) r.deleted++;
                    else r.added++;
                }
            } else {
                SizedIdFilter only = Violations.only(ids);
                for (int i = 0; i < n; i++) {
                    write(writes.get(i));
                    if (i < n - 1) part |= Violations.showsPart(q, ids, before, after, arm.search(q, n, only));
                }
            }
            maintain();
            writes.forEach(live::apply);
            r.batches++;
            if (part) r.violations.add(Violations.Kind.PART_OF_BATCH, where + " of " + n + " writes was seen in part");
            probe(q, ids);
        }

        /** applies a batch while a second thread searches for its ids, and says whether one of those searches saw it in part */
        private boolean watched(float[] q, long[] ids, float[][] before, float[][] after, List<Runbook.Op> writes) {
            AtomicBoolean done = new AtomicBoolean(), part = new AtomicBoolean();
            AtomicReference<Throwable> error = new AtomicReference<>();
            CountDownLatch started = new CountDownLatch(1);
            SizedIdFilter only = Violations.only(ids);
            Thread watcher = new Thread(() -> {
                try {
                    do {
                        if (Violations.showsPart(q, ids, before, after, arm.search(q, ids.length, only))) part.set(true);
                        started.countDown();
                    } while (!done.get());
                } catch (Throwable t) {
                    error.set(t);
                } finally {
                    started.countDown();
                }
            }, "batch-watcher");
            watcher.start();
            try {
                started.await();
                try {
                    long counted = DistanceCount.soFar(); // of this thread only, so the watching thread is not counted
                    arm.applyBatch(writes);
                    r.writeDistances += DistanceCount.soFar() - counted;
                } finally {
                    done.set(true);
                    watcher.join();
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("interrupted while a batch was applied", e);
            }
            if (error.get() != null) throw new IllegalStateException(where + ": a search beside the batch failed", error.get());
            return part.get();
        }
    }
}
