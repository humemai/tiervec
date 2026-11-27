package ai.humem.tiervec;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The tests of the agent-memory benchmark's own parts (issue #159): the runbook, the exact answers and the violation counters. Maven does not
 * compile src/bench, so the CI job "bench smoke" compiles and runs this program. It prints each check that fails and exits with code 1 when
 * one did.
 */
public final class AgentMemorySelfCheck {
    private static int checks, failures;

    private static void check(boolean ok, String what) {
        checks++;
        if (!ok) {
            failures++;
            System.out.println("SELFCHECK-FAILURE " + what);
        }
    }

    public static void main(String[] args) throws Exception {
        exactAnswersKnown();
        runbookFile();
        runbookWorkload();
        violationCounters();
        System.out.println("SELFCHECK-SUMMARY checks=" + checks + " failures=" + failures);
        System.exit(failures == 0 && checks > 0 ? 0 : 1);
    }

    // ---- the exact answers ----

    /** a unit vector of two dimensions at an angle; the cosine of two of them is the cosine of the angle between them */
    private static float[] at(double degrees) {
        double a = Math.toRadians(degrees);
        return new float[] {(float) Math.cos(a), (float) Math.sin(a)};
    }

    private static Runbook.Op create(long id, double degrees) { return Runbook.Op.write(Runbook.Kind.CREATE, id, at(degrees)); }

    private static Runbook.Op delete(long id) { return Runbook.Op.write(Runbook.Kind.DELETE, id, null); }

    /** the exact answer for the query at 0 degrees: these ids in this order, each with the cosine of its angle */
    private static void expectTop(LiveRows live, int k, String what, long[] ids, double[] degrees) {
        LiveRows.Top top = live.top(at(0), k);
        check(Arrays.equals(top.ids(), ids), what + ": ids " + Arrays.toString(top.ids()) + ", expected " + Arrays.toString(ids));
        for (int i = 0; i < degrees.length && i < top.scores().length; i++) {
            check(Math.abs(top.scores()[i] - Math.cos(Math.toRadians(degrees[i]))) < 1e-6, what + ": score " + i + " is " + top.scores()[i]);
        }
    }

    private static void exactAnswersKnown() {
        LiveRows live = new LiveRows();
        expectTop(live, 3, "no rows", new long[0], new double[0]);
        double[] angles = {40, 10, 90, 20, 180};
        for (int i = 0; i < angles.length; i++) live.apply(create(i + 1, angles[i]));
        expectTop(live, 3, "after the load", new long[] {2, 4, 1}, new double[] {10, 20, 40});
        expectTop(live, 10, "k above the live rows", new long[] {2, 4, 1, 3, 5}, new double[] {10, 20, 40, 90, 180});
        live.apply(delete(2));
        expectTop(live, 3, "after a delete", new long[] {4, 1, 3}, new double[] {20, 40, 90});
        live.apply(Runbook.Op.write(Runbook.Kind.UPDATE, 5, at(5)));
        expectTop(live, 3, "after a replacement", new long[] {5, 4, 1}, new double[] {5, 20, 40});
        live.apply(Runbook.Op.write(Runbook.Kind.UPDATE, 4, at(150)));
        expectTop(live, 3, "after a replacement that leaves the answer", new long[] {5, 1, 3}, new double[] {5, 40, 90});
        live.apply(Runbook.Op.write(Runbook.Kind.REUSE, 2, at(60)));
        expectTop(live, 3, "after a reuse of a deleted id", new long[] {5, 1, 2}, new double[] {5, 40, 60});
        live.apply(create(9, 60));
        live.apply(create(7, 60));
        expectTop(live, 5, "a tie goes to the smaller id", new long[] {5, 1, 2, 7, 9}, new double[] {5, 40, 60, 60, 60});
        check(live.size() == 7, "live rows: " + live.size());
        check(refused(live, create(5, 1)), "a create of a live id is refused");
        check(refused(live, delete(100)), "a delete of an id that is not live is refused");
        check(refused(live, Runbook.Op.write(Runbook.Kind.UPDATE, 100, at(1))), "an update of an id that is not live is refused");
        expectTop(live, 5, "a refused write changes nothing", new long[] {5, 1, 2, 7, 9}, new double[] {5, 40, 60, 60, 60});

        // recall: of the best three (5, 1, 2) an answer holds two, and a deleted id and a repeated id fill no place
        float[] q = at(0);
        LiveRows.Top top = live.top(q, 3);
        check(close(live.recall(top, q, hits(5, 1, 2)), 1), "recall of the exact answer");
        check(close(live.recall(top, q, hits(5, 1, 3)), 2.0 / 3), "recall with one wrong id");
        check(close(live.recall(top, q, hits(5, 5, 100)), 1.0 / 3), "recall with a repeated id and an id that is not live");
        check(close(live.recall(top, q, hits(5, 1, 9)), 1), "recall counts a tie at the boundary");
    }

    private static boolean refused(LiveRows live, Runbook.Op write) {
        try {
            live.apply(write);
            return false;
        } catch (IllegalStateException e) {
            return true;
        }
    }

    private static boolean close(double a, double b) { return Math.abs(a - b) < 1e-9; }

    private static Hit[] hits(long... ids) {
        Hit[] h = new Hit[ids.length];
        for (int i = 0; i < ids.length; i++) h[i] = new Hit(ids[i], 0f);
        return h;
    }

    // ---- the runbook ----

    private static Runbook.Params params(long seed) { return new Runbook.Params(seed, 8, 60, 2, 3.0, 6, 0.5, 4, 5, 10); }

    private static void runbookFile() throws IOException {
        Path dir = Files.createTempDirectory("runbook");
        Path a = dir.resolve("a.rb"), b = dir.resolve("b.rb"), c = dir.resolve("c.rb");
        Runbook made = Runbook.generate(params(7));
        String shaA = made.write(a), shaB = Runbook.generate(params(7)).write(b), shaC = Runbook.generate(params(8)).write(c);
        check(shaA.equals(shaB) && Files.mismatch(a, b) == -1, "one seed gives one file");
        check(!shaA.equals(shaC), "another seed gives another file");
        Runbook.Read read = Runbook.read(a);
        check(read.sha256().equals(shaA), "the checksum of the read is the one of the write");
        check(read.runbook().params.equals(made.params), "the parameters come back");
        check(same(read.runbook(), made), "the operations come back");
        check(read.runbook().write(b).equals(shaA), "the runbook that was read writes the same bytes");

        byte[] bytes = Files.readAllBytes(a);
        Files.write(c, Arrays.copyOf(bytes, bytes.length - 3));
        check(unreadable(c), "a file that is cut short is refused");
        Files.write(c, Arrays.copyOf(bytes, bytes.length + 1));
        check(unreadable(c), "a file with bytes after the last operation is refused");
        byte[] other = bytes.clone();
        other[0] ^= 1;
        Files.write(c, other);
        check(unreadable(c), "a file that is not a runbook is refused");
        for (Path p : List.of(a, b, c, dir)) Files.delete(p);
    }

    private static boolean unreadable(Path file) {
        try {
            Runbook.read(file);
            return false;
        } catch (IOException e) {
            return true;
        }
    }

    private static boolean same(Runbook a, Runbook b) {
        if (a.collections.size() != b.collections.size()) return false;
        for (int c = 0; c < a.collections.size(); c++) {
            List<Runbook.Op> x = a.collections.get(c), y = b.collections.get(c);
            if (x.size() != y.size()) return false;
            for (int i = 0; i < x.size(); i++) if (!same(x.get(i), y.get(i))) return false;
        }
        return true;
    }

    private static boolean same(Runbook.Op a, Runbook.Op b) {
        if (a.kind() != b.kind() || a.id() != b.id() || !Arrays.equals(a.vec(), b.vec()) || !Arrays.deepEquals(a.queries(), b.queries())) return false;
        if (a.writes() == null || b.writes() == null) return a.writes() == b.writes();
        if (a.writes().size() != b.writes().size()) return false;
        for (int i = 0; i < a.writes().size(); i++) if (!same(a.writes().get(i), b.writes().get(i))) return false;
        return true;
    }

    private static void runbookWorkload() {
        for (Runbook.Params p : List.of(params(7), new Runbook.Params(3, 8, 4, 3, 3.0, 4, 4.0, 3, 2, 50), new Runbook.Params(5, 8, 500, 1, 3.0, 16, 0.125, 10, 5, 50),
                new Runbook.Params(5, 8, 100, 1, 3.0, 16, 0.0, 10, 5, 0))) {
            Runbook rb = Runbook.generate(p);
            check(rb.collections.size() == p.collections(), "collections of " + p.describe());
            for (List<Runbook.Op> ops : rb.collections) workload(p, ops);
        }
        Runbook two = Runbook.generate(params(7));
        check(!same(two.collections.get(0).get(0), two.collections.get(1).get(0)), "two collections of a runbook differ");
    }

    /** replays the operations of one collection against a set of live ids and checks what the generator promises */
    private static void workload(Runbook.Params p, List<Runbook.Op> ops) {
        LiveRows live = new LiveRows();
        Set<Long> everDeleted = new HashSet<>();
        Map<Runbook.Kind, Integer> seen = new HashMap<>();
        long written = 0, retired = 0, searches = 0;
        int checkpoints = 0;
        boolean unit = true, batches = true, reuse = true, liveAtCheckpoints = true;
        for (Runbook.Op op : ops) {
            seen.merge(op.kind(), 1, Integer::sum);
            if (op.kind() == Runbook.Kind.SEARCH) {
                searches++;
                unit &= unit(op.vec());
            } else if (op.kind() == Runbook.Kind.CHECKPOINT) {
                checkpoints++;
                liveAtCheckpoints &= live.size() == p.rows();
                check(op.queries().length == p.queries(), "queries of a checkpoint");
                for (float[] q : op.queries()) unit &= unit(q);
            } else {
                List<Runbook.Op> writes = op.kind() == Runbook.Kind.BATCH ? op.writes() : List.of(op);
                if (op.kind() == Runbook.Kind.BATCH) {
                    batches &= !writes.isEmpty() && writes.size() <= p.maxBatch() && writes.stream().map(Runbook.Op::id).distinct().count() == writes.size();
                }
                for (Runbook.Op w : writes) {
                    if (op.kind() == Runbook.Kind.BATCH) seen.merge(w.kind(), 1, Integer::sum);
                    written++;
                    if (w.kind() == Runbook.Kind.UPDATE || w.kind() == Runbook.Kind.DELETE) retired++;
                    if (w.kind() == Runbook.Kind.DELETE) everDeleted.add(w.id());
                    if (w.kind() == Runbook.Kind.REUSE) reuse &= everDeleted.contains(w.id());
                    if (w.kind() == Runbook.Kind.CREATE) reuse &= !everDeleted.contains(w.id());
                    if (w.vec() != null) unit &= unit(w.vec()) && w.vec().length == p.dim();
                    live.apply(w); // throws when a write does not fit the live rows
                }
            }
        }
        String what = " of " + p.describe() + " " + seen;
        check(unit, "every vector has length 1" + what);
        check(batches, "a batch has 1 to maxBatch writes of different ids" + what);
        check(reuse, "a reused id was deleted before, and a created one never" + what);
        check(liveAtCheckpoints, "the live rows at every checkpoint" + what);
        check(live.size() == p.rows(), "the live rows at the end" + what);
        check(ops.get(ops.size() - 1).kind() == Runbook.Kind.CHECKPOINT, "the last operation is a checkpoint" + what);
        check(checkpoints >= 2 && checkpoints <= p.checkpoints() + 1, "checkpoints: " + checkpoints + what);
        check(retired >= p.turnover() * p.rows(), "turnover: " + retired + " versions retired" + what);
        check(Math.abs(searches - written * p.searchesPerWrite()) <= 1, "searches: " + searches + " for " + written + " written rows" + what);
        for (Runbook.Kind k : List.of(Runbook.Kind.CREATE, Runbook.Kind.UPDATE, Runbook.Kind.DELETE, Runbook.Kind.REUSE)) {
            check(seen.getOrDefault(k, 0) > 0, "the runbook has " + k + what);
        }
        check(p.maxBatch() == 0 ? !seen.containsKey(Runbook.Kind.BATCH) : p.rows() < 100 || seen.containsKey(Runbook.Kind.BATCH), "batches" + what);
    }

    private static boolean unit(float[] v) {
        double n = 0;
        for (float x : v) n += (double) x * x;
        return Math.abs(n - 1) < 1e-5;
    }

    // ---- the violation counters ----

    private static Hit hit(long id, double degrees) { return new Hit(id, (float) Math.cos(Math.toRadians(degrees))); }

    /** the counters after one check of the live rows 1, 2 and 3 at 10, 20 and 30 degrees, with the query at 0 degrees */
    private static String counted(boolean probe, long[] ids, Hit... answer) {
        LiveRows live = new LiveRows();
        for (int i = 1; i <= 3; i++) live.apply(create(i, 10 * i));
        Violations v = new Violations();
        if (probe) v.probeAnswer(live, at(0), ids, answer, "probe");
        else v.searchAnswer(live, at(0), answer, 2, "search");
        StringBuilder s = new StringBuilder();
        for (Violations.Kind k : Violations.Kind.values()) if (v.count(k) > 0) s.append(s.length() == 0 ? "" : " ").append(k.key).append('=').append(v.count(k));
        return s.toString();
    }

    private static void expect(String expected, String got, String what) { check(expected.equals(got), what + ": counted [" + got + "], expected [" + expected + "]"); }

    private static void violationCounters() {
        long[] none = new long[0];
        expect("", counted(false, none, hit(1, 10), hit(2, 20)), "a right answer");
        expect("deletedReturned=1", counted(false, none, hit(1, 10), hit(8, 5)), "a deleted id");
        expect("oldScore=1", counted(false, none, hit(1, 50), hit(2, 20)), "the score of an old vector");
        expect("tooFewHits=1", counted(false, none, hit(1, 10)), "one hit although two rows qualify");
        expect("tooFewHits=1", counted(false, none, hit(1, 10), hit(1, 10)), "the same id twice");
        expect("", counted(true, new long[] {2, 8}, hit(2, 20)), "a probe that sees the live id and not the dead one");
        expect("writeNotSeen=1", counted(true, new long[] {2}), "a probe that misses a write");
        expect("writeNotSeen=2", counted(true, new long[] {1, 2, 8}), "a probe that misses two writes");
        expect("deletedReturned=1", counted(true, new long[] {8}, hit(8, 5)), "a probe that returns a deleted id");
        expect("oldScore=1", counted(true, new long[] {2}, hit(2, 70)), "a probe that returns an old score");
        boolean threw = false;
        try {
            counted(true, new long[] {2}, hit(1, 10), hit(2, 20));
        } catch (IllegalStateException e) {
            threw = true;
        }
        check(threw, "a hit that the filter does not allow stops the run");

        // a batch that updates 1 (10 to 40 degrees), deletes 2 and creates 3
        long[] ids = {1, 2, 3};
        float[][] before = {at(10), at(20), null}, after = {at(40), null, at(30)};
        check(!Violations.showsPart(at(0), ids, before, after, new Hit[] {hit(1, 10), hit(2, 20)}), "the state before the batch");
        check(!Violations.showsPart(at(0), ids, before, after, new Hit[] {hit(1, 40), hit(3, 30)}), "the state after the batch");
        check(Violations.showsPart(at(0), ids, before, after, new Hit[] {hit(1, 40), hit(2, 20)}), "an update of the batch without its delete");
        check(Violations.showsPart(at(0), ids, before, after, new Hit[] {hit(1, 10), hit(2, 20), hit(3, 30)}), "a create of the batch alone");
        check(Violations.showsPart(at(0), ids, before, after, new Hit[] {hit(3, 30)}), "an updated row that is missing in both states");
        check(Violations.showsPart(at(0), ids, before, after, new Hit[] {hit(1, 75), hit(3, 30)}), "a score of neither state");

        Violations v = new Violations();
        v.add(Violations.Kind.PART_OF_BATCH, "x");
        check(v.failing(true) == 1 && v.failing(false) == 0, "a batch in part fails only an arm that promises atomic batches");
        v.add(Violations.Kind.WRITE_NOT_SEEN, "y");
        Violations sum = new Violations();
        sum.add(v);
        sum.add(v);
        check(sum.failing(false) == 2 && sum.count(Violations.Kind.PART_OF_BATCH) == 2 && sum.examples.size() == 4, "the counters add up");
        SizedIdFilter only = Violations.only(new long[] {4, 5, 5});
        check(only.test(4) && !only.test(6) && only.allowedCount() == 2, "the filter of a probe");
    }
}
