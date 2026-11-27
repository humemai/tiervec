package ai.humem.tiervec;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The agent-memory benchmark (docs/AGENT-MEMORY-BENCHMARK.md, issue #159). For every shape it generates a runbook from the seed, writes it
 * to a file, reads the file back, and replays it into every variant. It counts violations, scores recall at every checkpoint, and writes
 * {@code result.txt}. Nothing is timed. Exit code 0: no failure; 1: a violation that fails the run, a failed arm, or recall below the floor;
 * 2: wrong options.
 */
public final class AgentMemory {
    /** the size of a shape. One probe visits every row of a collection, so the large shapes probe only every {@code probeEvery}-th write. */
    private record Shape(String name, int rows, int collections, int queries, int probeEvery) {}

    private static final List<Shape> SHAPES = List.of(new Shape("smoke", 3000, 1, 50, 1), new Shape("p20", 20, 50, 20, 1),
            new Shape("p200", 200, 20, 20, 1), new Shape("p2000", 2000, 5, 40, 1), new Shape("a10k", 10_000, 1, 100, 10),
            new Shape("a100k", 100_000, 1, 100, 100));

    /** the arms of the smoke: the exact scan, the tier (defaults, and small limits that make it seal and merge), and the live graph */
    private static final String SMOKE_VARIANTS = "exact exact; tier tier; tier-small tier tailLimit=300 smallLimit=100 maxSegments=3;"
            + " tier-insert tier tailLimit=300 smallLimit=100 maxSegments=3 mergeMode=INSERT; graph-a10 graph graph.alpha=1.0; graph-a12 graph graph.alpha=1.2";

    private static Map<String, String> defaults() {
        Map<String, String> d = new LinkedHashMap<>();
        d.put("shapes", "smoke,p20");           // names of SHAPES, separated by ','
        d.put("variants", SMOKE_VARIANTS);      // 'name arm key=value ...' separated by ';', see MemoryArm.Variant
        d.put("seed", "1");
        d.put("dim", "64");
        d.put("mix", "read");                   // read: 4 searches per written row; write: 1 search per 8 written rows
        d.put("turnover", "3");                 // retired row versions per live row
        d.put("checkpoints", "10");
        d.put("min-recall", "0");               // the run fails when the mean recall of a checkpoint is below this
        d.put("out", "target/agent-memory/out"); // the directory of the runbooks and the results
        return d;
    }

    private final List<String> lines = new ArrayList<>();
    private int failures;

    private void line(String s) {
        lines.add(s);
        System.out.println(s);
    }

    private void failure(String s) {
        failures++;
        line("FAILURE " + s);
    }

    public static void main(String[] args) throws Exception {
        Map<String, String> o = defaults();
        List<Shape> shapes = new ArrayList<>();
        List<MemoryArm.Variant> variants;
        List<Runbook.Params> params = new ArrayList<>();
        double floor;
        try {
            for (int i = 0; i < args.length; i += 2) {
                if (!args[i].startsWith("--") || !o.containsKey(args[i].substring(2)) || i + 1 >= args.length) {
                    throw new IllegalArgumentException("bad option " + args[i] + " (options are --key value pairs: " + o.keySet() + ")");
                }
                o.put(args[i].substring(2), args[i + 1]);
            }
            for (String name : o.get("shapes").split(",")) {
                Shape shape = SHAPES.stream().filter(s -> s.name().equals(name.strip())).findFirst()
                        .orElseThrow(() -> new IllegalArgumentException("unknown shape '" + name + "', the shapes are " + SHAPES.stream().map(Shape::name).toList()));
                if (shapes.contains(shape)) throw new IllegalArgumentException("the shape " + shape.name() + " is named twice");
                shapes.add(shape);
            }
            variants = MemoryArm.Variant.parseAll(o.get("variants"));
            double searchesPerWrite = switch (o.get("mix")) {
                case "read" -> 4.0;
                case "write" -> 0.125;
                default -> throw new IllegalArgumentException("--mix is read or write");
            };
            for (Shape s : shapes) {
                params.add(new Runbook.Params(Long.parseLong(o.get("seed")), Integer.parseInt(o.get("dim")), s.rows(), s.collections(),
                        Double.parseDouble(o.get("turnover")), Math.max(4, Math.min(16, s.rows() / 20)), searchesPerWrite,
                        Integer.parseInt(o.get("checkpoints")), s.queries(), 50));
            }
            floor = Double.parseDouble(o.get("min-recall"));
        } catch (IllegalArgumentException e) {
            System.err.println(e.getMessage());
            System.exit(2);
            return;
        }
        AgentMemory run = new AgentMemory();
        Path out = Path.of(o.get("out"));
        Files.createDirectories(out);
        run.line("AGENT-MEMORY format=1");
        String commit = Bench.sh("git", "rev-parse", "HEAD");
        run.line("RUN started=" + Instant.now() + " commit=" + (commit == null ? "unknown" : commit) + " jdk=" + System.getProperty("java.runtime.version")
                + " arch=" + System.getProperty("os.arch") + " kernels=" + (Kernels.VECTOR ? "vector" : "scalar") + " lanes=" + Kern.lanes());
        for (int i = 0; i < shapes.size(); i++) run.shape(shapes.get(i), params.get(i), variants, floor, out);
        run.line("SUMMARY status=" + (run.failures == 0 ? "OK" : "FAILED") + " failures=" + run.failures);
        Files.write(out.resolve("result.txt"), run.lines);
        System.exit(run.failures == 0 ? 0 : 1);
    }

    private void shape(Shape shape, Runbook.Params params, List<MemoryArm.Variant> variants, double floor, Path out) throws IOException {
        Path file = out.resolve(shape.name() + ".runbook");
        String written = Runbook.generate(params).write(file);
        Runbook.Read read = Runbook.read(file);
        if (!read.sha256().equals(written)) throw new IOException(file + " changed between the write and the read");
        line("SHAPE name=" + shape.name() + " probeEvery=" + shape.probeEvery() + " runbookSha256=" + written + " " + params.describe());
        MemoryReplay.Answers answers = new MemoryReplay.Answers(read.runbook());
        for (MemoryArm.Variant v : variants) {
            String who = "shape=" + shape.name() + " variant=" + v.name();
            try {
                MemoryReplay.Result r = MemoryReplay.run(read.runbook(), answers, () -> v.open(params.dim()), shape.probeEvery());
                report(who, v, r, floor);
            } catch (RuntimeException | OutOfMemoryError e) {
                e.printStackTrace();
                failure(who + " the replay stopped: " + e);
            }
        }
    }

    private void report(String who, MemoryArm.Variant v, MemoryReplay.Result r, double floor) {
        line("VARIANT " + who + " arm=" + v.arm() + " atomicBatches=" + r.atomicBatches + " settings=" + v.describe());
        for (int i = 0; i < r.recallSum.length; i++) line(Bench.f("CHECKPOINT %s n=%d recall=%.4f", who, i, r.recall(i)));
        StringBuilder counts = new StringBuilder();
        for (Violations.Kind k : Violations.Kind.values()) counts.append(' ').append(k.key).append('=').append(r.violations.count(k));
        long failing = r.violations.failing(r.atomicBatches);
        line(Bench.f("RESULT %s violations=%d%s added=%d deleted=%d batches=%d searches=%d probes=%d recallFinal=%.4f recallLowest=%.4f"
                        + " rowsBuiltPerWrite=%.3f rowsRebuiltPerWrite=%.3f bytesHeldPerRow=%.0f",
                who, failing, counts, r.added, r.deleted, r.batches, r.searches, r.probes, r.finalRecall(), r.lowestRecall(),
                (double) r.rowsBuilt / r.added, (double) r.rowsRebuilt / r.added, (double) r.bytesHeld / r.liveRows));
        for (String x : r.violations.examples) line("EXAMPLE " + who + " " + x);
        if (failing > 0) failure(who + " " + failing + " violations");
        for (String x : r.failures) failure(who + " " + x);
        if (r.lowestRecall() < floor) failure(Bench.f("%s recall %.4f is below the floor %.4f", who, r.lowestRecall(), floor));
    }
}
