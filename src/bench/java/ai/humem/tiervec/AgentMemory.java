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
 * to a file, reads the file back, and replays it into every variant. It counts violations, scores recall at every checkpoint, and writes the
 * result files into {@code --out}: {@code result.txt} (one line per fact), {@code results.tsv} (one row per shape and variant),
 * {@code curve.tsv} (recall per checkpoint), and {@code summary.md}. Nothing is timed. Exit code 0: no failure; 1: a violation that fails the
 * run, a failed arm, or recall below the floor; 2: wrong options.
 *
 * <p>{@code --preset screen} is the quick round for an idea: the small shapes, many variants, and measures that do not depend on timing. It
 * needs the copy of the library that counts distance computations (scripts/instrument_distances.py).
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

    private static final String SCREEN_VARIANTS = "@src/bench/agent-memory-screen.variants";

    private static Map<String, String> defaults(String preset) {
        boolean screen = switch (preset) {
            case "smoke" -> false;
            case "screen" -> true;
            default -> throw new IllegalArgumentException("--preset is smoke or screen");
        };
        Map<String, String> d = new LinkedHashMap<>();
        d.put("preset", preset);
        d.put("shapes", screen ? "p20,p200,p2000,a10k" : "smoke,p20"); // names of SHAPES, separated by ','
        d.put("variants", screen ? SCREEN_VARIANTS : SMOKE_VARIANTS);  // 'name arm key=value ...' separated by ';', or @file with one per line
        d.put("seed", "1");
        d.put("dim", screen ? "768" : "64");     // the data of the screen was chosen by the rule of docs/SCREEN-IDEAS.md
        d.put("latent", screen ? "768" : "0");   // the kind of data, see Runbook: 0, or the dimensions that the noise of the rows spans
        d.put("clusters", screen ? "8" : "0");   // the clusters of a collection; 0: one per 20 rows, at least 4 and at most 16
        d.put("mix", screen ? "write" : "read"); // read: 4 searches per written row; write: 1 search per 8 written rows
        d.put("turnover", "3");                  // retired row versions per live row
        d.put("checkpoints", "10");
        d.put("min-recall", "0");                // the run fails when the mean recall of a checkpoint is below this
        d.put("out", "target/agent-memory/out"); // the directory of the runbooks and the results
        return d;
    }

    private final List<String> lines = new ArrayList<>();
    private final List<String> table = new ArrayList<>();
    private final List<String> curve = new ArrayList<>();
    private final List<String> graphs = new ArrayList<>();
    private final List<String> summary = new ArrayList<>();
    private final boolean counted = DistanceCount.active();
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
        Map<String, String> o;
        List<Shape> shapes = new ArrayList<>();
        List<MemoryArm.Variant> variants;
        List<Runbook.Params> params = new ArrayList<>();
        double floor;
        try {
            if (args.length % 2 != 0) throw new IllegalArgumentException("options are --key value pairs");
            String preset = "smoke";
            for (int i = 0; i < args.length; i += 2) if (args[i].equals("--preset")) preset = args[i + 1];
            o = defaults(preset);
            for (int i = 0; i < args.length; i += 2) {
                if (!args[i].startsWith("--") || !o.containsKey(args[i].substring(2))) {
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
            String text = o.get("variants");
            variants = MemoryArm.Variant.parseAll(text.startsWith("@") ? Files.readString(Path.of(text.substring(1))) : text);
            double searchesPerWrite = switch (o.get("mix")) {
                case "read" -> 4.0;
                case "write" -> 0.125;
                default -> throw new IllegalArgumentException("--mix is read or write");
            };
            int clusters = Integer.parseInt(o.get("clusters"));
            for (Shape s : shapes) {
                params.add(new Runbook.Params(Long.parseLong(o.get("seed")), Integer.parseInt(o.get("dim")), s.rows(), s.collections(),
                        Double.parseDouble(o.get("turnover")), clusters == 0 ? Math.max(4, Math.min(16, s.rows() / 20)) : clusters, searchesPerWrite,
                        Integer.parseInt(o.get("checkpoints")), s.queries(), 50, Integer.parseInt(o.get("latent"))));
            }
            floor = Double.parseDouble(o.get("min-recall"));
        } catch (IllegalArgumentException | IOException e) {
            System.err.println(e);
            System.exit(2);
            return;
        }
        AgentMemory run = new AgentMemory();
        if (o.get("preset").equals("screen") && !run.counted) {
            System.err.println("the screen reports distance computations: compile the copy of the library that scripts/instrument_distances.py makes");
            System.exit(2);
        }
        Path out = Path.of(o.get("out"));
        Files.createDirectories(out);
        run.line("AGENT-MEMORY format=1");
        String commit = Bench.sh("git", "rev-parse", "HEAD");
        String machine = "commit=" + (commit == null ? "unknown" : commit) + " jdk=" + System.getProperty("java.runtime.version")
                + " arch=" + System.getProperty("os.arch") + " kernels=" + (Kernels.VECTOR ? "vector" : "scalar") + " lanes=" + Kern.lanes();
        run.line("RUN started=" + Instant.now() + " " + machine);
        run.line("OPTIONS " + o);
        run.table.add(String.join("\t", "shape", "variant", "arm", "settings", "atomicBatches", "violations", "writeNotSeen", "deletedReturned", "oldScore",
                "partOfBatch", "tooFewHits", "recallFinal", "recallLowest", "recallChurnMean", "distancesPerSearch", "quantizedPerSearch", "distancesPerRowWritten",
                "rowsBuiltPerRowWritten", "rowsRebuiltPerRowWritten", "bytesHeldPerRow", "heapBytesPerRow"));
        run.curve.add(String.join("\t", "shape", "variant", "checkpoint", "recall"));
        run.graphs.add(String.join("\t", "shape", "variant", "checkpoint", GraphShape.HEADER));
        run.summary.add("## Agent-memory benchmark, preset " + o.get("preset"));
        run.summary.add("");
        run.summary.add("Seed " + o.get("seed") + ", " + o.get("dim") + " dimensions, " + o.get("mix") + " mix, generated data with latent "
                + o.get("latent") + ". " + machine.replace(" ", ", ")
                + ". No timing. A `*` marks an arm without atomic batches, whose batches in part do not fail the run. A `†` marks an arm that does not"
                + " promise `k` ids, whose shorter answers do not fail the run.");
        for (int i = 0; i < shapes.size(); i++) run.shape(shapes.get(i), params.get(i), variants, floor, out);
        run.line("SUMMARY status=" + (run.failures == 0 ? "OK" : "FAILED") + " failures=" + run.failures);
        run.summary.add("");
        run.summary.add(run.failures == 0 ? "No failure." : "**" + run.failures + " failures**, see `result.txt`.");
        Files.write(out.resolve("result.txt"), run.lines);
        Files.write(out.resolve("results.tsv"), run.table);
        Files.write(out.resolve("curve.tsv"), run.curve);
        Files.write(out.resolve("graph.tsv"), run.graphs);
        Files.write(out.resolve("summary.md"), run.summary);
        System.exit(run.failures == 0 ? 0 : 1);
    }

    private void shape(Shape shape, Runbook.Params params, List<MemoryArm.Variant> variants, double floor, Path out) throws IOException {
        Path file = out.resolve(shape.name() + ".runbook");
        String written = Runbook.generate(params).write(file);
        Runbook.Read read = Runbook.read(file);
        if (!read.sha256().equals(written)) throw new IOException(file + " changed between the write and the read");
        line("SHAPE name=" + shape.name() + " probeEvery=" + shape.probeEvery() + " runbookSha256=" + written + " " + params.describe());
        summary.add("");
        summary.add("### " + shape.name() + ": " + shape.collections() + " x " + shape.rows() + " rows, runbook `" + written.substring(0, 12) + "`");
        summary.add("");
        MemoryReplay.Answers answers = new MemoryReplay.Answers(read.runbook());
        int neighbours = Math.min(MemoryReplay.DIMENSION_NEIGHBOURS, shape.rows() - 1);
        line(Bench.f("DATA shape=%s queries=%d intrinsicDimension=%.1f dimensionNeighbours=%d contrast10=%.3f", shape.name(), answers.queries,
                answers.intrinsicDimension(), neighbours, answers.contrast()));
        summary.add(Bench.f("How hard the queries of the checkpoints are: local intrinsic dimension %.1f (from %d neighbours), relative contrast %.3f"
                + " (mean distance over the distance of the 10th nearest row).", answers.intrinsicDimension(), neighbours, answers.contrast()));
        summary.add("");
        summary.add("| Variant | Violations | Batches in part | Recall, lowest | Recall, mean after the load | Recall, final | Distances per search "
                + "| Rows built per row written | Distances per row written | Bytes held per row | Heap bytes per row |");
        summary.add("|---|---|---|---|---|---|---|---|---|---|---|");
        for (MemoryArm.Variant v : variants) {
            String who = "shape=" + shape.name() + " variant=" + v.name();
            try {
                PruneCount.reset();
                MemoryReplay.Result r = MemoryReplay.run(read.runbook(), answers, () -> v.open(params.dim()), shape.probeEvery());
                report(shape.name(), who, v, r, floor);
                pruning(who);
            } catch (RuntimeException | OutOfMemoryError e) {
                e.printStackTrace();
                failure(who + " the replay stopped: " + e);
                summary.add("| " + v.name() + " | the replay stopped | | | | | | | | | |");
            }
        }
    }

    /** what the neighbour selection did in the replay of one variant; no line in a build that does not count, or an arm without graphs */
    private void pruning(String who) {
        for (int s = 0; s < PruneCount.SITES.length; s++) {
            if (PruneCount.count(s, 0) == 0) continue;
            StringBuilder b = new StringBuilder("PRUNE " + who + " site=" + PruneCount.SITES[s]);
            for (int c = 0; c < PruneCount.COUNTS.length; c++) b.append(' ').append(PruneCount.COUNTS[c]).append('=').append(PruneCount.count(s, c));
            line(b.toString());
        }
    }

    /** a count of distances over a number of things, or "-" in a build that does not count */
    private String per(long distances, long things) { return counted ? Bench.f("%.1f", (double) distances / things) : "-"; }

    private void report(String shape, String who, MemoryArm.Variant v, MemoryReplay.Result r, double floor) {
        line("VARIANT " + who + " arm=" + v.arm() + " atomicBatches=" + r.atomicBatches + " fullAnswers=" + r.fullAnswers + " settings=" + v.describe());
        if (r.about != null) line("ABOUT " + who + " " + r.about);
        if (!r.counters.isEmpty()) {
            StringBuilder own = new StringBuilder("COUNTERS " + who);
            r.counters.forEach((key, value) -> own.append(' ').append(key).append('=').append(value));
            line(own.toString());
        }
        for (int i = 0; i < r.recallSum.length; i++) {
            line(Bench.f("CHECKPOINT %s n=%d recall=%.4f", who, i, r.recall(i)));
            curve.add(Bench.f("%s\t%s\t%d\t%.4f", shape, v.name(), i, r.recall(i)));
            if (r.shapes[i].graphs > 0) graphs.add(shape + "\t" + v.name() + "\t" + i + "\t" + r.shapes[i].row());
        }
        StringBuilder counts = new StringBuilder(), cells = new StringBuilder();
        for (Violations.Kind k : Violations.Kind.values()) {
            counts.append(' ').append(k.key).append('=').append(r.violations.count(k));
            cells.append('\t').append(r.violations.count(k));
        }
        long failing = r.violations.failing(r.atomicBatches, r.fullAnswers);
        String recallFinal = Bench.f("%.4f", r.finalRecall()), recallLowest = Bench.f("%.4f", r.lowestRecall());
        String recallChurnMean = Bench.f("%.4f", r.churnMeanRecall());
        String perSearch = per(r.searchDistances, r.checkpointSearches), quantized = per(r.searchQuantized, r.checkpointSearches);
        String perWrite = per(r.writeDistances, r.added);
        String built = Bench.f("%.3f", (double) r.rowsBuilt / r.added), rebuilt = Bench.f("%.3f", (double) r.rowsRebuilt / r.added);
        String held = Bench.f("%.0f", (double) r.bytesHeld / r.liveRows), heap = Bench.f("%.0f", (double) r.heapBytes / r.liveRows);
        line("RESULT " + who + " violations=" + failing + counts + " added=" + r.added + " deleted=" + r.deleted + " batches=" + r.batches
                + " searches=" + r.searches + " probes=" + r.probes + " recallFinal=" + recallFinal + " recallLowest=" + recallLowest
                + " recallChurnMean=" + recallChurnMean + " rowsBuiltPerWrite=" + built + " rowsRebuiltPerWrite=" + rebuilt + " bytesHeldPerRow=" + held + " distancesPerSearch=" + perSearch
                + " quantizedPerSearch=" + quantized + " distancesPerRowWritten=" + perWrite + " heapBytesPerRow=" + heap);
        table.add(String.join("\t", shape, v.name(), v.arm(), v.describe(), String.valueOf(r.atomicBatches), String.valueOf(failing)) + cells + "\t"
                + String.join("\t", recallFinal, recallLowest, recallChurnMean, perSearch, quantized, perWrite, built, rebuilt, held, heap));
        summary.add("| " + String.join(" | ", v.name() + (r.atomicBatches ? "" : " \\*") + (r.fullAnswers ? "" : " †"), String.valueOf(failing),
                String.valueOf(r.violations.count(Violations.Kind.PART_OF_BATCH)), recallLowest, recallChurnMean, recallFinal, perSearch, built, perWrite,
                held, heap) + " |");
        for (String x : r.violations.examples) line("EXAMPLE " + who + " " + x);
        if (failing > 0) failure(who + " " + failing + " violations");
        for (String x : r.failures) failure(who + " " + x);
        if (r.lowestRecall() < floor) failure(Bench.f("%s recall %.4f is below the floor %.4f", who, r.lowestRecall(), floor));
    }
}
