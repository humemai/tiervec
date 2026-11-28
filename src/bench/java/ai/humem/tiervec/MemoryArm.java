package ai.humem.tiervec;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executor;
import java.util.function.LongPredicate;

/**
 * One design under test in the agent-memory benchmark (docs/AGENT-MEMORY-BENCHMARK.md, issue #159): an index of this library, driven the way
 * a host drives it. All maintenance runs on the calling thread at a point that the writes alone decide, so that a replay of one runbook gives
 * the same index every time.
 */
interface MemoryArm extends AutoCloseable {
    /** a create, an update, or a reuse: the id has this vector now */
    void put(long id, float[] v);

    void delete(long id);

    /** whether a search sees all of a batch or none of it */
    default boolean atomicBatches() { return false; }

    /** applies the writes of a batch at once; the replay calls it only on an arm with atomic batches, and applies one by one on the others */
    default void applyBatch(List<Runbook.Op> writes) { throw new UnsupportedOperationException("this arm has no atomic batches"); }

    /** the maintenance that is due after a write or a batch */
    default void maintain() {}

    Hit[] search(float[] q, int k, LongPredicate allow);

    /** what an arm does only for a checkpoint, before its graphs are read and its searches are scored */
    default void beforeCheckpoint() {}

    /** the search of a checkpoint, which an arm may answer from what it built for the checkpoint */
    default Hit[] checkpointSearch(float[] q, int k) { return search(q, k, null); }

    /** adds the graphs of this arm, as they are at a checkpoint, to the counts of their shape */
    default void describe(GraphShape shape, float[][] centres) {}

    /** the rows that have been put into a graph, first builds and rebuilds */
    default long rowsBuilt() { return 0; }

    /** the rows that a merge has put into a graph again */
    default long rowsRebuilt() { return 0; }

    /** the bytes that the index says it holds, see {@link VecIndex#bytesHeld} */
    long bytesHeld();

    /** a failure of the maintenance, or null */
    default String failure() { return null; }

    @Override void close();

    /**
     * A named variant: an arm and its settings, written {@code name arm key=value ...}. The arms are {@code exact}, {@code tier},
     * {@code graph}, and {@code rebuild}.
     */
    record Variant(String name, String arm, Map<String, String> settings) {
        /** the settings of a tier that would make a replay depend on timing or on thread scheduling, with the values that the benchmark fixes */
        private static final Map<String, String> FIXED = Map.of("buildThreads", "1", "tailAgeMs", "0", "mergeRatePerSec", "0");

        /** variants separated by ';' or by line breaks; a part that starts with '#' is a comment. A name may appear once. */
        static List<Variant> parseAll(String text) {
            List<Variant> all = new ArrayList<>();
            for (String line : text.split("\n")) {
                if (line.strip().startsWith("#")) continue;
                for (String part : line.split(";")) {
                    if (part.isBlank()) continue;
                    Variant v = parse(part.strip());
                    for (Variant other : all) if (other.name().equals(v.name())) throw new IllegalArgumentException("two variants are named " + v.name());
                    all.add(v);
                }
            }
            if (all.isEmpty()) throw new IllegalArgumentException("no variant");
            return all;
        }

        /** one variant; its settings are checked here, so that a wrong one ends the run before any replay */
        static Variant parse(String text) {
            String[] words = text.split("\\s+");
            if (words.length < 2 || !words[0].matches("[A-Za-z0-9._-]+")) throw new IllegalArgumentException("a variant is 'name arm key=value ...', not '" + text + "'");
            Map<String, String> settings = new LinkedHashMap<>();
            for (int i = 2; i < words.length; i++) {
                int eq = words[i].indexOf('=');
                if (eq < 1 || settings.put(words[i].substring(0, eq), words[i].substring(eq + 1)) != null) {
                    throw new IllegalArgumentException("bad or repeated setting '" + words[i] + "' of the variant " + words[0]);
                }
            }
            Variant v = new Variant(words[0], words[1], settings);
            v.open(2).close();
            return v;
        }

        String describe() {
            StringBuilder s = new StringBuilder();
            for (Map.Entry<String, String> e : settings.entrySet()) s.append(s.length() == 0 ? "" : ",").append(e.getKey()).append('=').append(e.getValue());
            return s.length() == 0 ? "defaults" : s.toString();
        }

        MemoryArm open(int dim) {
            return switch (arm) {
                case "exact" -> {
                    refuse(settings, List.of());
                    yield new Exact(dim);
                }
                case "tier" -> new Tier(dim, tierConfig(settings));
                case "graph" -> {
                    refuse(settings, List.of("graph.M", "graph.efConstruction", "graph.efSearch", "graph.alpha", "graph.sq8", "consolidateAt"));
                    GraphIndex.Params p = graphParams();
                    double at = Double.parseDouble(value("consolidateAt", 0.2));
                    if (!(at >= 0)) throw new IllegalArgumentException("consolidateAt must be 0 or more");
                    yield new Graph(dim, p, at);
                }
                case "rebuild" -> {
                    refuse(settings, List.of("graph.M", "graph.efConstruction", "graph.efSearch", "graph.alpha", "graph.sq8"));
                    yield new Rebuild(dim, graphParams());
                }
                default -> throw new IllegalArgumentException("unknown arm '" + arm + "' of the variant " + name + " (exact, tier, graph, rebuild)");
            };
        }

        private GraphIndex.Params graphParams() {
            GraphIndex.Params d = GraphIndex.Params.defaults();
            String sq8 = value("graph.sq8", d.sq8());
            if (!sq8.equals("true") && !sq8.equals("false")) throw new IllegalArgumentException("graph.sq8 must be true or false");
            return new GraphIndex.Params(Integer.parseInt(value("graph.M", d.M())), Integer.parseInt(value("graph.efConstruction", d.efConstruction())),
                    Integer.parseInt(value("graph.efSearch", d.efSearch())), Float.parseFloat(value("graph.alpha", d.alpha())), sq8.equals("true"), d.seed());
        }

        private String value(String key, Object otherwise) { return settings.getOrDefault(key, String.valueOf(otherwise)); }

        private void refuse(Map<String, String> given, List<String> known) {
            for (String key : given.keySet()) {
                if (!known.contains(key)) throw new IllegalArgumentException("the arm " + arm + " has no setting '" + key + "' (variant " + name + ")");
            }
        }

        /** the defaults of the tier, the settings of the variant on top, and the fixed ones */
        private static TieredIndex.Config tierConfig(Map<String, String> settings) {
            Map<String, String> m = new LinkedHashMap<>(settings);
            for (Map.Entry<String, String> e : FIXED.entrySet()) {
                if (m.put(e.getKey(), e.getValue()) != null) {
                    throw new IllegalArgumentException("the benchmark fixes " + e.getKey() + "=" + e.getValue() + ", a variant cannot set it");
                }
            }
            return TieredIndex.Config.fromMap(m);
        }
    }

    /** every live row is scanned: {@link FlatIndex}, with its compaction run when its policy asks for it */
    final class Exact implements MemoryArm {
        private final FlatIndex index;

        Exact(int dim) { index = new FlatIndex(dim, Metric.COSINE); }

        @Override public void put(long id, float[] v) { index.add(id, v); }
        @Override public void delete(long id) { index.delete(id); }
        @Override public void maintain() { index.compactIfNeeded(); }
        @Override public Hit[] search(float[] q, int k, LongPredicate allow) { return index.search(q, k, allow); }
        @Override public long bytesHeld() { return index.bytesHeld(); }
        @Override public void close() { index.close(); }
    }

    /**
     * The tiervec tier with a host environment that runs every task on the submitting thread: a seal or a merge runs inside the write that
     * made it due.
     */
    final class Tier implements MemoryArm {
        private static final HostEnv INLINE = new HostEnv() {
            @Override public Executor executor() { return Runnable::run; }
            @Override public Permit acquireBuild(BuildKind kind, long estimatedPeakBytes) { return () -> {}; }
        };

        private final TieredIndex index;

        Tier(int dim, TieredIndex.Config config) { index = new TieredIndex(dim, Metric.COSINE, config, INLINE); }

        @Override public void put(long id, float[] v) { index.add(id, v); }
        @Override public void delete(long id) { index.delete(id); }
        @Override public boolean atomicBatches() { return true; }

        @Override public void applyBatch(List<Runbook.Op> writes) {
            List<TieredIndex.IdVector> adds = new ArrayList<>();
            List<Long> deletes = new ArrayList<>();
            for (Runbook.Op w : writes) {
                if (w.kind() == Runbook.Kind.DELETE) deletes.add(w.id());
                else adds.add(new TieredIndex.IdVector(w.id(), w.vec()));
            }
            index.apply(new TieredIndex.Batch(adds, deletes));
        }

        @Override public void maintain() { index.maintain(); }
        @Override public Hit[] search(float[] q, int k, LongPredicate allow) { return index.search(q, k, allow); }

        @Override public long rowsBuilt() {
            TieredIndex.Stats s = index.stats();
            return s.sealedVectors() + s.mergedVectors();
        }

        @Override public long rowsRebuilt() { return index.stats().mergedVectors(); }
        @Override public void describe(GraphShape shape, float[][] centres) { shape.add(index, centres); }
        @Override public long bytesHeld() { return index.bytesHeld(); }
        @Override public String failure() { return index.lastError() == null ? null : index.lastError().toString(); }
        @Override public void close() { index.close(); }
    }

    /**
     * One mutable {@link GraphIndex} with adds and deletes, and {@code consolidate()} when the deleted nodes are more than
     * {@code consolidateAt} times the live ones.
     */
    final class Graph implements MemoryArm {
        private final GraphIndex index;
        private final double consolidateAt;
        private long added;

        Graph(int dim, GraphIndex.Params params, double consolidateAt) {
            this.index = new GraphIndex(dim, Metric.COSINE, params);
            this.consolidateAt = consolidateAt;
        }

        @Override public void put(long id, float[] v) {
            index.add(id, v);
            added++;
        }

        @Override public void delete(long id) { index.delete(id); }

        @Override public void maintain() {
            int dead = index.tombstoneCount();
            if (dead > 0 && dead > consolidateAt * index.size()) index.consolidate();
        }

        @Override public Hit[] search(float[] q, int k, LongPredicate allow) { return index.search(q, k, allow); }
        @Override public void describe(GraphShape shape, float[][] centres) { shape.add(index, centres); }
        @Override public long rowsBuilt() { return added; }
        @Override public long bytesHeld() { return index.bytesHeld(); }
        @Override public void close() { index.close(); }
    }

    /**
     * The control of the graphs that live through changes (issue #161): at every checkpoint a new {@link GraphIndex} is built from the live
     * rows, in the order of their last write, and the checkpoint is searched in it. Between the checkpoints every search is exact, so this
     * arm says nothing about searches under writes.
     */
    final class Rebuild implements MemoryArm {
        private final int dim;
        private final GraphIndex.Params params;
        private final FlatIndex exact;
        private final LinkedHashMap<Long, float[]> rows = new LinkedHashMap<>();
        private GraphIndex built;
        private long builtRows;

        Rebuild(int dim, GraphIndex.Params params) {
            this.dim = dim;
            this.params = params;
            this.exact = new FlatIndex(dim, Metric.COSINE);
        }

        @Override public void put(long id, float[] v) {
            exact.add(id, v);
            rows.remove(id);
            rows.put(id, v);
        }

        @Override public void delete(long id) {
            exact.delete(id);
            rows.remove(id);
        }

        @Override public void maintain() { exact.compactIfNeeded(); }

        @Override public void beforeCheckpoint() {
            if (built != null) built.close();
            built = new GraphIndex(dim, Metric.COSINE, params);
            rows.forEach(built::add);
            builtRows += rows.size();
        }

        @Override public Hit[] search(float[] q, int k, LongPredicate allow) { return exact.search(q, k, allow); }
        @Override public Hit[] checkpointSearch(float[] q, int k) { return built.search(q, k, null); }
        @Override public void describe(GraphShape shape, float[][] centres) { shape.add(built, centres); }
        @Override public long rowsBuilt() { return builtRows; }
        @Override public long bytesHeld() { return built == null ? 0 : built.bytesHeld(); }

        @Override public void close() {
            if (built != null) built.close();
            exact.close();
        }
    }
}
