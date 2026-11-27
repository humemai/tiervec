package ai.humem.tiervec;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Collections;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;
import java.util.function.Consumer;

/**
 * The text form of {@link TieredIndex.Config} and the fingerprint over its data-affecting part: the code behind {@code Config.toMap},
 * {@code Config.fromMap}, {@code Config.fromStoredMap} and {@code Config.fingerprint}. The keys are the names of the record components, and
 * a graph parameter has the prefix {@code graph.}.
 */
final class ConfigMap {
    /** the keys that an open index accepts a new value for: the search-time knobs */
    static final String EF_SEARCH = "graph.efSearch", SPLIT_BEAM = "splitBeam";

    private ConfigMap() {}

    static Map<String, String> write(TieredIndex.Config c) {
        GraphIndex.Params g = c.graph();
        Map<String, String> m = new LinkedHashMap<>();
        m.put("tailLimit", Integer.toString(c.tailLimit()));
        m.put("tailAgeMs", Long.toString(c.tailAgeMs()));
        m.put("smallLimit", Integer.toString(c.smallLimit()));
        m.put("maxSegments", Integer.toString(c.maxSegments()));
        m.put("mergeFan", Integer.toString(c.mergeFan()));
        m.put("deadFraction", Double.toString(c.deadFraction()));
        m.put("mergeRatePerSec", Integer.toString(c.mergeRatePerSec()));
        m.put("hardTailFactor", Integer.toString(c.hardTailFactor()));
        m.put("tailQuant", c.tailQuant().name());
        m.put("graph.M", Integer.toString(g.M()));
        m.put("graph.efConstruction", Integer.toString(g.efConstruction()));
        m.put(EF_SEARCH, Integer.toString(g.efSearch()));
        m.put("graph.alpha", Float.toString(g.alpha()));
        m.put("graph.sq8", Boolean.toString(g.sq8()));
        m.put("graph.seed", Long.toString(g.seed()));
        m.put("buildThreads", Integer.toString(c.buildThreads()));
        m.put(SPLIT_BEAM, Boolean.toString(c.splitBeam()));
        m.put("mergeTriggerRatio", Double.toString(c.mergeTriggerRatio()));
        m.put("mergeMode", c.mergeMode().name());
        m.put("vectorStorage", c.vectorStorage().name());
        return Collections.unmodifiableMap(m);
    }

    /**
     * The configuration that {@code map} describes: the defaults, with the keys of the map on top. A value that cannot be read or is out of
     * range is refused. An unknown key is refused when {@code ignored} is null, and otherwise reported to it and left out.
     */
    static TieredIndex.Config read(Map<String, String> map, Consumer<String> ignored) {
        Map<String, String> m = new LinkedHashMap<>(write(TieredIndex.Config.defaults()));
        for (Map.Entry<String, String> e : new TreeMap<>(map).entrySet()) { // sorted: the first refusal and the warnings do not depend on the map
            String k = e.getKey();
            if (!m.containsKey(k)) {
                if (ignored == null) throw new IllegalArgumentException("unknown configuration key '" + k + "'");
                ignored.accept("unknown configuration key '" + k + "' ignored");
                continue;
            }
            if (e.getValue() == null) throw new IllegalArgumentException("configuration key '" + k + "' has no value");
            m.put(k, e.getValue());
        }
        GraphIndex.Params graph = new GraphIndex.Params((int) whole(m, "graph.M", 2, Integer.MAX_VALUE / 2),
                (int) whole(m, "graph.efConstruction", 1, Integer.MAX_VALUE), (int) whole(m, EF_SEARCH, 0, Integer.MAX_VALUE),
                alpha(m), flag(m, "graph.sq8"), whole(m, "graph.seed", Long.MIN_VALUE, Long.MAX_VALUE));
        Quant tailQuant = constant(m, "tailQuant", Quant.class);
        if (tailQuant == Quant.BINARY) throw bad(m, "tailQuant", "NONE or SQ8");
        // tailAgeMs and mergeRatePerSec take any number: 0 and below mean "never by age" and "no cap"
        return new TieredIndex.Config((int) whole(m, "tailLimit", 1, Integer.MAX_VALUE), whole(m, "tailAgeMs", Long.MIN_VALUE, Long.MAX_VALUE),
                (int) whole(m, "smallLimit", 0, Integer.MAX_VALUE), (int) whole(m, "maxSegments", 1, Integer.MAX_VALUE),
                (int) whole(m, "mergeFan", 2, Integer.MAX_VALUE), number(m, "deadFraction"),
                (int) whole(m, "mergeRatePerSec", Integer.MIN_VALUE, Integer.MAX_VALUE),
                (int) whole(m, "hardTailFactor", 2, Integer.MAX_VALUE), tailQuant, graph,
                (int) whole(m, "buildThreads", 1, Integer.MAX_VALUE), flag(m, SPLIT_BEAM), ratio(m, "mergeTriggerRatio"),
                constant(m, "mergeMode", TieredIndex.MergeMode.class), constant(m, "vectorStorage", TieredIndex.VectorStorage.class));
    }

    /** {@code c} with a new value for a search-time knob; every other key is refused */
    static TieredIndex.Config reconfigured(TieredIndex.Config c, String key, String value) {
        Map<String, String> one = Collections.singletonMap(key, String.valueOf(value));
        if (key.equals(SPLIT_BEAM)) return c.withSplitBeam(flag(one, key));
        if (key.equals(EF_SEARCH)) {
            GraphIndex.Params g = c.graph();
            int ef = (int) whole(one, key, 0, Integer.MAX_VALUE);
            return c.withGraph(new GraphIndex.Params(g.M(), g.efConstruction(), ef, g.alpha(), g.sq8(), g.seed()));
        }
        throw new IllegalArgumentException("configuration key '" + key + "' cannot change in an open index");
    }

    /**
     * The fingerprint of an index definition: what decides the data of the graphs and their files. The search-time knobs, the settings of the
     * tail (which is not persisted), the maintenance policy and the random seed of the graph levels are not in it.
     */
    static TieredIndex.Fingerprint fingerprint(TieredIndex.Config c, int dim, Metric metric) {
        if (dim < 1) throw new IllegalArgumentException("dim");
        Objects.requireNonNull(metric, "metric");
        GraphIndex.Params g = c.graph();
        String text = "tiervec-fingerprint=1;dim=" + dim + ";metric=" + metric.name() + ";idBytes=" + SegmentFormat.ID_WIDTH + ";graph.M=" + g.M()
                + ";graph.efConstruction=" + g.efConstruction() + ";graph.alpha=" + Float.toString(g.alpha()) + ";graph.sq8=" + g.sq8()
                + ";layer0=csr";
        try {
            byte[] d = MessageDigest.getInstance("SHA-256").digest(text.getBytes(StandardCharsets.UTF_8));
            return new TieredIndex.Fingerprint(text, HexFormat.of().formatHex(d));
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("every Java platform has SHA-256", e);
        }
    }

    private static IllegalArgumentException bad(Map<String, String> m, String key, String wanted) {
        return new IllegalArgumentException("configuration key '" + key + "' is '" + m.get(key) + "', it must be " + wanted);
    }

    private static long whole(Map<String, String> m, String key, long min, long max) {
        try {
            long v = Long.parseLong(m.get(key));
            if (v >= min && v <= max) return v;
        } catch (NumberFormatException e) {
            // refused below, with the key named
        }
        throw bad(m, key, "a whole number from " + min + " to " + max);
    }

    /** any number but NaN */
    private static double number(Map<String, String> m, String key) {
        try {
            double v = Double.parseDouble(m.get(key));
            if (!Double.isNaN(v)) return v;
        } catch (NumberFormatException e) {
            // refused below, with the key named
        }
        throw bad(m, key, "a number");
    }

    /** a finite number that is not negative */
    private static double ratio(Map<String, String> m, String key) {
        double v = number(m, key);
        if (v >= 0.0 && v < Double.POSITIVE_INFINITY) return v;
        throw bad(m, key, "a finite number of at least 0");
    }

    private static float alpha(Map<String, String> m) {
        try {
            float v = Float.parseFloat(m.get("graph.alpha"));
            if (v > 0f && v < Float.POSITIVE_INFINITY) return v;
        } catch (NumberFormatException e) {
            // refused below, with the key named
        }
        throw bad(m, "graph.alpha", "a finite number above 0");
    }

    private static boolean flag(Map<String, String> m, String key) {
        String v = m.get(key);
        if (v.equals("true")) return true;
        if (v.equals("false")) return false;
        throw bad(m, key, "true or false");
    }

    private static <E extends Enum<E>> E constant(Map<String, String> m, String key, Class<E> type) {
        for (E e : type.getEnumConstants()) {
            if (e.name().equals(m.get(key))) return e;
        }
        throw bad(m, key, "one of " + java.util.Arrays.toString(type.getEnumConstants()));
    }
}
