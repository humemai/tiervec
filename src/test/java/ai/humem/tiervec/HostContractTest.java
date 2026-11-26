package ai.humem.tiervec;

import static ai.humem.tiervec.SegmentTestSupport.vectors;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BooleanSupplier;
import java.util.function.LongConsumer;
import java.util.function.LongSupplier;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** The host contract of issue #18: the configuration as a map, the fingerprint, version and capabilities, metrics, health, and use after close. */
class HostContractTest {
    private static final int DIM = 8;

    @TempDir
    Path dir;

    private final List<TieredIndex> opened = new ArrayList<>();

    @AfterEach
    void closeAll() {
        for (TieredIndex t : opened) t.close();
    }

    private TieredIndex tier(TieredIndex.Config c) {
        TieredIndex t = new TieredIndex(DIM, Metric.L2, c);
        opened.add(t);
        return t;
    }

    /** no automatic seal: the tail limit is far away and the age rule is off */
    private static TieredIndex.Config quiet() {
        return new TieredIndex.Config(1_000_000, 0L, 10, 8, 4, 0.3, 0, 4, Quant.NONE, GraphIndex.Params.defaults().withEfConstruction(64), 1);
    }

    /** a configuration in which every component differs from the default */
    private static TieredIndex.Config unusual() {
        return new TieredIndex.Config(123, -1L, 7, 3, 5, 1.5, -1, 6, Quant.SQ8, new GraphIndex.Params(12, 77, 55, 1.2f, true, -9L), 17, true, 0.125,
                TieredIndex.MergeMode.INSERT);
    }

    private static void await(BooleanSupplier done, String what) throws InterruptedException {
        long end = System.nanoTime() + 60_000_000_000L;
        while (!done.getAsBoolean()) {
            assertTrue(System.nanoTime() < end, what);
            Thread.sleep(5);
        }
    }

    // ---- the configuration as a map ----

    @Test
    void theMapRoundTrips() {
        for (TieredIndex.Config c : List.of(TieredIndex.Config.defaults(), quiet(), unusual())) {
            Map<String, String> m = c.toMap();
            assertEquals(c, TieredIndex.Config.fromMap(m));
            assertEquals(c, TieredIndex.Config.fromStoredMap(m, w -> {
                throw new AssertionError(w);
            }));
            assertEquals(m, TieredIndex.Config.fromMap(m).toMap());
            assertThrows(UnsupportedOperationException.class, () -> m.put("tailLimit", "1"));
        }
        assertEquals(TieredIndex.Config.defaults(), TieredIndex.Config.fromMap(Map.of()));
        assertEquals(123, TieredIndex.Config.fromMap(Map.of("tailLimit", "123")).tailLimit());
        // the map has one entry per component, the graph parameters one by one in place of "graph"
        assertEquals(TieredIndex.Config.class.getRecordComponents().length - 1 + GraphIndex.Params.class.getRecordComponents().length,
                TieredIndex.Config.defaults().toMap().size());
    }

    @Test
    void anUnknownKeyIsRefusedAtCreationAndIgnoredAtLoad() {
        Map<String, String> m = new LinkedHashMap<>(unusual().toMap());
        m.put("graph.colour", "blue");
        m.put("aNewKnob", "1");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> TieredIndex.Config.fromMap(m));
        assertTrue(e.getMessage().contains("aNewKnob"), e.getMessage());
        List<String> warnings = new ArrayList<>();
        assertEquals(unusual(), TieredIndex.Config.fromStoredMap(m, warnings::add));
        assertEquals(2, warnings.size(), warnings.toString());
        assertTrue(warnings.get(0).contains("aNewKnob") && warnings.get(1).contains("graph.colour"), warnings.toString());
    }

    @Test
    void aValueThatCannotBeUsedIsRefusedWithItsKey() {
        String[][] bad = {{"tailLimit", "0"}, {"tailLimit", "many"}, {"smallLimit", "-1"}, {"maxSegments", "0"}, {"mergeFan", "1"},
                {"hardTailFactor", "1"}, {"buildThreads", "0"}, {"deadFraction", "NaN"}, {"mergeTriggerRatio", "-0.1"}, {"mergeTriggerRatio", "Infinity"},
                {"tailQuant", "BINARY"}, {"tailQuant", "sq8"}, {"mergeMode", "COPY"}, {"splitBeam", "yes"}, {"graph.M", "1"},
                {"graph.efConstruction", "0"}, {"graph.efSearch", "-1"}, {"graph.alpha", "0"}, {"graph.sq8", "TRUE"}, {"graph.seed", "1.5"}};
        for (String[] kv : bad) {
            Map<String, String> m = Map.of(kv[0], kv[1]);
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> TieredIndex.Config.fromMap(m), kv[0] + "=" + kv[1]);
            assertTrue(e.getMessage().contains(kv[0]), e.getMessage());
            assertThrows(IllegalArgumentException.class, () -> TieredIndex.Config.fromStoredMap(m, w -> {}), kv[0] + "=" + kv[1]);
        }
    }

    // ---- the fingerprint ----

    @Test
    void theFingerprintFollowsTheDataAffectingKeysOnly() {
        TieredIndex.Config base = TieredIndex.Config.defaults();
        TieredIndex.Fingerprint fp = base.fingerprint(DIM, Metric.L2);
        assertEquals(64, fp.sha256().length());
        assertTrue(fp.text().startsWith("tiervec-fingerprint=1;dim=8;metric=L2;idBytes=8;"), fp.text());

        // stable under the order of the keys
        List<String> keys = new ArrayList<>(base.toMap().keySet());
        Collections.reverse(keys);
        Map<String, String> reversed = new LinkedHashMap<>();
        for (String k : keys) reversed.put(k, base.toMap().get(k));
        assertEquals(fp, TieredIndex.Config.fromMap(reversed).fingerprint(DIM, Metric.L2));

        // changes with the dimension, the metric and every data-affecting key
        assertNotEquals(fp, base.fingerprint(DIM + 1, Metric.L2));
        assertNotEquals(fp, base.fingerprint(DIM, Metric.COSINE));
        Map<String, String> dataKeys = Map.of("graph.M", "17", "graph.efConstruction", "201", "graph.alpha", "1.1", "graph.sq8", "true");
        // and does not change with any other key
        Map<String, String> otherKeys = new LinkedHashMap<>();
        for (Map.Entry<String, String> e : unusual().toMap().entrySet()) {
            if (!dataKeys.containsKey(e.getKey())) otherKeys.put(e.getKey(), e.getValue());
        }
        assertEquals(base.toMap().size() - dataKeys.size(), otherKeys.size());
        for (Map.Entry<String, String> e : dataKeys.entrySet()) {
            assertNotEquals(fp, TieredIndex.Config.fromMap(Map.of(e.getKey(), e.getValue())).fingerprint(DIM, Metric.L2), e.getKey());
        }
        for (Map.Entry<String, String> e : otherKeys.entrySet()) {
            assertNotEquals(base.toMap().get(e.getKey()), e.getValue(), "the test value of " + e.getKey() + " is the default");
            assertEquals(fp, TieredIndex.Config.fromMap(Map.of(e.getKey(), e.getValue())).fingerprint(DIM, Metric.L2), e.getKey());
        }
    }

    // ---- run-time changes ----

    @Test
    void onlyTheSearchTimeKnobsChangeInAnOpenIndex() {
        TieredIndex t = tier(quiet());
        t.reconfigure(Map.of("graph.efSearch", "33", "splitBeam", "true"));
        assertEquals(33, t.config().graph().efSearch());
        assertTrue(t.config().splitBeam());
        t.reconfigure(t.config().toMap()); // nothing changes: every key is accepted with its own value
        TieredIndex.Config before = t.config();
        for (String[] kv : new String[][] {{"graph.M", "17"}, {"graph.sq8", "true"}, {"tailLimit", "5"}, {"unknown", "1"}, {"graph.efSearch", "x"}}) {
            Map<String, String> change = new LinkedHashMap<>();
            change.put("graph.efSearch", "44"); // a good change beside a bad one
            change.put(kv[0], kv[1]);
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> t.reconfigure(change), kv[0]);
            assertTrue(e.getMessage().contains(kv[0]), e.getMessage());
            assertEquals(before, t.config(), "all changes or none: " + kv[0]);
        }
    }

    // ---- version and capabilities ----

    @Test
    void aLargerMajorIsRefusedAndALargerMinorAccepted() {
        ContractVersion lib = TieredIndex.contractVersion();
        assertTrue(new ContractVersion(lib.major(), lib.minor()).accepts(lib));
        assertTrue(new ContractVersion(1, 0).accepts(new ContractVersion(1, 7)));
        assertTrue(new ContractVersion(2, 0).accepts(new ContractVersion(1, 0)));
        assertFalse(new ContractVersion(1, 9).accepts(new ContractVersion(2, 0)));
        assertThrows(IllegalArgumentException.class, () -> new ContractVersion(-1, 0));
        for (Capability c : List.of(Capability.ATOMIC_BATCH, Capability.DEFERRED_APPLY, Capability.SEARCH_CONTEXT, Capability.PERSISTENCE,
                Capability.SNAPSHOT_REBUILD)) {
            assertTrue(TieredIndex.capabilities().contains(c), c.name());
        }
        assertThrows(UnsupportedOperationException.class, () -> TieredIndex.capabilities().clear());
    }

    // ---- metrics ----

    @Test
    void theStatisticsAreAMapAndASinkReceivesTheIncrements() {
        TieredIndex t = tier(quiet());
        Map<String, AtomicLong> counted = new ConcurrentHashMap<>();
        Map<String, LongSupplier> gauges = new ConcurrentHashMap<>();
        t.bindMetrics(new MetricSink() {
            @Override
            public LongConsumer counter(String name) {
                AtomicLong a = counted.computeIfAbsent(name, n -> new AtomicLong());
                return a::addAndGet;
            }

            @Override
            public void gauge(String name, LongSupplier value) {
                gauges.put(name, value);
            }
        });
        float[][] data = vectors(60, DIM, 3);
        for (int i = 0; i < 30; i++) t.add(i, data[i]);
        t.sealNow();
        for (int i = 30; i < 60; i++) t.add(i, data[i]);
        t.sealNow();
        t.mergeNow();
        Map<String, Long> m = t.metrics();
        assertEquals(2L, m.get("tiervec.seals"));
        assertEquals(1L, m.get("tiervec.merges"));
        assertEquals(60L, m.get("tiervec.sealedVectors"));
        assertEquals(60L, m.get("tiervec.mergedVectors"));
        assertEquals(60L, m.get("tiervec.adds"));
        assertEquals(60L, m.get("tiervec.live"));
        assertEquals(1L, m.get("tiervec.segments"));
        assertEquals(1000L, m.get("tiervec.mergeWriteAmpThousandths"));
        assertEquals(2000L, m.get("tiervec.totalWriteAmpThousandths"));
        assertEquals(0L, m.get("tiervec.health"));
        for (Map.Entry<String, Long> e : m.entrySet()) {
            assertTrue(e.getKey().startsWith("tiervec."), e.getKey());
            boolean isCounter = counted.containsKey(e.getKey());
            assertTrue(isCounter != gauges.containsKey(e.getKey()), e.getKey() + " is a counter or a gauge");
            assertEquals(e.getValue(), isCounter ? counted.get(e.getKey()).get() : gauges.get(e.getKey()).getAsLong(), e.getKey());
        }
        assertThrows(UnsupportedOperationException.class, () -> m.put("x", 1L));
    }

    @Test
    void aSinkThatThrowsDoesNotDisturbTheIndex() {
        TieredIndex t = tier(quiet());
        t.bindMetrics(new MetricSink() {
            @Override
            public LongConsumer counter(String name) {
                return n -> {
                    throw new IllegalStateException("the host's metrics are broken");
                };
            }

            @Override
            public void gauge(String name, LongSupplier value) {}
        });
        float[][] data = vectors(20, DIM, 4);
        for (int i = 0; i < 20; i++) t.add(i, data[i]);
        t.sealNow();
        assertEquals(1, t.segmentCount());
        assertEquals(1L, t.metrics().get("tiervec.seals"));
    }

    // ---- health ----

    @Test
    void aSealThatKeepsFailingShowsAsDegradedUntilOneWorks() throws Exception {
        TieredIndex t = tier(quiet().withTail(20, 10));
        AtomicBoolean broken = new AtomicBoolean(true);
        t.setSealHooks(() -> {
            if (broken.get()) throw new IllegalStateException("a failing build");
        }, null);
        assertEquals(TieredIndex.Health.OK, t.health());
        float[][] data = vectors(20, DIM, 5);
        for (int i = 0; i < 20; i++) t.add(i, data[i]);
        await(() -> t.health() == TieredIndex.Health.DEGRADED, "three failed seals show as degraded");
        assertTrue(t.stats().failures() >= 3, "failures " + t.stats().failures());
        assertEquals("a failing build", t.lastError().getMessage());
        assertEquals(1L, t.metrics().get("tiervec.health"));
        assertEquals(1, t.search(data[3], 1, null).length, "searches go on");
        broken.set(false);
        await(() -> t.health() == TieredIndex.Health.OK && t.segmentCount() == 1, "a seal that works clears the state");
    }

    @Test
    void aDeadWorkerShowsAsFailed() throws Exception {
        TieredIndex t = tier(quiet());
        t.interruptWorker();
        await(() -> t.health() == TieredIndex.Health.FAILED, "the worker ended");
        assertEquals(2L, t.metrics().get("tiervec.health"));
    }

    // ---- after close ----

    @Test
    void callsAfterCloseThrowClosedException() {
        TieredIndex t = tier(quiet());
        float[][] data = vectors(20, DIM, 6);
        for (int i = 0; i < 20; i++) t.add(i, data[i]);
        t.sealNow();
        t.close();
        t.close();
        float[] v = data[0];
        assertThrows(ClosedException.class, () -> t.add(100, v));
        assertThrows(ClosedException.class, () -> t.delete(1));
        assertThrows(ClosedException.class, () -> t.apply(new TieredIndex.Batch(List.of(), List.of(1L))));
        assertThrows(ClosedException.class, () -> t.search(v, 1, null));
        assertThrows(ClosedException.class, () -> t.search(v, 1, null, TieredIndex.Freshness.SEALED_ONLY));
        assertThrows(ClosedException.class, () -> t.search(v, 1, null, 50));
        assertThrows(ClosedException.class, t::sealNow);
        assertThrows(ClosedException.class, t::mergeNow);
        assertThrows(ClosedException.class, () -> t.rebuildFrom(List.<TieredIndex.IdVector>of()));
        assertThrows(ClosedException.class, () -> t.rebuildFrom(() -> new TieredIndex.Snapshot(0, List.of()), 1L));
        assertThrows(ClosedException.class, () -> t.checkpoint(dir));
        assertThrows(ClosedException.class, () -> t.reconfigure(Map.of()));
        assertFalse(t.workerAlive());
        // the reports keep working
        assertEquals(20, t.size());
        assertEquals(1, t.stats().segments());
        assertEquals(20L, t.metrics().get("tiervec.live"));
        assertEquals(TieredIndex.Health.OK, t.health());
        assertTrue(t.config() != null);
    }

    @Test
    void aSearchThatRunsWhileTheIndexClosesFinishesWithACorrectAnswer() throws Exception {
        TieredIndex t = tier(quiet());
        float[][] data = vectors(50, DIM, 7);
        for (int i = 0; i < 50; i++) t.add(i, data[i]);
        t.sealNow();
        TieredIndex.Config c = t.config();
        Hit[] expected = t.search(data[7], 5, null);
        AtomicBoolean closed = new AtomicBoolean();
        Hit[] got = t.search(data[7], 5, id -> {
            if (closed.compareAndSet(false, true)) t.close(); // closed in the middle of the scan
            return true;
        });
        assertArrayEquals(expected, got);
        assertEquals(c, t.config());
        assertThrows(ClosedException.class, () -> t.search(data[7], 5, null));
    }
}
