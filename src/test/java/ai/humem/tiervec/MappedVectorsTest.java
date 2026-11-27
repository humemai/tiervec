package ai.humem.tiervec;

import static ai.humem.tiervec.SegmentTestSupport.graph;
import static ai.humem.tiervec.SegmentTestSupport.idOf;
import static ai.humem.tiervec.SegmentTestSupport.params;
import static ai.humem.tiervec.SegmentTestSupport.vectors;
import static ai.humem.tiervec.SegmentTestSupport.vectorsOf;
import static ai.humem.tiervec.SegmentTestSupport.versOf;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.humem.tiervec.SegmentFormat.CorruptException;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.function.LongPredicate;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.CleanupMode;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * The mapped reader of the vector section (#27): rows over one and over several mappings, a scratch array per thread, a tier that opens
 * mapped and goes on working, a refused file, and a segment file that is retired while it is mapped. The round-trip and corruption tests of
 * the heap reader run with the mapped one in the {@code Mapped...Test} subclasses. This class runs on every platform; it leaves its
 * temporary directory behind, because a file can still be mapped when a test ends, and Windows can refuse to delete it then.
 */
class MappedVectorsTest {
    private static final TieredIndex.VectorStorage MAPPED = TieredIndex.VectorStorage.MAPPED;
    private static final int DIM = TierReopenTest.DIM;

    @TempDir(cleanup = CleanupMode.NEVER)
    Path tmp;

    // the last column is the most bytes of one mapping: below one row, a few rows, and the real limit
    @ParameterizedTest(name = "{0} sq8={1} dim={2} piece={3}")
    @CsvSource({"L2,false,24,1", "COSINE,false,24,200", "DOT,false,24,4096", "L2,true,24,1000", "COSINE,true,24,2147483647", "L2,false,512,100000"})
    void mappedRowsBehaveLikeTheHeapArray(Metric metric, boolean sq8, int dim, int piece) throws IOException {
        final int n = 600;
        SealedGraph heap = graph(vectors(n, dim, 31), metric, params(sq8));
        Path f = tmp.resolve("a.tvs");
        SegmentWriter.write(heap, versOf(n), f);
        SealedGraph mapped = SegmentReader.readFull(f, true, MAPPED, piece).graph();
        SealedVectors.Mapped v = assertInstanceOf(SealedVectors.Mapped.class, mapped.vectors);
        // a power of two of whole rows in each mapping, one row at least
        final int perPiece = Integer.highestOneBit(Math.max(1, piece / (4 * dim)));
        assertEquals((n + perPiece - 1) / perPiece, v.pieces());
        assertEquals(heap.bytesHeld() - 4L * n * dim, mapped.bytesHeld());

        float[] all = vectorsOf(heap);
        assertEquals(all.length, v.length());
        assertArrayEquals(all, vectorsOf(mapped));
        assertArrayEquals(heap.rawVector(n - 1), mapped.rawVector(n - 1));
        // a range that starts and ends inside a row, over as many mappings as there are
        final int from = 3 * dim + 5, count = all.length - from - 7;
        FloatBuffer out = FloatBuffer.allocate(count);
        v.put(out, from, count);
        assertArrayEquals(Arrays.copyOfRange(all, from, from + count), out.array());

        LongPredicate few = id -> id % 7 == 0;
        for (float[] q : vectors(40, dim, 32)) {
            assertArrayEquals(heap.search(q, 10, null), mapped.search(q, 10, null));
            assertArrayEquals(heap.search(q, 10, few), mapped.search(q, 10, few));
            assertArrayEquals(heap.scan(q, 10, few, null), mapped.scan(q, 10, few, null));
        }

        // written again from the mapping, the file is the same byte for byte, and a merge reads the same rows into a graph
        Path again = tmp.resolve("b.tvs");
        SegmentWriter.write(mapped, versOf(n), again);
        assertArrayEquals(Files.readAllBytes(f), Files.readAllBytes(again));
        assertArrayEquals(all, vectorsOf(GraphIndex.fromSealed(mapped).seal()));
    }

    @Test
    void searchingThreadsDoNotShareAScratchArray() throws Exception {
        final int n = 1500, dim = 32, threads = 8;
        SealedGraph heap = graph(vectors(n, dim, 41), Metric.COSINE, params(false));
        Path f = tmp.resolve("threads.tvs");
        SegmentWriter.write(heap, versOf(n), f);
        SealedGraph mapped = SegmentReader.readFull(f, true, MAPPED, 4096).graph();
        float[][] qs = vectors(64, dim, 42);
        Hit[][] want = new Hit[qs.length][];
        for (int i = 0; i < qs.length; i++) want[i] = heap.search(qs[i], 10, null);

        ExecutorService pool = Executors.newFixedThreadPool(threads);
        try {
            List<Future<?>> done = new ArrayList<>();
            for (int t = 0; t < threads; t++) {
                final int first = 7 * t;
                done.add(pool.submit(() -> {
                    for (int r = 0; r < 20 * qs.length; r++) {
                        int i = (first + r) % qs.length;
                        assertArrayEquals(want[i], mapped.search(qs[i], 10, null), "query " + i);
                    }
                }));
            }
            for (Future<?> d : done) d.get(120, TimeUnit.SECONDS);
        } finally {
            pool.shutdownNow();
        }
    }

    /** a checkpoint of a new tier with one segment per 800 rows */
    private static void checkpointSegments(Path dir, float[][] data) throws IOException {
        try (TieredIndex t = new TieredIndex(DIM, Metric.L2, TierReopenTest.cfg())) {
            for (int i = 0; i < data.length; i++) {
                t.add(idOf(i), data[i]);
                if (i % 800 == 799) t.sealNow();
            }
            assertTrue(t.awaitIdle(60_000));
            assertEquals(data.length / 800, t.segmentCount());
            t.checkpoint(dir);
        }
    }

    private static void assertSameAnswers(TieredIndex a, TieredIndex b, float[][] qs) {
        assertEquals(a.size(), b.size());
        assertArrayEquals(a.segmentSizes(), b.segmentSizes());
        for (float[] q : qs) assertArrayEquals(a.search(q, 10, null), b.search(q, 10, null));
    }

    @Test
    void aTierOpenedMappedAnswersLikeOneOnTheHeapAndGoesOnWorking() throws IOException {
        float[][] data = vectors(2400, DIM, 51);
        float[][] more = vectors(400, DIM, 53);
        float[][] qs = vectors(50, DIM, 52);
        Path forHeap = tmp.resolve("heap"), forMapped = tmp.resolve("mapped"), copy = tmp.resolve("copy");
        checkpointSegments(forHeap, data);
        checkpointSegments(forMapped, data);
        TieredIndex.Config mappedCfg = TierReopenTest.cfg().withVectorStorage(MAPPED);
        try (TieredIndex heap = TieredIndex.open(forHeap, TierReopenTest.cfg()); TieredIndex mapped = TieredIndex.open(forMapped, mappedCfg)) {
            assertEquals(MAPPED, mapped.config().vectorStorage());
            assertThrows(IllegalArgumentException.class, () -> mapped.reconfigure(Map.of("vectorStorage", "HEAP")));
            assertEquals(heap.bytesHeld() - 4L * data.length * DIM, mapped.bytesHeld());
            assertSameAnswers(heap, mapped, qs);

            // the segments of a mapped tier are written into another directory from their mappings
            mapped.checkpoint(copy);
            try (TieredIndex read = TieredIndex.open(copy, TierReopenTest.cfg())) {
                assertSameAnswers(heap, read, qs);
            }

            // deletes, replaced ids and new rows; fewer hidden rows in a segment than starts a compaction
            for (TieredIndex t : List.of(heap, mapped)) {
                for (int i = 0; i < 100; i++) assertTrue(t.delete(idOf(i)));
                for (int i = 0; i < 50; i++) t.add(idOf(100 + i), more[i]);
                for (int i = 50; i < more.length; i++) t.add(idOf(data.length + i), more[i]);
                t.sealNow();
                assertTrue(t.awaitIdle(60_000));
            }
            assertSameAnswers(heap, mapped, qs);

            // a merge reads the mapped rows and builds on the heap
            heap.mergeNow();
            mapped.mergeNow();
            assertTrue(heap.awaitIdle(60_000) && mapped.awaitIdle(60_000));
            assertSameAnswers(heap, mapped, qs);
            assertEquals(heap.bytesHeld(), mapped.bytesHeld(), "no mapped segment is left after the merge");

            mapped.checkpoint(copy);
            try (TieredIndex read = TieredIndex.open(copy, mappedCfg)) {
                assertSameAnswers(heap, read, qs);
            }
        }
    }

    @ParameterizedTest(name = "section id {0}")
    @ValueSource(ints = {4, 5, 7}) // VEC, and two sections that are read after it
    void aRefusedFileIsNotLeftMapped(int id) throws IOException {
        SealedGraph g = graph(vectors(300, 8, 5), Metric.L2, params(false));
        Path f = tmp.resolve("damaged.tvs");
        SegmentInfo info = SegmentWriter.write(g, versOf(300), f);
        byte[] raw = Files.readAllBytes(f);
        raw[(int) (info.section(id).offset() + 11)] ^= 0x20;
        Files.write(f, raw);
        CorruptException e = assertThrows(CorruptException.class, () -> SegmentReader.readFull(f, true, MAPPED));
        assertEquals(SegmentFormat.sectionName(id), e.section());
        Files.delete(f); // nothing was mapped, so no platform has a reason to refuse this
    }

    /** opens the directory mapped, merges its two segments into one, and publishes that, which retires the files of the two */
    private static void mergeAndCheckpointMapped(Path dir) throws IOException {
        try (TieredIndex u = TieredIndex.open(dir, TierReopenTest.cfg().withVectorStorage(MAPPED))) {
            u.mergeNow();
            u.checkpoint(dir);
            assertEquals(1, Manifest.readLatest(dir).entries().size());
        }
    }

    @Test
    void aRetiredFileThatIsStillMappedDoesNotFailACheckpoint() throws Exception {
        Path dir = tmp.resolve("retired");
        checkpointSegments(dir, vectors(1600, DIM, 61));
        Path first = dir.resolve(Manifest.segName(1));
        SealedGraph held = SegmentReader.readFull(first, true, MAPPED).graph(); // the file stays mapped while this graph is reachable
        float[] row = held.rawVector(0);

        mergeAndCheckpointMapped(dir);
        boolean left = Files.exists(first);
        System.out.println("MAPPED-DELETE os=" + System.getProperty("os.name") + " retiredFileLeftWhileMapped=" + left);
        if (!SegmentFormat.WINDOWS) assertFalse(left, "only Windows refuses to delete a mapped file");
        assertArrayEquals(row, held.rawVector(0), "the mapping serves the same bytes after the checkpoint");

        // once the garbage collector has released the mappings, a later open removes what was left
        held = null;
        for (int i = 0; i < 200 && Files.exists(first); i++) {
            System.gc();
            Thread.sleep(50);
            TieredIndex.open(dir, TierReopenTest.cfg()).close();
        }
        assertFalse(Files.exists(first), "the retired file is gone after its mappings were released");
    }
}
