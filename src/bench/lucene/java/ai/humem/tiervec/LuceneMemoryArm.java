package ai.humem.tiervec;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorScorer;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsReader;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsWriter;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsWriter;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.ReaderUtil;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.search.ConstantScoreScorer;
import org.apache.lucene.search.ConstantScoreWeight;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.RescoreTopNQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.SearcherFactory;
import org.apache.lucene.search.SearcherManager;
import org.apache.lucene.search.TermInSetQuery;
import org.apache.lucene.search.TwoPhaseIterator;
import org.apache.lucene.search.Weight;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.IORunnable;
import org.apache.lucene.util.Version;
import org.apache.lucene.util.hnsw.HnswGraphSearcher;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.apache.lucene.util.hnsw.UpdateableRandomVectorScorer;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.LongPredicate;

/**
 * The Lucene arm of the agent-memory benchmark (docs/AGENT-MEMORY-BENCHMARK.md, issue #160): an {@link IndexWriter} that updates and deletes
 * by id, and a near-real-time reader, through Lucene's public API; no Lucene code is copied. It is not compiled by Maven; it needs the jars of
 * the profile {@code bench} of the pom, and {@link MemoryArm} reaches it by reflection.
 *
 * <ul>
 * <li>A write is {@code updateDocument} or {@code deleteDocuments} by the term of the id. A batch is one {@code updateDocuments} call that
 * deletes the ids of the batch and adds its rows (a {@code deleteDocuments} call when it only deletes), so batches are atomic.
 * <li>A search sees the writes up to the last {@link #refresh}, which flushes the buffered rows into a segment and runs the merges that are
 * due on the calling thread ({@link SerialMergeScheduler}).
 * <li>{@code quant=none} is the HNSW format of the default codec. {@code quant=int8} is the HNSW format on 8-bit scalar quantized vectors; its
 * candidates are scored again with the float32 vectors ({@link RescoreTopNQuery}), so that a hit carries the exact score.
 * <li>The codec differs from the default one in two points only: its scorers count the distance computations, and a vector may have up to
 * {@value #MAX_DIMENSIONS} dimensions.
 * </ul>
 */
final class LuceneMemoryArm implements MemoryArm {
    private static final String ID = "id", NUMBER = "n", VECTOR = "v";
    private static final int MAX_DIMENSIONS = 4096;
    private static final List<String> SETTINGS = List.of("maxConn", "beamWidth", "efSearch", "quant");

    private final int maxConn, beamWidth, efSearch;
    private final boolean int8;
    private final boolean counting = DistanceCount.active();
    private final ByteBuffersDirectory directory = new ByteBuffersDirectory();
    private final IndexWriter writer;
    private final SearcherManager searchers;
    private final LongAdder flushes = new LongAdder(), merges = new LongAdder(), graphs = new LongAdder(), rowsFlushed = new LongAdder(),
            rowsMerged = new LongAdder();

    LuceneMemoryArm(int dim, Map<String, String> settings) {
        for (String key : settings.keySet()) {
            if (!SETTINGS.contains(key)) throw new IllegalArgumentException("the arm lucene has no setting '" + key + "' (" + SETTINGS + ")");
        }
        // maxConn is Lucene's default and the M of the graphs of tiervec; efConstruction 200 and efSearch 100 are those of tiervec
        maxConn = Integer.parseInt(settings.getOrDefault("maxConn", "16"));
        beamWidth = Integer.parseInt(settings.getOrDefault("beamWidth", "200"));
        efSearch = Integer.parseInt(settings.getOrDefault("efSearch", "100"));
        String quant = settings.getOrDefault("quant", "none");
        if (!quant.equals("none") && !quant.equals("int8")) throw new IllegalArgumentException("quant must be none or int8");
        int8 = quant.equals("int8");
        if (efSearch < 1 || dim > MAX_DIMENSIONS) throw new IllegalArgumentException("efSearch must be 1 or more, the dimension at most " + MAX_DIMENSIONS);
        IndexWriterConfig config = new IndexWriterConfig();
        config.setCodec(new CountingCodec(new HnswFormat()));
        config.setMergeScheduler(new SerialMergeScheduler());
        config.setCommitOnClose(false);
        try {
            writer = new IndexWriter(directory, config);
            searchers = new SearcherManager(writer, new SearcherFactory() {
                @Override public IndexSearcher newSearcher(IndexReader reader, IndexReader previous) {
                    IndexSearcher searcher = new IndexSearcher(reader);
                    searcher.setQueryCache(null); // a filter of this arm is never the same twice
                    return searcher;
                }
            });
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static Term term(long id) { return new Term(ID, Long.toString(id)); }

    private static Document document(long id, float[] v) {
        Document d = new Document();
        d.add(new StringField(ID, Long.toString(id), Field.Store.NO));
        d.add(new NumericDocValuesField(NUMBER, id));
        d.add(new KnnFloatVectorField(VECTOR, v.clone(), VectorSimilarityFunction.DOT_PRODUCT));
        return d;
    }

    @Override public void put(long id, float[] v) {
        try {
            writer.updateDocument(term(id), document(id, v));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override public void delete(long id) {
        try {
            writer.deleteDocuments(term(id));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override public boolean atomicBatches() { return true; }

    @Override public void applyBatch(List<Runbook.Op> writes) {
        List<BytesRef> ids = new ArrayList<>();
        List<Document> rows = new ArrayList<>();
        for (Runbook.Op w : writes) {
            ids.add(new BytesRef(Long.toString(w.id())));
            if (w.kind() != Runbook.Kind.DELETE) rows.add(document(w.id(), w.vec()));
        }
        Query old = new TermInSetQuery(ID, ids);
        try {
            if (rows.isEmpty()) writer.deleteDocuments(old);
            else writer.updateDocuments(old, rows);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override public void refresh() {
        try {
            searchers.maybeRefreshBlocking();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override public Hit[] search(float[] q, int k, LongPredicate allow) {
        try {
            IndexSearcher searcher = searchers.acquire();
            try {
                // Lucene's k is also the width of its search, so it asks for efSearch candidates and keeps the first k
                Query filter = allow == null ? null : new Allowed(allow);
                Query found = searcher.rewrite(new KnnFloatVectorQuery(VECTOR, q, Math.max(efSearch, k), filter));
                // RescoreTopNQuery cannot be rewritten over a query that matches nothing
                if (found instanceof MatchNoDocsQuery) return new Hit[0];
                if (int8) found = RescoreTopNQuery.createFullPrecisionRescorerQuery(found, q, VECTOR, k);
                ScoreDoc[] docs = searcher.search(found, k).scoreDocs;
                List<LeafReaderContext> leaves = searcher.getIndexReader().leaves();
                Hit[] hits = new Hit[docs.length];
                for (int i = 0; i < docs.length; i++) {
                    LeafReaderContext leaf = leaves.get(ReaderUtil.subIndex(docs[i].doc, leaves));
                    NumericDocValues number = leaf.reader().getNumericDocValues(NUMBER);
                    if (!number.advanceExact(docs[i].doc - leaf.docBase)) throw new IllegalStateException("a hit without an id");
                    // Lucene's score of DOT_PRODUCT is (1 + dot) / 2, and a hit carries the dot product, as a hit of tiervec does
                    hits[i] = new Hit(number.longValue(), 2f * docs[i].score - 1f);
                }
                return hits;
            } finally {
                searchers.release(searcher);
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** what {@link RescoreTopNQuery} does over a query that matches nothing, which {@link #search} guards against; for the check of this arm */
    String rescoreOfNothing(float[] q) {
        try {
            IndexSearcher searcher = searchers.acquire();
            try {
                Query unguarded = RescoreTopNQuery.createFullPrecisionRescorerQuery(new MatchNoDocsQuery(), q, VECTOR, 1);
                return "returned " + searcher.search(unguarded, 1).scoreDocs.length + " hits";
            } catch (RuntimeException | AssertionError e) {
                return "threw " + e.getClass().getName() + ": " + e.getMessage();
            } finally {
                searchers.release(searcher);
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** the vectors of the segments that a flush or a merge gave a graph; Lucene gives a small segment none */
    @Override public long rowsBuilt() { return rowsFlushed.sum() + rowsMerged.sum(); }

    @Override public long rowsRebuilt() { return rowsMerged.sum(); }

    /** the bytes of the files of the directory, which is in memory, and of the rows that the writer buffers */
    @Override public long bytesHeld() {
        try {
            long held = writer.ramBytesUsed();
            for (String file : directory.listAll()) held += directory.fileLength(file);
            return held;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override public String about() {
        String jar = IndexWriter.class.getProtectionDomain().getCodeSource().getLocation().getPath();
        return "library=" + jar.substring(jar.lastIndexOf('/') + 1) + " version=" + Version.LATEST
                + " scorer=" + FlatVectorScorerUtil.getLucene99FlatVectorsScorer().getClass().getSimpleName() + " similarity=DOT_PRODUCT quant="
                + (int8 ? "int8" : "none") + " maxConn=" + maxConn + " beamWidth=" + beamWidth + " efSearch=" + efSearch
                + " tinySegmentsThreshold=" + Lucene99HnswVectorsFormat.HNSW_GRAPH_THRESHOLD
                + " directory=ByteBuffersDirectory mergeScheduler=SerialMergeScheduler mergePolicy=" + writer.getConfig().getMergePolicy().getClass().getSimpleName();
    }

    @Override public Map<String, Long> counters() {
        Map<String, Long> m = new LinkedHashMap<>();
        m.put("flushes", flushes.sum());
        m.put("merges", merges.sum());
        m.put("graphs", graphs.sum());
        return m;
    }

    @Override public void close() {
        try {
            searchers.close();
            writer.close();
            directory.close();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** the live documents whose id the predicate allows */
    private static final class Allowed extends Query {
        private final LongPredicate allow;

        Allowed(LongPredicate allow) { this.allow = allow; }

        @Override public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) {
            return new ConstantScoreWeight(this, boost) {
                @Override public ScorerSupplier scorerSupplier(LeafReaderContext context) {
                    float score = score();
                    return new ScorerSupplier() {
                        @Override public Scorer get(long leadCost) throws IOException {
                            NumericDocValues number = context.reader().getNumericDocValues(NUMBER);
                            DocIdSetIterator all = number == null ? DocIdSetIterator.empty() : DocIdSetIterator.all(context.reader().maxDoc());
                            return new ConstantScoreScorer(score, scoreMode, new TwoPhaseIterator(all) {
                                @Override public boolean matches() throws IOException {
                                    return number.advanceExact(approximation.docID()) && allow.test(number.longValue());
                                }

                                @Override public float matchCost() { return 10; }
                            });
                        }

                        @Override public long cost() { return context.reader().maxDoc(); }
                    };
                }

                @Override public boolean isCacheable(LeafReaderContext context) { return false; }
            };
        }

        @Override public void visit(QueryVisitor visitor) { visitor.visitLeaf(this); }
        @Override public String toString(String field) { return "Allowed"; }
        @Override public boolean equals(Object other) { return this == other; }
        @Override public int hashCode() { return System.identityHashCode(this); }
    }

    // ---- the codec: Lucene's own writers and readers, with scorers that count ----

    /**
     * The default codec with the vector format of this arm. It has a name of its own and no service entry, so a lookup of the codec by its
     * name fails loudly instead of reading with scorers that do not count.
     */
    private static final class CountingCodec extends FilterCodec {
        private final KnnVectorsFormat vectors;

        CountingCodec(KnnVectorsFormat vectors) {
            super("TiervecBenchCodec", Codec.getDefault());
            this.vectors = vectors;
        }

        @Override public KnnVectorsFormat knnVectorsFormat() { return vectors; }
    }

    /** the HNSW format, put together as {@link Lucene99HnswVectorsFormat} and the scalar quantized format of Lucene put it together */
    private final class HnswFormat extends KnnVectorsFormat {
        private final FlatVectorsFormat flat;

        HnswFormat() {
            super("TiervecBenchHnsw");
            FlatVectorsScorer plain = FlatVectorScorerUtil.getLucene99FlatVectorsScorer();
            FlatVectorsFormat float32 = new Lucene99FlatVectorsFormat(counting ? new CountedFloat(plain) : plain);
            Lucene104ScalarQuantizedVectorScorer quantized = counting ? new CountedQuantized(plain) : new Lucene104ScalarQuantizedVectorScorer(plain);
            flat = int8 ? new QuantizedFormat(float32, quantized) : float32;
        }

        @Override public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
            return new Rows(new Lucene99HnswVectorsWriter(state, maxConn, beamWidth, flat, flat.fieldsWriter(state), 1, null,
                    Lucene99HnswVectorsFormat.HNSW_GRAPH_THRESHOLD));
        }

        @Override public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
            return new Lucene99HnswVectorsReader(state, flat.fieldsReader(state));
        }

        @Override public int getMaxDimensions(String field) { return MAX_DIMENSIONS; }
    }

    /** the 8-bit flat format: the quantized codes beside the float32 vectors, as Lucene's own format of that encoding stores them */
    private static final class QuantizedFormat extends FlatVectorsFormat {
        private final FlatVectorsFormat raw;
        private final Lucene104ScalarQuantizedVectorScorer scorer;

        QuantizedFormat(FlatVectorsFormat raw, Lucene104ScalarQuantizedVectorScorer scorer) {
            super("TiervecBenchScalarQuantized");
            this.raw = raw;
            this.scorer = scorer;
        }

        @Override public FlatVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
            return new Lucene104ScalarQuantizedVectorsWriter(state, ScalarEncoding.UNSIGNED_BYTE, raw.fieldsWriter(state), scorer);
        }

        @Override public FlatVectorsReader fieldsReader(SegmentReadState state) throws IOException {
            return new Lucene104ScalarQuantizedVectorsReader(state, raw.fieldsReader(state), scorer);
        }

        @Override public int getMaxDimensions(String field) { return MAX_DIMENSIONS; }
    }

    /** counts the segments and rows that a flush or a merge writes with a graph */
    private final class Rows extends KnnVectorsWriter {
        private final KnnVectorsWriter in;

        Rows(KnnVectorsWriter in) { this.in = in; }

        @Override public KnnFieldVectorsWriter<?> addField(FieldInfo field) throws IOException { return in.addField(field); }

        @Override public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
            in.flush(maxDoc, sortMap);
            flushes.increment();
            count(maxDoc, rowsFlushed);
        }

        @Override public IORunnable mergeOneField(FieldInfo field, MergeState state) throws IOException {
            merges.increment();
            count(state.segmentInfo.maxDoc(), rowsMerged);
            return in.mergeOneField(field, state);
        }

        /** every document of this arm has one vector; the rule of Lucene99HnswVectorsWriter for a segment that gets a graph */
        private void count(int rows, LongAdder into) {
            int visited = HnswGraphSearcher.expectedVisitedNodes(Lucene99HnswVectorsFormat.HNSW_GRAPH_THRESHOLD, rows);
            if (rows > visited && visited > 0) {
                graphs.increment();
                into.add(rows);
            }
        }

        @Override public void finish() throws IOException { in.finish(); }
        @Override public long ramBytesUsed() { return in.ramBytesUsed(); }
        @Override public void close() throws IOException { in.close(); }
    }

    // ---- the count of the distance computations: Lucene's own scorers, each score counted on the thread that computes it ----

    private static void count(boolean quantized, int n) {
        if (quantized) DistanceCount.quantized(n);
        else DistanceCount.full(n);
    }

    private static final class CountedFloat implements FlatVectorsScorer {
        private final FlatVectorsScorer in;

        CountedFloat(FlatVectorsScorer in) { this.in = in; }

        @Override public RandomVectorScorerSupplier getRandomVectorScorerSupplier(VectorSimilarityFunction f, KnnVectorValues values) throws IOException {
            return counted(in.getRandomVectorScorerSupplier(f, values), false);
        }

        @Override public RandomVectorScorer getRandomVectorScorer(VectorSimilarityFunction f, KnnVectorValues values, float[] target) throws IOException {
            return new Counted(in.getRandomVectorScorer(f, values, target), false);
        }

        @Override public RandomVectorScorer getRandomVectorScorer(VectorSimilarityFunction f, KnnVectorValues values, byte[] target) throws IOException {
            return new Counted(in.getRandomVectorScorer(f, values, target), false);
        }

        @Override public String toString() { return "CountedFloat(" + in + ")"; }
    }

    /** the scorer of the quantized format; its delegate for float32 values is not counted, so that no score is counted twice */
    private static final class CountedQuantized extends Lucene104ScalarQuantizedVectorScorer {
        CountedQuantized(FlatVectorsScorer plain) { super(plain); }

        @Override public RandomVectorScorerSupplier getRandomVectorScorerSupplier(VectorSimilarityFunction f, KnnVectorValues values) throws IOException {
            return counted(super.getRandomVectorScorerSupplier(f, values), values instanceof QuantizedByteVectorValues);
        }

        @Override public RandomVectorScorer getRandomVectorScorer(VectorSimilarityFunction f, KnnVectorValues values, float[] target) throws IOException {
            return new Counted(super.getRandomVectorScorer(f, values, target), values instanceof QuantizedByteVectorValues);
        }

        @Override public RandomVectorScorer getRandomVectorScorer(VectorSimilarityFunction f, KnnVectorValues values, byte[] target) throws IOException {
            return new Counted(super.getRandomVectorScorer(f, values, target), false);
        }
    }

    private static RandomVectorScorerSupplier counted(RandomVectorScorerSupplier in, boolean quantized) {
        return new RandomVectorScorerSupplier() {
            @Override public UpdateableRandomVectorScorer scorer() throws IOException { return new CountedUpdateable(in.scorer(), quantized); }
            @Override public RandomVectorScorerSupplier copy() throws IOException { return counted(in.copy(), quantized); }
        };
    }

    private static class Counted implements RandomVectorScorer {
        private final RandomVectorScorer in;
        private final boolean quantized;

        Counted(RandomVectorScorer in, boolean quantized) {
            this.in = in;
            this.quantized = quantized;
        }

        @Override public float score(int node) throws IOException {
            count(quantized, 1);
            return in.score(node);
        }

        @Override public float bulkScore(int[] nodes, float[] scores, int n) throws IOException {
            count(quantized, n);
            return in.bulkScore(nodes, scores, n);
        }

        @Override public int maxOrd() { return in.maxOrd(); }
        @Override public int ordToDoc(int ord) { return in.ordToDoc(ord); }
        @Override public Bits getAcceptOrds(Bits docs) { return in.getAcceptOrds(docs); }
    }

    private static final class CountedUpdateable extends Counted implements UpdateableRandomVectorScorer {
        private final UpdateableRandomVectorScorer in;

        CountedUpdateable(UpdateableRandomVectorScorer in, boolean quantized) {
            super(in, quantized);
            this.in = in;
        }

        @Override public void setScoringOrdinal(int node) throws IOException { in.setScoringOrdinal(node); }
    }
}
