package ai.humem.tiervec;

import io.github.jbellis.jvector.graph.GraphIndexBuilder;
import io.github.jbellis.jvector.graph.GraphSearcher;
import io.github.jbellis.jvector.graph.ImmutableGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.SearchResult;
import io.github.jbellis.jvector.graph.similarity.BuildScoreProvider;
import io.github.jbellis.jvector.graph.similarity.DefaultSearchScoreProvider;
import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.graph.similarity.SearchScoreProvider;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.LongPredicate;

/**
 * The JVector arm of the agent-memory benchmark (docs/AGENT-MEMORY-BENCHMARK.md, issue #160): one in-memory graph that is updated in place,
 * through JVector's public builder and searcher, and no JVector code is copied. It is not compiled by Maven; it needs the jars of the profile
 * {@code bench} of the pom, and {@link MemoryArm} reaches it by reflection.
 *
 * <ul>
 * <li>A write adds a node under a new ordinal with {@code addGraphNode}; an update marks the old ordinal with {@code markNodeDeleted} first, so
 * two versions of an id are never visible together, and between the two calls the id is not visible. A delete is {@code markNodeDeleted}.
 * <li>{@link #maintain} calls {@code removeDeletedNodes} when the marked nodes are more than {@code removeAt} times the live ones, the schedule
 * of the live graph arm. JVector says that the call is not thread-safe against other modifications, so writers wait while it runs, and the arm
 * counts the writes that waited and the time they waited. It does not call {@code cleanup()}, which ends the snapshot views of later searches.
 * <li>Searches run beside writes on JVector's own snapshot view, taken again for every search. They wait only for the short step after a removal
 * that drops the vectors of the removed nodes.
 * <li>There are no atomic batches.
 * <li>A search whose filter says that it allows at most {@code efSearch} ids scores those ids exactly with JVector's similarity function, over
 * the nodes that the view of the graph holds and reports as live. JVector has no exact path of its own and leaves a small allowed set to the
 * caller.
 * </ul>
 *
 * <p>Writes of one id must not run concurrently with each other, as in a host that orders the writes of a row.
 */
final class JVectorMemoryArm implements MemoryArm {
    private static final VectorTypeSupport VTS = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final VectorSimilarityFunction DOT = VectorSimilarityFunction.DOT_PRODUCT;
    private static final List<String> SETTINGS = List.of("degree", "upperDegree", "efConstruction", "efSearch", "alpha", "neighborOverflow", "removeAt");

    private final int degree, upperDegree, efConstruction, efSearch;
    private final float alpha, overflow;
    private final double removeAt;
    private final Store store;
    private final GraphIndexBuilder builder;
    private final ImmutableGraphIndex graph;
    /** one worker, so that a removal does its work in one order and a replay of one runbook gives the same graph */
    private final ForkJoinPool pool = new ForkJoinPool(1);
    private final ConcurrentHashMap<Long, Integer> ordinalOf = new ConcurrentHashMap<>();
    private final ConcurrentLinkedQueue<Integer> marked = new ConcurrentLinkedQueue<>();
    private final AtomicInteger markedCount = new AtomicInteger(), nextOrdinal = new AtomicInteger();
    /** a write holds the read side, a removal the write side */
    private final ReentrantReadWriteLock writes = new ReentrantReadWriteLock(true);
    /** a search holds the read side; the vectors of removed nodes are dropped under the write side */
    private final ReentrantReadWriteLock vectors = new ReentrantReadWriteLock(true);
    private final LongAdder added = new LongAdder(), removals = new LongAdder(), removed = new LongAdder(), writerWaits = new LongAdder(),
            writerWaitNanos = new LongAdder();
    /** distance computations that are not yet given to {@link DistanceCount}; the work of a removal runs on the thread of the pool */
    private final LongAdder distances = new LongAdder();
    private final boolean counting = DistanceCount.active();
    private final ConcurrentLinkedQueue<GraphSearcher> opened = new ConcurrentLinkedQueue<>();
    private final ThreadLocal<GraphSearcher> searchers;

    JVectorMemoryArm(int dim, Map<String, String> settings) {
        for (String key : settings.keySet()) {
            if (!SETTINGS.contains(key)) throw new IllegalArgumentException("the arm jvector has no setting '" + key + "' (" + SETTINGS + ")");
        }
        // the build and search settings of the graphs of tiervec where they mean the same (M 16: 32 links on the base layer and 16 above,
        // efConstruction 200, efSearch 100), and the values of JVector's tuning page for the two settings of its pruning
        degree = Integer.parseInt(settings.getOrDefault("degree", "32"));
        upperDegree = Integer.parseInt(settings.getOrDefault("upperDegree", "16"));
        efConstruction = Integer.parseInt(settings.getOrDefault("efConstruction", "200"));
        efSearch = Integer.parseInt(settings.getOrDefault("efSearch", "100"));
        alpha = Float.parseFloat(settings.getOrDefault("alpha", "1.2"));
        overflow = Float.parseFloat(settings.getOrDefault("neighborOverflow", "1.2"));
        removeAt = Double.parseDouble(settings.getOrDefault("removeAt", "0.2"));
        if (efSearch < 1 || !(removeAt >= 0)) throw new IllegalArgumentException("efSearch must be 1 or more, removeAt 0 or more");
        store = new Store(dim);
        BuildScoreProvider scores = BuildScoreProvider.randomAccessScoreProvider(store, DOT);
        builder = GraphIndexBuilder.builder(counting ? counted(scores) : scores, dim, List.of(degree, upperDegree)).withBeamWidth(efConstruction)
                .withNeighborOverflow(overflow).withAlpha(alpha).withSimdExecutor(pool).withParallelExecutor(pool).build();
        graph = builder.getGraph();
        searchers = ThreadLocal.withInitial(() -> {
            GraphSearcher s = new GraphSearcher(graph);
            opened.add(s);
            return s;
        });
    }

    @Override public void put(long id, float[] v) {
        enterWrite();
        try {
            mark(ordinalOf.remove(id));
            int ordinal = nextOrdinal.getAndIncrement();
            VectorFloat<?> vector = VTS.createFloatVector(v.clone());
            store.set(ordinal, id, vector);
            builder.addGraphNode(ordinal, vector);
            ordinalOf.put(id, ordinal);
            added.increment();
        } finally {
            writes.readLock().unlock();
            give();
        }
    }

    @Override public void delete(long id) {
        enterWrite();
        try {
            mark(ordinalOf.remove(id));
        } finally {
            writes.readLock().unlock();
        }
    }

    /** a write enters; when a removal runs, or waits to run, the wait is counted */
    private void enterWrite() {
        Lock lock = writes.readLock();
        try {
            if (lock.tryLock(0, TimeUnit.NANOSECONDS)) return;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted before a write", e);
        }
        writerWaits.increment();
        long t0 = System.nanoTime();
        lock.lock();
        writerWaitNanos.add(System.nanoTime() - t0);
    }

    /** the CI smoke showed unfiltered searches of small collections that return fewer ids than there are live rows (issue #160) */
    @Override public boolean fullAnswers() { return false; }

    /** the base layer as a search sees it, walked from the entry node; for the check of this arm */
    String reachability() {
        ImmutableGraphIndex.View view = graph.getView();
        var entry = view.entryNode();
        if (entry == null) return "no entry node";
        Bits live = view.liveNodes();
        Set<Integer> seen = new HashSet<>();
        Deque<Integer> next = new ArrayDeque<>();
        seen.add(entry.node);
        next.add(entry.node);
        int entryDegree = -1;
        while (!next.isEmpty()) {
            var neighbors = view.getNeighborsIterator(0, next.poll());
            int degree = 0;
            while (neighbors.hasNext()) {
                int n = neighbors.nextInt();
                degree++;
                if (seen.add(n)) next.add(n);
            }
            if (entryDegree < 0) entryDegree = degree;
        }
        return "entry=" + entry.node + " entryLevel=" + entry.level + " entryMarked=" + !live.get(entry.node) + " entryBaseDegree=" + entryDegree
                + " liveReached=" + seen.stream().filter(live::get).count() + " live=" + ordinalOf.size() + " marked=" + markedCount.get();
    }

    private void mark(Integer ordinal) {
        if (ordinal == null) return;
        builder.markNodeDeleted(ordinal);
        marked.add(ordinal);
        markedCount.incrementAndGet();
    }

    @Override public void maintain() {
        int dead = markedCount.get();
        if (dead > 0 && dead > removeAt * ordinalOf.size()) remove();
    }

    /** {@code removeDeletedNodes} with no write beside it, then the vectors of the removed nodes are dropped with no search beside it */
    private void remove() {
        writes.writeLock().lock();
        try {
            List<Integer> gone = new ArrayList<>();
            for (Integer o = marked.poll(); o != null; o = marked.poll()) gone.add(o);
            if (gone.isEmpty()) return;
            markedCount.addAndGet(-gone.size());
            builder.removeDeletedNodes();
            vectors.writeLock().lock();
            try {
                for (int o : gone) store.drop(o);
            } finally {
                vectors.writeLock().unlock();
            }
            removals.increment();
            removed.add(gone.size());
        } finally {
            writes.writeLock().unlock();
            give();
        }
    }

    @Override public Hit[] search(float[] q, int k, LongPredicate allow) {
        VectorFloat<?> query = VTS.createFloatVector(q);
        vectors.readLock().lock();
        try {
            if (FilterPlan.smallByHint(allow, Math.max(efSearch, k))) return exact(query, k, allow);
            GraphSearcher searcher = searchers.get();
            searcher.setView(graph.getView());
            Bits bits = allow == null ? Bits.ALL : node -> allow.test(store.idOf(node));
            SearchScoreProvider scores = DefaultSearchScoreProvider.exact(query, DOT, store);
            SearchResult found = searcher.search(counting ? counted(scores) : scores, k, Math.max(efSearch, k), Float.NEGATIVE_INFINITY, 0.0f, bits);
            SearchResult.NodeScore[] nodes = found.getNodes();
            Hit[] hits = new Hit[nodes.length];
            for (int i = 0; i < nodes.length; i++) hits[i] = new Hit(store.idOf(nodes[i].node), dot(nodes[i].score));
            return hits;
        } finally {
            vectors.readLock().unlock();
            give();
        }
    }

    /** JVector's score of {@code DOT_PRODUCT} is (1 + dot) / 2, and a hit carries the dot product, as a hit of tiervec does */
    private static float dot(float score) { return 2f * score - 1f; }

    private Hit[] exact(VectorFloat<?> query, int k, LongPredicate allow) {
        ImmutableGraphIndex.View view = graph.getView();
        Bits live = view.liveNodes();
        List<Hit> all = new ArrayList<>();
        for (Map.Entry<Long, Integer> e : ordinalOf.entrySet()) {
            int o = e.getValue();
            if (!allow.test(e.getKey()) || !view.contains(0, o) || !live.get(o)) continue;
            all.add(new Hit(e.getKey(), dot(DOT.compare(query, store.getVector(o)))));
        }
        all.sort(Comparator.comparingDouble((Hit h) -> h.score()).reversed().thenComparingLong(Hit::id));
        return all.subList(0, Math.min(k, all.size())).toArray(Hit[]::new);
    }

    /** hands the counted distances to the counter of the calling thread, which the replay reads */
    private void give() {
        if (!counting) return;
        long n = distances.sumThenReset();
        if (n > 0) DistanceCount.full(Math.toIntExact(n));
    }

    @Override public long rowsBuilt() { return added.sum(); }

    /** JVector's estimate of the graph, the float32 vectors that are held (removed ones are dropped), and 12 bytes for each ordinal of the arm */
    @Override public long bytesHeld() {
        long held = nextOrdinal.get() - removed.sum();
        return graph.ramBytesUsed() + held * (4L * store.dimension() + 16) + 12L * nextOrdinal.get();
    }

    @Override public String about() {
        String jar = GraphIndexBuilder.class.getProtectionDomain().getCodeSource().getLocation().getPath();
        return "library=" + jar.substring(jar.lastIndexOf('/') + 1) + " provider=" + VectorizationProvider.getInstance().getClass().getSimpleName()
                + " similarity=DOT_PRODUCT degree=" + degree + " upperDegree=" + upperDegree + " efConstruction=" + efConstruction + " efSearch=" + efSearch
                + " alpha=" + alpha + " neighborOverflow=" + overflow + " removal=removeDeletedNodes removeAt=" + removeAt;
    }

    @Override public Map<String, Long> counters() {
        Map<String, Long> m = new LinkedHashMap<>();
        m.put("removals", removals.sum());
        m.put("nodesRemoved", removed.sum());
        m.put("writerWaits", writerWaits.sum());
        return m;
    }

    /** the writes that found a removal running or waiting, and the nanoseconds they waited; for a timed study */
    long writerWaits() { return writerWaits.sum(); }
    long writerWaitNanos() { return writerWaitNanos.sum(); }

    /** the lock that a removal holds, for the check of the waits */
    ReentrantReadWriteLock writeLockOfRemoval() { return writes; }

    @Override public void close() {
        try {
            for (GraphSearcher s : opened) s.close();
            builder.close();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        } finally {
            pool.shutdown();
        }
    }

    /**
     * The vectors and ids by ordinal. An ordinal is used once. It is written before the builder learns of it, and the builder publishes a node
     * through its own concurrent maps, so a reader that reaches an ordinal through the graph sees its vector.
     */
    private static final class Store implements RandomAccessVectorValues {
        private static final int SHIFT = 10, SIZE = 1 << SHIFT, MASK = SIZE - 1;

        private record Chunk(VectorFloat<?>[] vectors, long[] ids) {}

        private final int dim;
        private volatile Chunk[] chunks = new Chunk[0];
        private volatile int upper;

        Store(int dim) { this.dim = dim; }

        synchronized void set(int ordinal, long id, VectorFloat<?> v) {
            int c = ordinal >>> SHIFT;
            if (c >= chunks.length) {
                Chunk[] more = Arrays.copyOf(chunks, c + 1);
                for (int i = chunks.length; i < more.length; i++) more[i] = new Chunk(new VectorFloat<?>[SIZE], new long[SIZE]);
                chunks = more;
            }
            chunks[c].vectors()[ordinal & MASK] = v;
            chunks[c].ids()[ordinal & MASK] = id;
            upper = Math.max(upper, ordinal + 1);
        }

        synchronized void drop(int ordinal) { chunks[ordinal >>> SHIFT].vectors()[ordinal & MASK] = null; }

        long idOf(int ordinal) { return chunks[ordinal >>> SHIFT].ids()[ordinal & MASK]; }

        @Override public VectorFloat<?> getVector(int ordinal) { return chunks[ordinal >>> SHIFT].vectors()[ordinal & MASK]; }
        @Override public int size() { return upper; }
        @Override public int dimension() { return dim; }
        @Override public boolean isValueShared() { return false; }
        @Override public RandomAccessVectorValues copy() { return this; }
    }

    // ---- the count of the distance computations: JVector's own score functions, each call counted ----

    private BuildScoreProvider counted(BuildScoreProvider in) {
        return new BuildScoreProvider() {
            @Override public boolean isExact() { return in.isExact(); }
            @Override public VectorFloat<?> approximateCentroid() { return in.approximateCentroid(); }
            @Override public SearchScoreProvider searchProviderFor(VectorFloat<?> v) { return counted(in.searchProviderFor(v)); }
            @Override public SearchScoreProvider searchProviderFor(ByteSequence<?> v) { return counted(in.searchProviderFor(v)); }
            @Override public SearchScoreProvider searchProviderFor(int node) { return counted(in.searchProviderFor(node)); }
            @Override public SearchScoreProvider diversityProviderFor(int node) { return counted(in.diversityProviderFor(node)); }
            @Override public ScoreFunction diversityScoreFunctionFor(int node) { return new Counted(in.diversityScoreFunctionFor(node)); }
        };
    }

    private SearchScoreProvider counted(SearchScoreProvider in) {
        return new SearchScoreProvider() {
            @Override public ScoreFunction scoreFunction() { return new Counted(in.scoreFunction()); }
            @Override public ScoreFunction.ExactScoreFunction reranker() { return in.reranker() == null ? null : new Counted(in.reranker()); }

            @Override public ScoreFunction.ExactScoreFunction exactScoreFunction() {
                ScoreFunction.ExactScoreFunction f = in.exactScoreFunction();
                return f == null ? null : new Counted(f);
            }

            @Override public VectorSimilarityFunction adaptiveSimilarity() { return in.adaptiveSimilarity(); }
            @Override public float dotProductBound() { return in.dotProductBound(); }
        };
    }

    private final class Counted implements ScoreFunction.ExactScoreFunction {
        private final ScoreFunction in;

        Counted(ScoreFunction in) { this.in = in; }

        @Override public boolean isExact() { return in.isExact(); }

        @Override public float similarityTo(int node) {
            distances.increment();
            return in.similarityTo(node);
        }

        @Override public void similarityTo(int[] nodes, int count, float[] scores) {
            distances.add(count);
            in.similarityTo(nodes, count, scores);
        }

        @Override public float similarityToNeighbor(int origin, int neighbor) {
            distances.increment();
            return in.similarityToNeighbor(origin, neighbor);
        }

        @Override public boolean supportsBatchScoring() { return in.supportsBatchScoring(); }
        @Override public void enableSimilarityToNeighbors(int origin) { in.enableSimilarityToNeighbors(origin); }
        @Override public boolean supportsSimilarityToNeighbors() { return in.supportsSimilarityToNeighbors(); }
        @Override public boolean requiresAdaptiveTermination() { return in.requiresAdaptiveTermination(); }
        @Override public float dotProductBound() { return in.dotProductBound(); }
        @Override public VectorSimilarityFunction similarityFunction() { return in.similarityFunction(); }
    }
}
