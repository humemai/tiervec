package ai.humem.tiervec;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntFunction;
import java.util.function.IntPredicate;

/**
 * The shape of the layer-0 graphs of an arm at a checkpoint of the agent-memory benchmark (docs/SCREEN-IDEAS.md, issue #161): out-degrees to
 * live nodes, live nodes that no live list holds, live nodes that no walk from the entry reaches, and edges that lead to another cluster of
 * the generated data. It reads the graphs through reflection and calls no distance kernel of the library, so it changes neither an index nor
 * the counts of {@link DistanceCount}.
 */
final class GraphShape {
    /** a layer-0 graph as it is read: node numbers from 0; a node that is not passable is not walked and does not count */
    private record Graph(int nodes, IntPredicate passable, IntPredicate live, IntFunction<int[]> list, int entry, IntFunction<float[]> vector, int m) {}

    int graphs;
    long nodes, edges, listed, crossEdges, belowM, listedByNone, unreached;
    /** live nodes by their number of out-edges to live nodes */
    private long[] degrees = new long[1];

    /** the graph of a live {@link GraphIndex}; the caller has no write running */
    void add(GraphIndex g, float[][] centres) {
        GraphIndex.Chunk[] chunks = (GraphIndex.Chunk[]) read(g, "chunks");
        int n = ((AtomicInteger) read(g, "nextOrd")).get();
        GraphIndex.Entry entry = (GraphIndex.Entry) read(g, "entry");
        int dim = g.dim();
        IntFunction<GraphIndex.Chunk> chunk = o -> chunks[o >>> GraphIndex.CHUNK_SHIFT];
        IntPredicate live = o -> chunk.apply(o).state[o & GraphIndex.MASK] == GraphIndex.LIVE;
        IntPredicate passable = o -> live.test(o) || chunk.apply(o).state[o & GraphIndex.MASK] == GraphIndex.DELETED;
        IntFunction<int[]> list = o -> {
            int[][] layers = chunk.apply(o).links[o & GraphIndex.MASK];
            return layers == null || layers.length == 0 ? GraphIndex.EMPTY : layers[0];
        };
        IntFunction<float[]> vector = o -> Arrays.copyOfRange(chunk.apply(o).vec, (o & GraphIndex.MASK) * dim, ((o & GraphIndex.MASK) + 1) * dim);
        add(new Graph(n, passable, live, list, entry == null ? -1 : entry.node(), vector, g.params().M()), centres);
    }

    /** the graphs of the sealed segments of a tier; a hidden row is walked and is not live */
    void add(TieredIndex tier, float[][] centres) {
        for (Object segment : (Object[]) read(read(tier, "view"), "segs")) {
            SealedGraph g = (SealedGraph) read(segment, "g");
            Method hidden;
            try {
                hidden = segment.getClass().getDeclaredMethod("isHidden", int.class);
                hidden.setAccessible(true);
            } catch (NoSuchMethodException e) {
                throw new IllegalStateException("the segment of the tier has no method isHidden", e);
            }
            IntPredicate live = r -> {
                try {
                    return !g.isDead(r) && !(Boolean) hidden.invoke(segment, r);
                } catch (ReflectiveOperationException e) {
                    throw new IllegalStateException("the hidden rows of a segment cannot be read", e);
                }
            };
            add(new Graph(g.n, r -> true, live, r -> Arrays.copyOfRange(g.edges0, g.off0[r], g.off0[r + 1]), g.entryRow, g::rawVector,
                    g.params.M()), centres);
        }
    }

    private static Object read(Object owner, String name) {
        try {
            Field f = owner.getClass().getDeclaredField(name);
            f.setAccessible(true);
            return f.get(owner);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException(owner.getClass().getSimpleName() + " has no readable field " + name, e);
        }
    }

    /** the cluster of a unit vector: the centre with the largest dot product */
    static int cluster(float[] v, float[][] centres) {
        int best = 0;
        double most = -Double.MAX_VALUE;
        for (int c = 0; c < centres.length; c++) {
            double dot = 0;
            for (int i = 0; i < v.length; i++) dot += (double) v[i] * centres[c][i];
            if (dot > most) {
                most = dot;
                best = c;
            }
        }
        return best;
    }

    private void add(Graph g, float[][] centres) {
        int n = g.nodes();
        boolean[] live = new boolean[n];
        int[] cluster = new int[n];
        for (int i = 0; i < n; i++) {
            live[i] = g.live().test(i);
            if (live[i]) cluster[i] = cluster(g.vector().apply(i), centres);
        }
        boolean[] held = new boolean[n];
        for (int i = 0; i < n; i++) {
            if (!live[i]) continue;
            int degree = 0;
            for (int v : g.list().apply(i)) {
                listed++;
                if (v >= n || !live[v]) continue;
                degree++;
                held[v] = true;
                if (cluster[v] != cluster[i]) crossEdges++;
            }
            nodes++;
            edges += degree;
            if (degree < g.m()) belowM++;
            if (degree >= degrees.length) degrees = Arrays.copyOf(degrees, degree + 1);
            degrees[degree]++;
        }
        boolean[] seen = new boolean[n];
        if (g.entry() >= 0 && g.entry() < n) {
            int[] queue = new int[n];
            int head = 0, tail = 0;
            queue[tail++] = g.entry();
            seen[g.entry()] = true;
            while (head < tail) {
                int u = queue[head++];
                if (!g.passable().test(u)) continue;
                for (int v : g.list().apply(u)) {
                    if (v < n && !seen[v]) {
                        seen[v] = true;
                        queue[tail++] = v;
                    }
                }
            }
        }
        for (int i = 0; i < n; i++) {
            if (!live[i]) continue;
            if (!held[i]) listedByNone++;
            if (!seen[i]) unreached++;
        }
        graphs++;
    }

    /** the smallest degree that at least this share of the live nodes has or is below; share 0 gives the smallest degree */
    int degreeAt(double share) {
        long need = Math.max(1, (long) Math.ceil(share * nodes)), seen = 0;
        for (int d = 0; d < degrees.length; d++) {
            seen += degrees[d];
            if (seen >= need) return d;
        }
        return degrees.length - 1;
    }

    /** the live nodes by degree in steps of four, as "0-3:n,4-7:n,..." */
    String histogram() {
        StringBuilder s = new StringBuilder();
        for (int from = 0; from < degrees.length; from += 4) {
            long sum = 0;
            for (int d = from; d < Math.min(from + 4, degrees.length); d++) sum += degrees[d];
            s.append(s.length() == 0 ? "" : ",").append(from).append('-').append(from + 3).append(':').append(sum);
        }
        return s.toString();
    }

    static final String HEADER = String.join("\t", "graphs", "nodes", "degreeMean", "degreeMin", "degreeP10", "degreeMedian", "degreeP90",
            "degreeMax", "shareBelowM", "listedByNone", "unreached", "crossClusterShare", "listedPerNode", "degreeHistogram");

    /** the values of {@link #HEADER}; a node below M has fewer live neighbours than half of the layer-0 limit */
    String row() {
        double of = Math.max(1, nodes);
        return String.join("\t", String.valueOf(graphs), String.valueOf(nodes), Bench.f("%.2f", edges / of), String.valueOf(degreeAt(0)),
                String.valueOf(degreeAt(0.1)), String.valueOf(degreeAt(0.5)), String.valueOf(degreeAt(0.9)), String.valueOf(degreeAt(1)),
                Bench.f("%.4f", belowM / of), String.valueOf(listedByNone), String.valueOf(unreached),
                Bench.f("%.4f", crossEdges / (double) Math.max(1, edges)), Bench.f("%.2f", listed / of), histogram());
    }
}
