package ai.humem.tiervec;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.DigestInputStream;
import java.security.DigestOutputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * A runbook of the agent-memory benchmark (docs/AGENT-MEMORY-BENCHMARK.md, issue #159): the operations that every arm replays, generated
 * once from a seed and written to one file. The file holds every vector, so a replay reads nothing else and every arm gets the same bytes
 * (the SHA-256 of the file goes into the results). A runbook has one or more collections, each with its own operations; a collection is
 * replayed into its own index. Cosine only: every vector has length 1.
 *
 * <p>A collection starts with a load of {@code rows} rows and a checkpoint. Then it runs in rounds. A round retires {@code max(4, rows / 10)}
 * row versions: 30 percent expire (the oldest rows are deleted), 30 percent are rows of one cluster that are deleted, 30 percent are
 * updated, and the rest are deleted and later added again under the same id (reuse). New rows replace the deleted ones. New and replaced
 * vectors are drawn from the cluster that grows in the round, so the distribution moves. The rounds end when {@code turnover * rows} versions
 * have been retired. Some writes arrive in batches of 1 to {@code maxBatch} writes, and no id appears twice in a batch. Searches are placed
 * between the writes at {@code searchesPerWrite} per written row.
 *
 * <p>There are two kinds of data. With {@code latent} 0 a row is the centre of its cluster plus noise in all dimensions, and a query is near a
 * live row. With {@code latent} above 0 the noise of a row lies in the span of {@code latent} random directions, which sets how many
 * dimensions the rows of a cluster really use (all of them when {@code latent} is the dimension), and a query is drawn as a new row of the
 * cluster of a random live row, so it is no near copy of a stored row.
 */
final class Runbook {
    /** the kinds of operation; the ordinal is the byte in the file, so the order is part of the format */
    enum Kind { CREATE, UPDATE, DELETE, REUSE, BATCH, SEARCH, CHECKPOINT }

    /**
     * One operation. A write names its id and, unless it is a delete, its vector. A batch holds its writes. A search holds its query in
     * {@code vec}. A checkpoint holds the queries that recall is scored on; nothing is persisted at a checkpoint yet.
     */
    record Op(Kind kind, long id, float[] vec, List<Op> writes, float[][] queries) {
        static Op write(Kind kind, long id, float[] vec) { return new Op(kind, id, vec, null, null); }
        static Op batch(List<Op> writes) { return new Op(Kind.BATCH, 0, null, List.copyOf(writes), null); }
        static Op search(float[] query) { return new Op(Kind.SEARCH, 0, query, null, null); }
        static Op checkpoint(float[][] queries) { return new Op(Kind.CHECKPOINT, 0, null, null, queries); }
    }

    /**
     * What the generator is asked for. rows: the live rows of a collection. turnover: the retired row versions of a collection, as a
     * multiple of rows. searchesPerWrite: search operations per written row. checkpoints: the checkpoints after the one that follows the
     * load (fewer when there are fewer rounds). queries: the queries of a checkpoint. maxBatch: the largest batch, 0 for no batches. latent:
     * the kind of data (see the class), from 0 to the dimension.
     */
    record Params(long seed, int dim, int rows, int collections, double turnover, int clusters, double searchesPerWrite, int checkpoints,
                  int queries, int maxBatch, int latent) {
        Params {
            if (dim < 2 || rows < 4 || collections < 1 || !(turnover >= 0 && turnover < 1000) || clusters < 2
                    || !(searchesPerWrite >= 0 && searchesPerWrite < 1000) || checkpoints < 1 || queries < 1 || maxBatch < 0
                    || latent < 0 || latent > dim) {
                throw new IllegalArgumentException("bad runbook parameters");
            }
        }

        /** the first kind of data: noise in all dimensions, and queries near live rows */
        Params(long seed, int dim, int rows, int collections, double turnover, int clusters, double searchesPerWrite, int checkpoints,
               int queries, int maxBatch) {
            this(seed, dim, rows, collections, turnover, clusters, searchesPerWrite, checkpoints, queries, maxBatch, 0);
        }

        String describe() {
            return "seed=" + seed + " dim=" + dim + " rows=" + rows + " collections=" + collections + " turnover=" + turnover + " clusters=" + clusters
                    + " searchesPerWrite=" + searchesPerWrite + " checkpoints=" + checkpoints + " queries=" + queries + " maxBatch=" + maxBatch
                    + " latent=" + latent;
        }
    }

    private static final int MAGIC = 0x54564D52; // "TVMR"
    private static final int FORMAT = 2;
    /** the chance that a batch starts at a write that is not in one; with batches of 25 writes on average about half of the writes are batched */
    private static final double BATCH_START = 0.04;
    /**
     * The length of the noise that separates a row from the centre of its cluster (a centre has length 1), and, with latent 0, a query from the
     * row it is near.
     */
    private static final double SPREAD = 0.8, QUERY_NOISE = 0.3;

    final Params params;
    final List<List<Op>> collections;

    private Runbook(Params params, List<List<Op>> collections) {
        this.params = params;
        this.collections = collections;
    }

    static Runbook generate(Params p) {
        List<List<Op>> all = new ArrayList<>(p.collections());
        for (int c = 0; c < p.collections(); c++) all.add(new Generator(p, c).run());
        return new Runbook(p, all);
    }

    /** the cluster centres of one collection, as the generator draws them; the shape of a graph is measured against them ({@link GraphShape}) */
    static float[][] centres(Params p, int collection) { return new Generator(p, collection).centres; }

    // ---- the file ----

    /** writes the runbook and returns the SHA-256 of the file */
    String write(Path file) throws IOException {
        if (file.getParent() != null) Files.createDirectories(file.getParent());
        MessageDigest md = sha256();
        byte[] buf = new byte[4 * params.dim()];
        try (DataOutputStream o = new DataOutputStream(new BufferedOutputStream(new DigestOutputStream(Files.newOutputStream(file), md), 1 << 16))) {
            o.writeInt(MAGIC);
            o.writeInt(FORMAT);
            o.writeLong(params.seed());
            o.writeInt(params.dim());
            o.writeInt(params.rows());
            o.writeInt(params.collections());
            o.writeDouble(params.turnover());
            o.writeInt(params.clusters());
            o.writeDouble(params.searchesPerWrite());
            o.writeInt(params.checkpoints());
            o.writeInt(params.queries());
            o.writeInt(params.maxBatch());
            o.writeInt(params.latent());
            for (List<Op> ops : collections) {
                o.writeInt(ops.size());
                for (Op op : ops) writeOp(o, op, buf);
            }
        }
        return HexFormat.of().formatHex(md.digest());
    }

    private static void writeOp(DataOutputStream o, Op op, byte[] buf) throws IOException {
        o.writeByte(op.kind().ordinal());
        switch (op.kind()) {
            case CREATE, UPDATE, REUSE -> {
                o.writeLong(op.id());
                writeVector(o, op.vec(), buf);
            }
            case DELETE -> o.writeLong(op.id());
            case BATCH -> {
                o.writeInt(op.writes().size());
                for (Op w : op.writes()) writeOp(o, w, buf);
            }
            case SEARCH -> writeVector(o, op.vec(), buf);
            case CHECKPOINT -> {
                o.writeInt(op.queries().length);
                for (float[] q : op.queries()) writeVector(o, q, buf);
            }
        }
    }

    private static void writeVector(DataOutputStream o, float[] v, byte[] buf) throws IOException {
        ByteBuffer.wrap(buf).asFloatBuffer().put(v);
        o.write(buf);
    }

    /** a runbook file as read: the runbook and the SHA-256 of the bytes it was read from */
    record Read(Runbook runbook, String sha256) {}

    static Read read(Path file) throws IOException {
        MessageDigest md = sha256();
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new DigestInputStream(Files.newInputStream(file), md), 1 << 16))) {
            if (in.readInt() != MAGIC) throw new IOException(file + " is not a runbook");
            int format = in.readInt();
            if (format != FORMAT) throw new IOException(file + " has runbook format " + format + ", this program reads " + FORMAT);
            Params p = new Params(in.readLong(), in.readInt(), in.readInt(), in.readInt(), in.readDouble(), in.readInt(), in.readDouble(), in.readInt(),
                    in.readInt(), in.readInt(), in.readInt());
            List<List<Op>> all = new ArrayList<>();
            byte[] buf = new byte[4 * p.dim()];
            for (int c = 0; c < p.collections(); c++) {
                int n = in.readInt();
                List<Op> ops = new ArrayList<>();
                for (int i = 0; i < n; i++) ops.add(readOp(in, p.dim(), buf, false));
                all.add(ops);
            }
            if (in.read() != -1) throw new IOException(file + " has bytes after its last operation");
            return new Read(new Runbook(p, all), HexFormat.of().formatHex(md.digest()));
        }
    }

    private static Op readOp(DataInputStream in, int dim, byte[] buf, boolean inBatch) throws IOException {
        int code = in.readUnsignedByte();
        if (code >= Kind.values().length) throw new IOException("unknown operation " + code);
        Kind kind = Kind.values()[code];
        if (inBatch && (kind == Kind.BATCH || kind == Kind.SEARCH || kind == Kind.CHECKPOINT)) throw new IOException(kind + " inside a batch");
        switch (kind) {
            case CREATE, UPDATE, REUSE -> {
                long id = in.readLong();
                return Op.write(kind, id, readVector(in, dim, buf));
            }
            case DELETE -> {
                return Op.write(kind, in.readLong(), null);
            }
            case BATCH -> {
                int n = in.readInt();
                List<Op> writes = new ArrayList<>();
                for (int i = 0; i < n; i++) writes.add(readOp(in, dim, buf, true));
                return Op.batch(writes);
            }
            case SEARCH -> {
                return Op.search(readVector(in, dim, buf));
            }
            default -> {
                int n = in.readInt();
                float[][] queries = new float[n][];
                for (int i = 0; i < n; i++) queries[i] = readVector(in, dim, buf);
                return Op.checkpoint(queries);
            }
        }
    }

    private static float[] readVector(DataInputStream in, int dim, byte[] buf) throws IOException {
        in.readFully(buf);
        float[] v = new float[dim];
        ByteBuffer.wrap(buf).asFloatBuffer().get(v);
        return v;
    }

    private static MessageDigest sha256() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("every Java platform has SHA-256", e);
        }
    }

    // ---- the generator ----

    private record Row(int cluster, float[] vec) {}

    private record Planned(Kind kind, long id) {}

    private static final class Generator {
        private final Params p;
        private final SplittableRandom r;
        private final float[][] centres;
        /** the directions that the noise of the rows spans, or null when it spans every dimension */
        private final float[][] directions;
        private final List<Op> ops = new ArrayList<>();
        /** the live rows in the order in which their ids were added, so the first ones are the oldest (a reused id moves to the end) */
        private final LinkedHashMap<Long, Row> live = new LinkedHashMap<>();
        /** the live ids again, for a uniform pick in constant time */
        private final List<Long> ids = new ArrayList<>();
        private final Map<Long, Integer> at = new HashMap<>();
        private long nextId = 1;
        private double owedSearches;

        Generator(Params p, int collection) {
            this.p = p;
            this.r = new SplittableRandom(Bench.mix(Bench.mix(p.seed()) + collection));
            this.centres = randomUnit(p.clusters());
            this.directions = p.latent() > 0 && p.latent() < p.dim() ? randomUnit(p.latent()) : null;
        }

        private float[][] randomUnit(int n) {
            float[][] all = new float[n][p.dim()];
            for (float[] v : all) {
                for (int i = 0; i < v.length; i++) v[i] = (float) Bench.Gen.gauss(r);
                Bench.Gen.norm(v);
            }
            return all;
        }

        List<Op> run() {
            int half = Math.max(1, p.clusters() / 2);
            for (int i = 0; i < p.rows(); i++) {
                ops.add(put(Kind.CREATE, nextId++, r.nextInt(half)));
                searches(1);
            }
            checkpoint();
            int retire = Math.max(4, p.rows() / 10);
            int rounds = (int) Math.ceil(p.turnover() * p.rows() / retire);
            int points = Math.min(p.checkpoints(), rounds);
            for (int round = 0; round < rounds; round++) {
                round(retire, (half + round) % p.clusters(), round % p.clusters());
                if ((long) (round + 1) * points / rounds > (long) round * points / rounds) checkpoint();
            }
            return ops;
        }

        private void round(int retire, int grows, int shrinks) {
            int expire = retire * 3 / 10, drop = retire * 3 / 10, update = retire * 3 / 10, reuse = retire - expire - drop - update;
            Set<Long> chosen = new HashSet<>();
            List<Planned> plan = new ArrayList<>();
            for (Long id : live.keySet()) {
                if (chosen.size() == expire) break;
                chosen.add(id);
                plan.add(new Planned(Kind.DELETE, id));
            }
            List<Long> ofCluster = new ArrayList<>();
            for (Map.Entry<Long, Row> e : live.entrySet()) {
                if (e.getValue().cluster() == shrinks && !chosen.contains(e.getKey())) ofCluster.add(e.getKey());
            }
            for (int i = 0; i < drop; i++) {
                long id;
                if (i < ofCluster.size()) { // a part of a Fisher-Yates shuffle
                    int j = i + r.nextInt(ofCluster.size() - i);
                    Collections.swap(ofCluster, i, j);
                    id = ofCluster.get(i);
                    chosen.add(id);
                } else {
                    id = pickOther(chosen); // the cluster has too few rows left
                }
                plan.add(new Planned(Kind.DELETE, id));
            }
            for (int i = 0; i < update; i++) plan.add(new Planned(Kind.UPDATE, pickOther(chosen)));
            for (int i = 0; i < reuse; i++) {
                long id = pickOther(chosen);
                plan.add(new Planned(Kind.REUSE, id)); // twice: the first of the two in the shuffled order is the delete
                plan.add(new Planned(Kind.REUSE, id));
            }
            for (int i = 0; i < expire + drop; i++) plan.add(new Planned(Kind.CREATE, 0));
            for (int i = plan.size() - 1; i > 0; i--) Collections.swap(plan, i, r.nextInt(i + 1));
            Set<Long> deletedForReuse = new HashSet<>();
            for (int i = 0; i < plan.size(); i++) {
                Planned x = plan.get(i);
                if (x.kind() == Kind.REUSE && deletedForReuse.add(x.id())) plan.set(i, new Planned(Kind.DELETE, x.id()));
            }
            for (int i = 0; i < plan.size();) {
                if (p.maxBatch() > 0 && r.nextDouble() < BATCH_START) {
                    int length = 1 + r.nextInt(p.maxBatch());
                    List<Op> batch = new ArrayList<>();
                    Set<Long> inBatch = new HashSet<>();
                    while (batch.size() < length && i < plan.size()) {
                        Planned x = plan.get(i);
                        if (x.kind() != Kind.CREATE && !inBatch.add(x.id())) break;
                        batch.add(emit(x, grows));
                        i++;
                    }
                    ops.add(Op.batch(batch));
                    searches(batch.size());
                } else {
                    ops.add(emit(plan.get(i), grows));
                    i++;
                    searches(1);
                }
            }
        }

        /** one planned write as an operation, with a new vector from the cluster that grows, applied to the live rows of the generator */
        private Op emit(Planned x, int grows) {
            long id = x.kind() == Kind.CREATE ? nextId++ : x.id();
            if (x.kind() == Kind.DELETE) {
                remove(id);
                return Op.write(Kind.DELETE, id, null);
            }
            return put(x.kind(), id, grows);
        }

        private Op put(Kind kind, long id, int cluster) {
            float[] v = draw(cluster);
            if (live.put(id, new Row(cluster, v)) == null) {
                at.put(id, ids.size());
                ids.add(id);
            }
            return Op.write(kind, id, v);
        }

        private void remove(long id) {
            live.remove(id);
            int i = at.remove(id);
            long last = ids.remove(ids.size() - 1);
            if (last != id) {
                ids.set(i, last);
                at.put(last, i);
            }
        }

        private long pickOther(Set<Long> chosen) {
            while (true) {
                long id = ids.get(r.nextInt(ids.size()));
                if (chosen.add(id)) return id;
            }
        }

        private float[] near(float[] v, double noise) {
            float scale = (float) (noise / StrictMath.sqrt(v.length));
            float[] w = new float[v.length];
            for (int i = 0; i < v.length; i++) w[i] = v[i] + scale * (float) Bench.Gen.gauss(r);
            Bench.Gen.norm(w);
            return w;
        }

        /** a new vector of a cluster: its centre plus noise of length about SPREAD, in the span of the directions when there are any */
        private float[] draw(int cluster) {
            if (directions == null) return near(centres[cluster], SPREAD);
            float scale = (float) (SPREAD / StrictMath.sqrt(directions.length));
            float[] w = centres[cluster].clone();
            for (float[] d : directions) {
                float g = scale * (float) Bench.Gen.gauss(r);
                for (int i = 0; i < w.length; i++) w[i] += g * d[i];
            }
            Bench.Gen.norm(w);
            return w;
        }

        private float[] query() {
            Row row = live.get(ids.get(r.nextInt(ids.size())));
            return p.latent() == 0 ? near(row.vec(), QUERY_NOISE) : draw(row.cluster());
        }

        private void searches(int written) {
            owedSearches += written * p.searchesPerWrite();
            while (owedSearches >= 1) {
                owedSearches -= 1;
                if (!ids.isEmpty()) ops.add(Op.search(query()));
            }
        }

        private void checkpoint() {
            float[][] queries = new float[p.queries()][];
            for (int i = 0; i < queries.length; i++) queries[i] = query();
            ops.add(Op.checkpoint(queries));
        }
    }
}
