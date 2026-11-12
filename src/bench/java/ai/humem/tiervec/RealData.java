package ai.humem.tiervec;

import java.io.EOFException;
import java.io.IOException;
import java.io.Reader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import java.util.Properties;
import java.util.TreeMap;

/**
 * A real embedding set for the harness ({@link Bench}, option {@code --data}; docs/BENCHMARKS.md, issue #108). It is a directory that
 * {@code scripts/prep_real_dataset.py} writes, with four files:
 *
 * <ul>
 * <li>{@code meta.properties}: {@code key=value} lines. The keys the loader reads are {@code format} (1), {@code dim}, {@code count} (vectors in
 * the index), {@code queries}, {@code truth-k} (exact answers per query) and {@code normalized} (must be {@code true}). Every other key is
 * copied into the DATASET line of the result file (where the data came from, the seed, the range of the lengths before normalising).</li>
 * <li>{@code vectors.f32}: {@code count} vectors of {@code dim} floats, 4 bytes each, little endian, row after row. The id of a vector is its row.</li>
 * <li>{@code queries.f32}: {@code queries} vectors, the same layout. They are held out: none of them is a row of {@code vectors.f32}.</li>
 * <li>{@code truth.i32}: {@code queries} rows of {@code truth-k} int32 values, little endian: the rows of {@code vectors.f32} that are nearest to the
 * query by the cosine, best first. The vectors have unit length, so the cosine is the dot product, which is what the harness scores by.</li>
 * </ul>
 *
 * <p>The loader checks the sizes of the files against the header, and every vector for a non-finite value and for a length other than 1 (to 1e-3),
 * so a file that was not normalised fails before a measurement is made on it.
 */
final class RealData {
    static final String META = "meta.properties", VECTORS = "vectors.f32", QUERIES = "queries.f32", TRUTH = "truth.i32";

    final Path dir;
    final int dim, count, queries, truthK;
    /** every key of meta.properties, sorted */
    final Map<String, String> meta;

    private RealData(Path dir, Map<String, String> meta, int dim, int count, int queries, int truthK) {
        this.dir = dir;
        this.meta = meta;
        this.dim = dim;
        this.count = count;
        this.queries = queries;
        this.truthK = truthK;
    }

    /** {@code p} is the directory, or any file inside it (so that the path of {@code vectors.f32} works too) */
    static RealData open(Path p) throws IOException {
        if (!Files.exists(p)) throw new IllegalArgumentException("--data " + p + " does not exist");
        Path dir = Files.isDirectory(p) ? p : p.toAbsolutePath().getParent();
        Path mp = dir.resolve(META);
        if (!Files.isRegularFile(mp)) throw new IllegalArgumentException(dir + " has no " + META + " (scripts/prep_real_dataset.py writes it last; docs/BENCHMARKS.md)");
        Properties pr = new Properties();
        try (Reader r = Files.newBufferedReader(mp, StandardCharsets.UTF_8)) {
            pr.load(r);
        }
        Map<String, String> meta = new TreeMap<>();
        for (String key : pr.stringPropertyNames()) meta.put(key, pr.getProperty(key).trim());
        int format = intOf(meta, "format");
        if (format != 1) throw new IllegalArgumentException(mp + ": format " + format + " is not supported (this harness reads format 1)");
        int dim = intOf(meta, "dim"), count = intOf(meta, "count"), queries = intOf(meta, "queries"), truthK = intOf(meta, "truth-k");
        if (dim < 1 || count < 1 || queries < 1 || truthK < 1) throw new IllegalArgumentException(mp + ": dim, count, queries and truth-k must be positive");
        if (!"true".equals(meta.get("normalized")))
            throw new IllegalArgumentException(mp + ": normalized must be true (the harness scores by the dot product, which is the cosine only for unit vectors)");
        expectSize(dir.resolve(VECTORS), (long) count * dim * 4);
        expectSize(dir.resolve(QUERIES), (long) queries * dim * 4);
        expectSize(dir.resolve(TRUTH), (long) queries * truthK * 4);
        return new RealData(dir, meta, dim, count, queries, truthK);
    }

    private static int intOf(Map<String, String> meta, String key) {
        String v = meta.get(key);
        if (v == null) throw new IllegalArgumentException(META + " has no key " + key);
        try {
            return Integer.parseInt(v);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(META + ": " + key + "=" + v + " is not an int");
        }
    }

    private static void expectSize(Path file, long bytes) throws IOException {
        if (!Files.isRegularFile(file)) throw new IllegalArgumentException("missing " + file);
        long have = Files.size(file);
        if (have != bytes) throw new IllegalArgumentException(file + " is " + have + " bytes, the header says " + bytes + " (a partial or a wrong file)");
    }

    /** the DATASET line of the result file: where the files are, and every key of the header */
    String describe() {
        StringBuilder sb = new StringBuilder("path=").append(dir.toAbsolutePath());
        for (Map.Entry<String, String> e : meta.entrySet()) sb.append(' ').append(e.getKey()).append('=').append(e.getValue());
        return sb.toString();
    }

    /** all vectors of the index: row i is the vector with id i */
    float[][] readVectors() throws IOException {
        return readRows(dir.resolve(VECTORS), count);
    }

    /** the first {@code nq} queries */
    float[][] readQueries(int nq) throws IOException {
        if (nq > queries) throw new IllegalArgumentException(nq + " queries asked, the file holds " + queries);
        return readRows(dir.resolve(QUERIES), nq);
    }

    /** the exact answers of the first {@code nq} queries, the first {@code k} of each, as ids (rows of the vectors), best first */
    long[][] readTruth(int nq, int k) throws IOException {
        if (nq > queries || k > truthK) throw new IllegalArgumentException("asked for " + nq + " queries and " + k + " answers, the file holds " + queries + " and " + truthK);
        long[][] t = new long[nq][k];
        ByteBuffer bb = ByteBuffer.allocate(truthK * 4).order(ByteOrder.LITTLE_ENDIAN);
        Path file = dir.resolve(TRUTH);
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
            for (int q = 0; q < nq; q++) {
                bb.clear();
                while (bb.hasRemaining()) if (ch.read(bb) < 0) throw new EOFException(file + " ends early at query " + q);
                bb.flip();
                for (int j = 0; j < truthK; j++) {
                    int id = bb.getInt();
                    if (j >= k) continue;
                    if (id < 0 || id >= count) throw new IllegalArgumentException(file + ": query " + q + " has the id " + id + ", the index holds " + count + " vectors");
                    t[q][j] = id;
                }
            }
        }
        return t;
    }

    private float[][] readRows(Path file, int rows) throws IOException {
        float[][] v = new float[rows][];
        final int perRead = Math.max(1, (8 << 20) / (dim * 4)); // about 8 MiB per read
        ByteBuffer bb = ByteBuffer.allocateDirect(perRead * dim * 4).order(ByteOrder.LITTLE_ENDIAN);
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
            int row = 0;
            while (row < rows) {
                int take = Math.min(perRead, rows - row);
                bb.clear();
                bb.limit(take * dim * 4);
                while (bb.hasRemaining()) if (ch.read(bb) < 0) throw new EOFException(file + " ends early at row " + row);
                bb.flip();
                FloatBuffer fb = bb.asFloatBuffer();
                for (int r = 0; r < take; r++) {
                    float[] x = new float[dim];
                    fb.get(x);
                    check(file, row + r, x);
                    v[row + r] = x;
                }
                row += take;
            }
        }
        return v;
    }

    private static void check(Path file, int row, float[] x) {
        double s = 0;
        for (float f : x) {
            if (!Float.isFinite(f)) throw new IllegalArgumentException(file + ": row " + row + " holds a value that is not a finite number");
            s += (double) f * f;
        }
        if (Math.abs(Math.sqrt(s) - 1.0) > 1e-3)
            throw new IllegalArgumentException(file + ": row " + row + " has length " + Math.sqrt(s) + ", not 1 (scripts/prep_real_dataset.py normalises the vectors)");
    }
}
