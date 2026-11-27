package vq;

import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.IOException;
import java.lang.management.BufferPoolMXBean;
import java.lang.management.ManagementFactory;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/** Dataset loader (the prep_data.py layout), JSON line helper, memory sampler. Shared by the CLI and the JMH benchmarks. */
public final class U {
  private U() { }

  /** One prepared cell: base.npy (float32 N x dim), q.npy (float32), gt.npy (int32 exact top-10), meta.json. */
  public static final class Data {
    public String dataset, similarity;
    public int n, dim, nq;
    public float[][] base;       // null when loaded without the base (ArcadeDB query process: vectors live in the records)
    public float[][] queries;
    public int[][] gt;
  }

  private static int rows, cols; private static long off; private static String descr;

  private static void npyHeader(final Path p) throws IOException {
    try (FileChannel ch = FileChannel.open(p)) {
      final ByteBuffer b = ByteBuffer.allocate(512).order(ByteOrder.LITTLE_ENDIAN); ch.read(b, 0); b.flip();
      final int major = b.get(6); final int hl = major == 1 ? (b.getShort(8) & 0xffff) : b.getInt(8); final int hs = major == 1 ? 10 : 12;
      final byte[] h = new byte[hl]; b.position(hs); b.get(h); final String s = new String(h);
      final Matcher d = Pattern.compile("'descr':\\s*'([^']+)'").matcher(s); if (!d.find()) throw new IOException("descr " + s); descr = d.group(1);
      final Matcher sh = Pattern.compile("\\((\\d+),\\s*(\\d+)\\)").matcher(s); if (!sh.find()) throw new IOException("shape " + s);
      rows = Integer.parseInt(sh.group(1)); cols = Integer.parseInt(sh.group(2)); off = hs + hl;
    }
  }

  static float[][] floats(final Path p, final int limitRows) throws IOException {
    npyHeader(p); if (!descr.equals("<f4")) throw new IOException("float32 expected: " + descr);
    final int n = Math.min(rows, limitRows), dim = cols; final float[][] out = new float[n][];
    try (FileChannel ch = FileChannel.open(p)) {
      final int chunk = 20000; final ByteBuffer b = ByteBuffer.allocate(chunk * dim * 4).order(ByteOrder.LITTLE_ENDIAN);
      for (int s = 0; s < n; s += chunk) {
        final int c = Math.min(chunk, n - s); b.clear(); b.limit(c * dim * 4); ch.read(b, off + (long) s * dim * 4); b.flip();
        for (int i = 0; i < c; i++) { final float[] r = new float[dim]; b.asFloatBuffer().get(r); b.position(b.position() + dim * 4); out[s + i] = r; }
      }
    }
    return out;
  }

  static int[][] ints(final Path p, final int limitRows) throws IOException {
    npyHeader(p); if (!descr.equals("<i4")) throw new IOException("int32 expected: " + descr);
    final int n = Math.min(rows, limitRows); final int[][] g = new int[n][cols];
    try (FileChannel ch = FileChannel.open(p)) {
      final ByteBuffer b = ByteBuffer.allocate(rows * cols * 4).order(ByteOrder.LITTLE_ENDIAN); ch.read(b, off); b.flip();
      for (int i = 0; i < n; i++) for (int j = 0; j < cols; j++) g[i][j] = b.getInt();
    }
    return g;
  }

  static String meta(final String text, final String key) throws IOException {
    final Matcher m = Pattern.compile("\"" + key + "\":\\s*\"?([A-Za-z0-9_.-]+)\"?").matcher(text); if (!m.find()) throw new IOException("meta " + key); return m.group(1);
  }

  /** @param withBase false for the ArcadeDB query process (saves the fp32 copy on the heap). @param nq number of queries to use (0 = all in q.npy) */
  public static Data load(final Path dir, final boolean withBase, final int nq) throws IOException {
    final Data d = new Data(); final String m = Files.readString(dir.resolve("meta.json"));
    d.dataset = meta(m, "dataset"); d.similarity = meta(m, "similarity");
    d.queries = floats(dir.resolve("q.npy"), nq > 0 ? nq : Integer.MAX_VALUE); d.nq = d.queries.length;
    d.gt = ints(dir.resolve("gt.npy"), d.nq);
    if (withBase) { d.base = floats(dir.resolve("base.npy"), Integer.MAX_VALUE); d.n = d.base.length; d.dim = d.base[0].length; }
    else { npyHeader(dir.resolve("base.npy")); d.n = rows; d.dim = cols; }
    return d;
  }

  // ---------- JSON line helper (flat key/value pairs)
  public static String json(final Object... kv) {
    final StringBuilder sb = new StringBuilder("{");
    for (int i = 0; i < kv.length; i += 2) {
      if (i > 0) sb.append(','); sb.append('"').append(kv[i]).append("\":"); final Object v = kv[i + 1];
      if (v instanceof String) sb.append('"').append(((String) v).replace("\\", "\\\\").replace("\"", "\\\"")).append('"'); else sb.append(v);
    }
    return sb.append('}').toString();
  }
  /** A leading newline: the engine logs INFO lines without a trailing newline, which would swallow the RESULT marker (the arcade_arm.py convention). */
  public static void result(final Object... kv) { System.out.println("\nRESULT " + json(kv)); System.out.flush(); }
  public static double r(final double v, final int digits) { final double p = Math.pow(10, digits); return Math.round(v * p) / p; }

  // ---------- memory
  static long procKb(final String key) {
    try { for (final String l : Files.readAllLines(Paths.get("/proc/self/status"))) if (l.startsWith(key)) return Long.parseLong(l.replaceAll("\\D+", "")); } catch (final IOException e) { }
    return -1;
  }
  public static long rssMb() { return procKb("VmRSS") / 1024; }
  public static long hwmMb() { return procKb("VmHWM") / 1024; }
  /** RssAnon / RssFile of /proc/self/status: anonymous (heap, native) versus file-backed (mmapped index files: Lucene, jvector graph readers) resident pages; RSS = anon + file (+ shmem). */
  public static long anonMb() { return procKb("RssAnon") / 1024; }
  public static long fileMb() { return procKb("RssFile") / 1024; }
  public static long heapUsedMb() { final Runtime rt = Runtime.getRuntime(); return (rt.totalMemory() - rt.freeMemory()) >> 20; }
  public static long heapAfterGcMb() { System.gc(); try { Thread.sleep(200); } catch (final InterruptedException e) { } System.gc(); return heapUsedMb(); }
  public static long directMb() {
    for (final BufferPoolMXBean b : ManagementFactory.getPlatformMXBeans(BufferPoolMXBean.class)) if (b.getName().equals("direct")) return b.getMemoryUsed() >> 20;
    return -1;
  }
  public static long nonHeapMb() { return ManagementFactory.getMemoryMXBean().getNonHeapMemoryUsage().getUsed() >> 20; }

  /** Samples heap used and RSS every 50 ms from a daemon thread: peaks over a phase (build, query). Sampled, not a timing figure. */
  public static final class Peak implements AutoCloseable {
    private volatile boolean run = true; private volatile long heap, rss, anon; private final Thread t;
    public Peak() {
      t = new Thread(() -> { while (run) { heap = Math.max(heap, heapUsedMb()); rss = Math.max(rss, rssMb()); anon = Math.max(anon, anonMb()); try { Thread.sleep(50); } catch (final InterruptedException e) { return; } } }, "peak-sampler");
      t.setDaemon(true); t.start();
    }
    public long heapMb() { return Math.max(heap, heapUsedMb()); }
    public long rssMb() { return Math.max(rss, U.rssMb()); }
    public long anonMb() { return Math.max(anon, U.anonMb()); }
    @Override public void close() { run = false; t.interrupt(); }
  }

  public static long dirBytes(final Path dir) throws IOException { return dirBytesByExt(dir).values().stream().mapToLong(Long::longValue).sum(); }
  public static SortedMap<String, Long> dirBytesByExt(final Path dir) throws IOException {
    final SortedMap<String, Long> by = new TreeMap<>();
    try (Stream<Path> s = Files.walk(dir)) {
      s.filter(Files::isRegularFile).forEach(p -> { final String f = p.getFileName().toString(); final int i = f.lastIndexOf('.'); final String e = i >= 0 ? f.substring(i + 1) : "other";
        try { by.merge(e, Files.size(p), Long::sum); } catch (final IOException x) { } });
    }
    return by;
  }
  /** ArcadeDB: the index's own files = everything but the record buckets (which keep the fp32 vectors), the dictionary, the schema json, wal, lck. */
  public static long indexBytes(final SortedMap<String, Long> by) { long s = 0; for (final Map.Entry<String, Long> e : by.entrySet()) if (!Set.of("bucket", "dict", "json", "bin", "wal", "lck").contains(e.getKey())) s += e.getValue(); return s; }
  /** Lucene: the index files except the raw fp32 vectors (.vec, which the quantized arms keep for re-ranking, like ArcadeDB keeps them in its record buckets) and the lock. */
  public static long luIndexBytes(final SortedMap<String, Long> by) { long s = 0; for (final Map.Entry<String, Long> e : by.entrySet()) if (!Set.of("vec", "lock").contains(e.getKey())) s += e.getValue(); return s; }
  public static String byExtJson(final SortedMap<String, Long> by) { final List<Object> kv = new ArrayList<>(); by.forEach((k, v) -> { kv.add(k); kv.add(v); }); return json(kv.toArray()); }
  public static void deleteTree(final Path dir) throws IOException {
    if (!Files.exists(dir)) return;
    try (Stream<Path> s = Files.walk(dir)) { s.sorted(Comparator.reverseOrder()).forEach(p -> { try { Files.delete(p); } catch (final IOException e) { } }); }
  }

  public static String vectorization() { return VectorizationProvider.getInstance().getClass().getSimpleName(); }
  public static final VectorTypeSupport VTS = VectorizationProvider.getInstance().getVectorTypeSupport();
  public static VectorFloat<?> vec(final float[] a) { return VTS.createFloatVector(a); }
}
