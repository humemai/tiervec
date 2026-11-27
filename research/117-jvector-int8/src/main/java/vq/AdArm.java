package vq;

import com.arcadedb.database.Database;
import com.arcadedb.database.DatabaseFactory;
import com.arcadedb.database.Document;
import com.arcadedb.database.MutableDocument;
import com.arcadedb.database.RID;
import com.arcadedb.index.Index;
import com.arcadedb.index.TypeIndex;
import com.arcadedb.index.vector.LSMVectorIndex;
import com.arcadedb.query.sql.executor.WorkGuard;
import com.arcadedb.utility.Pair;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.nio.file.*;
import java.util.*;

/** ArcadeDB's Java API: type Doc (id LONG, embedding ARRAY_OF_FLOATS, ONE bucket so there is exactly one LSMVectorIndex), CREATE INDEX ... LSM_VECTOR METADATA {...} once at build (DDL, never in a
 *  timed region), then the search is a DIRECT call on the LSMVectorIndex object: no SQL parser, no result set, no record load (RIDs come back; ids() maps them untimed for recall).
 *  Search API (recorded in the INFO line): when the engine has findNeighbors(float[],k,efSearch,rerankOversample,allowed,guard) (26.11.1 line, ArcadeData/arcadedb#10545) it is used for
 *  every arm (the entry point SQL vectorNeighbors, Cypher and the select share); on the pinned 26.10.2 engine SQL vectorNeighbors answers every quantization with
 *  findNeighborsFromVector(q, k, efSearch) (the exact walk; bindings/python vector.py says the same), so that is called; PRODUCT with rr = 0 calls findNeighborsFromVectorApproximate(q, k)
 *  (PQ scores alone, no efSearch on that engine). Index parameters are the engine defaults (the metadata dump in the INFO line shows them). */
final class AdArm {
  private AdArm() { }
  static final String TYPE = "Doc", IDX = "Doc[embedding]";

  static final class Handle { Database db; DatabaseFactory factory; Path path; double ingestS; }

  static Handle ingest(final Arms.Spec s, final U.Data d, final Path dbPath) throws Exception {
    U.deleteTree(dbPath); Files.createDirectories(dbPath.getParent());
    final Handle h = new Handle(); h.path = dbPath; h.factory = new DatabaseFactory(dbPath.toString()); h.db = h.factory.create();
    final Database db = h.db;
    db.command("sql", "CREATE DOCUMENT TYPE " + TYPE); db.command("sql", "CREATE PROPERTY Doc.id LONG"); db.command("sql", "CREATE PROPERTY Doc.embedding ARRAY_OF_FLOATS");
    final long t = System.nanoTime();
    for (int b = 0; b < d.n; b += 5000) {
      db.begin();
      for (int i = b; i < Math.min(d.n, b + 5000); i++) { final MutableDocument doc = db.newDocument(TYPE); doc.set("id", (long) i); doc.set("embedding", d.base[i]); doc.save(); }
      db.commit();
    }
    h.ingestS = (System.nanoTime() - t) / 1e9;
    return h;
  }

  static String metadataJson(final Arms.Spec s, final U.Data d) {
    final String q = s.kind.equals("NONE") ? "NONE" : s.kind.equals("INT8") ? "INT8" : s.kind.equals("BINARY") ? "BINARY" : "PRODUCT";
    String md = "{\"dimensions\": " + d.dim + ", \"similarity\": \"" + d.similarity + "\", \"quantization\": \"" + q + "\"";
    if (q.equals("PRODUCT")) md += ", \"pqSubspaces\": " + s.subs + ", \"pqClusters\": 256";
    if (s.svg) md += ", \"storeVectorsInGraph\": true";   // ad-NONE-SVG only (README "storeVectorsInGraph arm"): LSMVectorIndexMetadata key, vectors inline in the graph file
    return md + "}";
  }

  static LSMVectorIndex lsm(final Database db) {
    final Index idx = db.getSchema().getIndexByName(IDX);
    if (!(idx instanceof TypeIndex)) throw new IllegalStateException("not a TypeIndex: " + idx);
    final Index[] subs = ((TypeIndex) idx).getIndexesOnBuckets();
    if (subs.length != 1) throw new IllegalStateException("expected exactly one bucket index, got " + subs.length);
    return (LSMVectorIndex) subs[0];
  }

  /** CREATE INDEX + buildVectorGraphIfPending, then close. peak = outer sampler or null. CREATE INDEX over loaded rows already builds the graph;
   *  until 2026-10-28 the second step was buildVectorGraphNow, which always rebuilds, so every ArcadeDB build_s before then counts two graph builds. */
  static Map<String, Object> index(final Arms.Spec s, final U.Data d, final Handle h, final U.Peak outer) throws Exception {
    try (U.Peak own = outer == null ? new U.Peak() : null) {
      final U.Peak peak = outer != null ? outer : own;
      long t = System.nanoTime(); h.db.command("sql", "CREATE INDEX ON Doc (embedding) LSM_VECTOR METADATA " + metadataJson(s, d)); final double createS = (System.nanoTime() - t) / 1e9;
      t = System.nanoTime(); final LSMVectorIndex idx = lsm(h.db); final boolean rebuilt = idx.buildVectorGraphIfPending(null); final double graphS = (System.nanoTime() - t) / 1e9;
      final long cnt = idx.countEntries(); if (cnt != d.n) throw new IllegalStateException("index holds " + cnt + " of " + d.n);
      h.db.close(); h.factory.close();
      final Map<String, Object> info = new LinkedHashMap<>();
      info.put("ingest_s", U.r(h.ingestS, 2)); info.put("create_index_s", U.r(createS, 2)); info.put("build_graph_now_s", U.r(graphS, 2)); info.put("graph_pending_after_create", rebuilt); info.put("build_s", U.r(createS + graphS, 2));
      final SortedMap<String, Long> by = U.dirBytesByExt(h.path);
      info.put("db_bytes", by.values().stream().mapToLong(Long::longValue).sum()); info.put("files", U.byExtJson(by)); info.put("index_bytes", U.indexBytes(by)); info.put("fp32_bytes", (long) d.n * d.dim * 4);
      info.put("peak_heap_mb", peak.heapMb()); info.put("peak_rss_mb", peak.rssMb()); info.put("vmhwm_mb", U.hwmMb());
      return info;
    }
  }

  static Map<String, Object> build(final Arms.Spec s, final U.Data d, final Path work) throws Exception {
    try (U.Peak peak = new U.Peak()) {
      final Handle h = ingest(s, d, work.resolve(s.name));
      return index(s, d, h, peak);
    }
  }

  static String metaDump(final Object meta) {
    final List<Object> kv = new ArrayList<>();
    for (final Field f : meta.getClass().getFields()) {
      if (Modifier.isStatic(f.getModifiers())) continue;
      try { final Object v = f.get(meta); if (v instanceof Number || v instanceof Boolean || v instanceof Enum || v instanceof String) { kv.add(f.getName()); kv.add(v instanceof Enum || v instanceof String ? v.toString() : v); } } catch (final Exception e) { }
    }
    return U.json(kv.toArray());
  }

  static Arms.Searcher open(final Arms.Spec s, final U.Data d, final Path work) throws Exception {
    final DatabaseFactory factory = new DatabaseFactory(work.resolve(s.name).toString());
    if (!factory.exists()) throw new IllegalStateException("not built: " + work.resolve(s.name));
    final Database db = factory.open(); final LSMVectorIndex idx = lsm(db);
    Method nf = null;
    try { nf = LSMVectorIndex.class.getMethod("findNeighbors", float[].class, int.class, int.class, int.class, Set.class, WorkGuard.class); } catch (final NoSuchMethodException e) { }
    final Method findNeighbors = nf; final boolean pq = s.kind.equals("PQ");
    final boolean binRr = s.kind.equals("BINARY") && System.getenv("L1BINRR") != null;   // PR #10975: a BINARY index takes rerankOversample; unset = -1 as before
    final String api = nf != null ? "LSMVectorIndex.findNeighbors(q,10,efSearch,rerankOversample,null,guard) (reflective)" : pq ? "LSMVectorIndex.findNeighborsFromVector(q,10,efSearch) [rr=-1]; findNeighborsFromVectorApproximate(q,10) [rr=0]" : "LSMVectorIndex.findNeighborsFromVector(q,10,efSearch)";
    final float[][] queries = d.queries; final Object[] meta = {idx.getMetadata()};
    final Map<RID, Integer> idOf = new HashMap<>();
    return new Arms.Searcher() {
      @Override public Object search(final int qi, final int ef, final int rr) throws Exception {
        final float[] q = queries[qi]; final int e = ef > 0 ? ef : -1;
        if (findNeighbors != null) return findNeighbors.invoke(idx, q, 10, e, pq || binRr ? rr : -1, null, WorkGuard.forCommandDeadline(null));
        if (pq && rr == 0) return idx.findNeighborsFromVectorApproximate(q, 10);
        return ef > 0 ? idx.findNeighborsFromVector(q, 10, ef) : idx.findNeighborsFromVector(q, 10);
      }
      @SuppressWarnings("unchecked") @Override public int[] ids(final Object raw) {
        synchronized (idOf) {
          if (idOf.isEmpty()) { final Iterator<? extends com.arcadedb.database.Record> it = db.iterateType(TYPE, false); while (it.hasNext()) { final Document doc = it.next().asDocument(); idOf.put(doc.getIdentity(), doc.getLong("id").intValue()); } }
        }
        final List<Pair<RID, Float>> l = (List<Pair<RID, Float>>) raw; final int[] out = new int[l.size()]; for (int i = 0; i < out.length; i++) out[i] = idOf.get(l.get(i).getFirst()); return out;
      }
      @Override public int[] rerankChoices(final int[] wanted) { return findNeighbors != null || !pq ? wanted : Arrays.stream(wanted).filter(r -> r <= 0).toArray(); }
      @Override public Object[] info() { return binRr ? new Object[] {"api", api, "has_findNeighbors", findNeighbors != null, "binary_rerank_passed", true, "index_metadata", metaDump(meta[0]), "entries", idx.countEntries()} : new Object[] {"api", api, "has_findNeighbors", findNeighbors != null, "index_metadata", metaDump(meta[0]), "entries", idx.countEntries()}; }
      @Override public void close() { try { db.close(); factory.close(); } catch (final Exception e) { } }
    };
  }
}
