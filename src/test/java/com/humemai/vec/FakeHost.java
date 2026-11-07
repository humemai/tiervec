package com.humemai.vec;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * A host for the epoch and replay tests. It holds the records (the source of truth), numbers its transactions, mints ids that increase,
 * keeps a log of committed transactions of which a prefix is durable, and can lose the rest in a simulated crash. After a crash it mints the
 * lost ids again, for other vectors: that is the case the durable-prefix rule exists for.
 *
 * <p>A host that calls the tier from several threads holds the monitor of this object around {@code commit} and the call into the tier, which
 * is how a real host serializes {@code apply} per index.
 */
final class FakeHost implements RecordSource {
    /** one committed transaction: the adds, then the deletes */
    record Txn(List<TieredIndex.IdVector> adds, List<Long> deletes) {}

    final int dim;
    final long seed;
    private long epoch;
    private final List<Txn> log = new ArrayList<>();
    private int durable;                 // transactions of the log that survive a crash
    private long nextId = 1;
    private long serial;                 // never rewound: an id that is minted again gets another vector
    private final TreeMap<Long, float[]> records = new TreeMap<>();
    private final TreeSet<Long> everMinted = new TreeSet<>();
    private boolean pinned;              // after a crash the recovered maximum is the one found at recovery, whatever the host mints before it opens the index
    private long pinnedMax;

    FakeHost(int dim, long seed, long epoch) {
        this.dim = dim;
        this.seed = seed;
        this.epoch = epoch;
    }

    /** no automatic seal, and a graph from 10 vectors on, so that a rebuild of a few dozen records makes a segment */
    static TieredIndex.Config cfg() {
        return new TieredIndex.Config(1_000_000, 0L, 10, 8, 4, 0.3, 0, 4, Quant.NONE, GraphIndex.Params.defaults().withEfConstruction(64), 1);
    }

    static TieredIndex.Batch batch(Txn t) {
        return new TieredIndex.Batch(t.adds(), t.deletes());
    }

    /** Commits a transaction: {@code adds} new records with new ids and new vectors, and the deletion of the listed ids (those that are live; the others are skipped). */
    synchronized Txn commit(int adds, long... deletes) {
        List<TieredIndex.IdVector> a = new ArrayList<>();
        for (int i = 0; i < adds; i++) {
            long id = nextId++;
            a.add(new TieredIndex.IdVector(id, TierT.vec(seed, serial++, dim)));
            everMinted.add(id);
        }
        List<Long> d = new ArrayList<>();
        for (long id : deletes) {
            if (records.containsKey(id) && !d.contains(id)) d.add(id);
        }
        Txn t = new Txn(a, d);
        log.add(t);
        applyToRecords(t);
        return t;
    }

    /** a transaction of a few adds and a few deletes of random live ids */
    synchronized Txn commitRandom(Random r) {
        List<Long> ids = new ArrayList<>(records.keySet());
        Collections.shuffle(ids, r);
        int nd = Math.min(ids.size(), r.nextInt(3));
        long[] del = new long[nd];
        for (int i = 0; i < nd; i++) del[i] = ids.get(i);
        return commit(1 + r.nextInt(3), del);
    }

    private void applyToRecords(Txn t) {
        for (TieredIndex.IdVector iv : t.adds()) records.put(iv.id(), iv.v());
        for (long id : t.deletes()) records.remove(id);
    }

    synchronized Txn txn(int i) {
        return log.get(i);
    }

    /** the highest id minted so far and every live record: what a host hands to a rebuild */
    synchronized TieredIndex.Snapshot snapshot() {
        return new TieredIndex.Snapshot(nextId - 1, liveFrom(Long.MIN_VALUE));
    }

    /** every transaction so far survives a crash */
    synchronized void flush() {
        durable = log.size();
    }

    /** the log up to and including the last transaction that deletes something survives a crash: the host flushes its deletes before it lets the library checkpoint */
    synchronized void flushDeletes() {
        for (int i = log.size() - 1; i >= 0; i--) {
            if (!log.get(i).deletes().isEmpty()) {
                durable = Math.max(durable, i + 1);
                return;
            }
        }
    }

    /**
     * The host forgets every transaction after the durable prefix, and mints the ids above its recovered maximum again. The recovered maximum is the
     * highest id of what it kept, and stays that value, whatever the host mints before the index is opened, until {@link #restart}.
     */
    synchronized void crash() {
        while (log.size() > durable) log.remove(log.size() - 1);
        records.clear();
        for (Txn t : log) applyToRecords(t);
        long max = maxMinted();
        nextId = (max == Manifest.NONE ? 0 : max) + 1;
        pinnedMax = max;
        pinned = true;
    }

    /** a restart that loses nothing: the recovered maximum is the highest id of the log again */
    synchronized void restart() {
        pinned = false;
    }

    private long maxMinted() {
        long max = Manifest.NONE;
        for (Txn t : log) {
            for (TieredIndex.IdVector iv : t.adds()) max = Math.max(max, iv.id());
        }
        return max;
    }

    /** a compaction that renumbers: the live records are reissued in rank order from 1 under a new epoch, so that id 7 may now be another vector */
    synchronized void renumber(long newEpoch) {
        List<float[]> vs = new ArrayList<>(records.values());
        records.clear();
        log.clear();
        everMinted.clear();
        List<TieredIndex.IdVector> a = new ArrayList<>();
        long id = 1;
        for (float[] v : vs) {
            a.add(new TieredIndex.IdVector(id, v));
            records.put(id, v);
            everMinted.add(id);
            id++;
        }
        log.add(new Txn(a, List.of()));
        durable = 1;
        nextId = id;
        epoch = newEpoch;
        pinned = false;
    }

    synchronized Map<Long, float[]> live() {
        return new TreeMap<>(records);
    }

    private List<TieredIndex.IdVector> liveFrom(long above) {
        List<TieredIndex.IdVector> out = new ArrayList<>();
        for (Map.Entry<Long, float[]> e : records.tailMap(above, false).entrySet()) out.add(new TieredIndex.IdVector(e.getKey(), e.getValue()));
        return out;
    }

    @Override
    public synchronized long epoch() {
        return epoch;
    }

    @Override
    public synchronized long recoveredMax() {
        return pinned ? pinnedMax : maxMinted();
    }

    @Override
    public synchronized boolean isLive(long id) {
        return records.containsKey(id);
    }

    @Override
    public synchronized Iterable<TieredIndex.IdVector> liveAbove(long id) {
        return liveFrom(id);
    }

    @Override
    public synchronized Iterable<TieredIndex.IdVector> allLive() {
        return liveFrom(Long.MIN_VALUE);
    }

    /**
     * The tier holds exactly the host's live records: its size, every live id found by its own vector with the exact score of that vector (a
     * stale vector under the same id gives a lower score), and no id that the host minted and does not hold now is returned.
     */
    static void assertMatches(TieredIndex t, FakeHost h, Metric metric, String what) {
        Map<Long, float[]> live = h.live();
        assertEquals(live.size(), t.size(), what + ": number of live ids");
        for (Map.Entry<Long, float[]> e : live.entrySet()) {
            final long target = e.getKey();
            Hit[] r = t.search(e.getValue(), 1, id -> id == target);
            assertEquals(1, r.length, what + ": live id " + target + " is not found");
            assertEquals(target, r[0].id(), what + ": id");
            assertEquals(TierT.score(metric, e.getValue(), e.getValue()), r[0].score(), 1e-3f, what + ": id " + target + " holds another vector than the host's");
        }
        float[] probe = TierT.vec(h.seed, 987_654_321L, h.dim);
        TreeSet<Long> dead;
        synchronized (h) {
            dead = new TreeSet<>(h.everMinted);
        }
        dead.removeAll(live.keySet());
        for (long id : dead) {
            Hit[] r = t.search(probe, 1, x -> x == id);
            assertEquals(0, r.length, what + ": id " + id + " is not live in the host but the tier returns it");
        }
        for (Hit hit : t.search(probe, 10, null)) {
            assertTrue(live.containsKey(hit.id()), what + ": a plain search returned id " + hit.id() + ", which the host does not hold");
        }
    }
}
