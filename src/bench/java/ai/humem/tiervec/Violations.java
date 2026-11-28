package ai.humem.tiervec;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * What the agent-memory benchmark counts, and what fails a run (issue #159). Every check compares an answer with the live rows, which are
 * the state after the last write that returned. A run fails on each kind, with one exception: a batch seen in part does not fail an arm that
 * does not promise atomic batches.
 */
final class Violations {
    enum Kind {
        /** a write that a search which started after it returned does not see */
        WRITE_NOT_SEEN("writeNotSeen"),
        /** an id that is not live was returned: deleted, or never added */
        DELETED_RETURNED("deletedReturned"),
        /** an id was returned with a score that is not the one of its current vector */
        OLD_SCORE("oldScore"),
        /** a search saw some writes of a batch and not others */
        PART_OF_BATCH("partOfBatch"),
        /** fewer than k different ids although k live rows qualify */
        TOO_FEW_HITS("tooFewHits");

        final String key;

        Kind(String key) { this.key = key; }
    }

    /** the examples kept per kind, so that the many batches in part of an arm without atomic batches cannot hide another kind */
    private static final int EXAMPLES_PER_KIND = 3;

    private final long[] counts = new long[Kind.values().length];
    final List<String> examples = new ArrayList<>();

    long count(Kind kind) { return counts[kind.ordinal()]; }

    void add(Kind kind, String what) {
        if (counts[kind.ordinal()]++ < EXAMPLES_PER_KIND) examples.add(kind.key + ": " + what);
    }

    /** the violations that fail a run of an arm that promises {@code k} ids */
    long failing(boolean atomicBatches) { return failing(atomicBatches, true); }

    /** the violations that fail a run of an arm, by the guarantees that it gives */
    long failing(boolean atomicBatches, boolean fullAnswers) {
        long n = 0;
        for (Kind k : Kind.values()) {
            boolean promised = (k != Kind.PART_OF_BATCH || atomicBatches) && (k != Kind.TOO_FEW_HITS || fullAnswers);
            if (promised) n += counts[k.ordinal()];
        }
        return n;
    }

    /** the answer of an unfiltered search for the {@code k} best */
    void searchAnswer(LiveRows live, float[] q, Hit[] hits, int k, String where) {
        Set<Long> seen = new HashSet<>();
        for (Hit h : hits) {
            seen.add(h.id());
            checkHit(live, q, h, where);
        }
        int expected = Math.min(k, live.size());
        if (seen.size() < expected) add(Kind.TOO_FEW_HITS, where + " returned " + seen.size() + " of " + expected + " ids");
    }

    /**
     * The answer of a search that allows only {@code ids} and asks for as many hits as there are ids. It is on the exact path of every arm
     * (a {@link SizedIdFilter} of that size), so every live id of the set has to be in it.
     */
    void probeAnswer(LiveRows live, float[] q, long[] ids, Hit[] hits, String where) {
        Set<Long> allowed = new HashSet<>();
        for (long id : ids) allowed.add(id);
        Set<Long> seen = new HashSet<>();
        for (Hit h : hits) {
            if (!allowed.contains(h.id())) throw new IllegalStateException(where + ": the search returned the id " + h.id() + ", which its filter does not allow");
            seen.add(h.id());
            checkHit(live, q, h, where);
        }
        for (long id : allowed) if (live.holds(id) && !seen.contains(id)) add(Kind.WRITE_NOT_SEEN, where + " does not see the live id " + id);
    }

    private void checkHit(LiveRows live, float[] q, Hit h, String where) {
        float[] v = live.vector(h.id());
        if (v == null) {
            add(Kind.DELETED_RETURNED, where + " returned the id " + h.id() + ", which is not live");
        } else if (!sameScore(h.score(), q, v)) {
            add(Kind.OLD_SCORE, where + " returned the id " + h.id() + " with score " + h.score() + ", its current vector scores " + LiveRows.cosine(q, v));
        }
    }

    private static boolean sameScore(double score, float[] q, float[] v) {
        return Math.abs(score - LiveRows.cosine(q, v)) <= LiveRows.TOLERANCE;
    }

    /**
     * Whether an answer shows a batch in part. {@code ids} are the ids of the batch, {@code before[i]} and {@code after[i]} the vector of
     * {@code ids[i]} before and after the batch (null: not live), and the answer is the one of a search that allows only these ids. It is in
     * part when one id shows the state before the batch and another the state after it, or when an id shows neither.
     */
    static boolean showsPart(float[] q, long[] ids, float[][] before, float[][] after, Hit[] hits) {
        boolean sawBefore = false, sawAfter = false;
        for (int i = 0; i < ids.length; i++) {
            Hit hit = null;
            for (Hit h : hits) if (h.id() == ids[i]) hit = h;
            boolean isBefore = shows(hit, q, before[i]), isAfter = shows(hit, q, after[i]);
            if (!isBefore && !isAfter) return true;
            sawBefore |= isBefore && !isAfter;
            sawAfter |= isAfter && !isBefore;
        }
        return sawBefore && sawAfter;
    }

    private static boolean shows(Hit hit, float[] q, float[] state) {
        return state == null ? hit == null : hit != null && sameScore(hit.score(), q, state);
    }

    /** a filter that allows exactly these ids and says how many they are */
    static SizedIdFilter only(long[] ids) {
        Set<Long> set = new HashSet<>();
        for (long id : ids) set.add(id);
        return new SizedIdFilter() {
            @Override public boolean test(long id) { return set.contains(id); }
            @Override public long allowedCount() { return set.size(); }
        };
    }
}
