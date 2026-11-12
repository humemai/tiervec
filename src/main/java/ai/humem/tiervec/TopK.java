package ai.humem.tiervec;

/**
 * Bounded selection of the best cap entries: a binary heap whose root is the WORST kept entry. Order: higher score is better, ties broken
 * by the smaller id (so results are deterministic). Fixed arrays, no allocation per offered vector.
 */
final class TopK {
    final int cap;
    final float[] s;
    final long[] id;
    final int[] slot;
    int n;

    TopK(int cap) {
        this.cap = Math.max(cap, 1);
        s = new float[this.cap];
        id = new long[this.cap];
        slot = new int[this.cap];
    }

    /** true when entry 1 is worse than entry 2 */
    static boolean worse(float s1, long id1, float s2, long id2) {
        return s1 < s2 || (s1 == s2 && id1 > id2);
    }

    /** a cheap test for the scan loop: could this score possibly enter? (false positives are fine, offer re-checks) */
    boolean full() {
        return n == cap;
    }

    float rootScore() {
        return s[0];
    }

    void offer(float sc, long ident, int sl) {
        if (n < cap) {
            int i = n++;
            while (i > 0) {
                int p = (i - 1) >> 1;
                if (worse(sc, ident, s[p], id[p])) {
                    s[i] = s[p];
                    id[i] = id[p];
                    slot[i] = slot[p];
                    i = p;
                } else break;
            }
            s[i] = sc;
            id[i] = ident;
            slot[i] = sl;
        } else if (worse(s[0], id[0], sc, ident)) {
            siftDown(0, sc, ident, sl, n);
        }
    }

    private void siftDown(int i, float sc, long ident, int sl, int size) {
        int half = size >> 1;
        while (i < half) {
            int c = 2 * i + 1;
            int r = c + 1;
            if (r < size && worse(s[r], id[r], s[c], id[c])) c = r;
            if (worse(s[c], id[c], sc, ident)) {
                s[i] = s[c];
                id[i] = id[c];
                slot[i] = slot[c];
                i = c;
            } else break;
        }
        s[i] = sc;
        id[i] = ident;
        slot[i] = sl;
    }

    /** empties the heap; returns the entries best first as Hits */
    Hit[] drainSorted() {
        int m = n;
        Hit[] out = new Hit[m];
        for (int i = m - 1; i >= 0; i--) {
            out[i] = new Hit(id[0], s[0]);
            n--;
            if (n > 0) {
                float ls = s[n];
                long li = id[n];
                int lsl = slot[n];
                siftDown(0, ls, li, lsl, n);
            }
        }
        return out;
    }
}
