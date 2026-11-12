package ai.humem.tiervec;

import java.util.Arrays;

/** Open-addressing long to non-negative int map (linear probing, backward-shift delete). NOT thread-safe: callers lock. -1 means absent. */
final class LongIntMap {
    private long[] keys = new long[16];
    private int[] vals = new int[16];
    private int size;

    LongIntMap() {
        Arrays.fill(vals, -1);
    }

    private static int hash(long k) {
        k ^= k >>> 33;
        k *= 0xff51afd7ed558ccdL;
        k ^= k >>> 33;
        k *= 0xc4ceb9fe1a85ec53L;
        k ^= k >>> 33;
        return (int) k;
    }

    int size() {
        return size;
    }

    int get(long k) {
        int mask = keys.length - 1;
        int i = hash(k) & mask;
        while (vals[i] != -1) {
            if (keys[i] == k) return vals[i];
            i = (i + 1) & mask;
        }
        return -1;
    }

    void put(long k, int v) {
        if ((size + 1) * 2 > keys.length) grow();
        int mask = keys.length - 1;
        int i = hash(k) & mask;
        while (vals[i] != -1) {
            if (keys[i] == k) {
                vals[i] = v;
                return;
            }
            i = (i + 1) & mask;
        }
        keys[i] = k;
        vals[i] = v;
        size++;
    }

    /** @return the old value, or -1 */
    int remove(long k) {
        int mask = keys.length - 1;
        int i = hash(k) & mask;
        while (vals[i] != -1 && keys[i] != k) i = (i + 1) & mask;
        if (vals[i] == -1) return -1;
        int old = vals[i];
        int j = i;
        while (true) {
            j = (j + 1) & mask;
            if (vals[j] == -1) break;
            int h = hash(keys[j]) & mask;
            boolean move = (i <= j) ? (h <= i || h > j) : (h <= i && h > j);
            if (move) {
                keys[i] = keys[j];
                vals[i] = vals[j];
                i = j;
            }
        }
        vals[i] = -1;
        size--;
        return old;
    }

    void clear() {
        keys = new long[16];
        vals = new int[16];
        Arrays.fill(vals, -1);
        size = 0;
    }

    private void grow() {
        long[] ok = keys;
        int[] ov = vals;
        keys = new long[ok.length * 2];
        vals = new int[ok.length * 2];
        Arrays.fill(vals, -1);
        int mask = keys.length - 1;
        for (int j = 0; j < ok.length; j++) {
            if (ov[j] == -1) continue;
            int i = hash(ok[j]) & mask;
            while (vals[i] != -1) i = (i + 1) & mask;
            keys[i] = ok[j];
            vals[i] = ov[j];
        }
    }

    long bytes() {
        return 16L * 2 + 16 + keys.length * 8L + vals.length * 4L;
    }
}
