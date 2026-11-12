package ai.humem.tiervec;

import java.util.Arrays;

/** Binary min-heap on a float key with an int payload (primitive, reusable). For a max-heap push the negated key. */
final class Heap {
    float[] k = new float[64];
    int[] v = new int[64];
    int n;

    void clear() { n = 0; }
    int size() { return n; }
    float topKey() { return k[0]; }
    int topVal() { return v[0]; }

    void push(float key, int val) {
        if (n == k.length) { k = Arrays.copyOf(k, n * 2); v = Arrays.copyOf(v, n * 2); }
        int i = n++;
        while (i > 0) {
            int p = (i - 1) >>> 1;
            if (k[p] <= key) break;
            k[i] = k[p]; v[i] = v[p];
            i = p;
        }
        k[i] = key; v[i] = val;
    }

    void pop() {
        int last = --n;
        if (last == 0) return;
        float key = k[last]; int val = v[last];
        int i = 0;
        for (;;) {
            int c = 2 * i + 1;
            if (c >= last) break;
            if (c + 1 < last && k[c + 1] < k[c]) c++;
            if (k[c] >= key) break;
            k[i] = k[c]; v[i] = v[c];
            i = c;
        }
        k[i] = key; v[i] = val;
    }
}
