package ch.lolo.tensor;

import java.util.Arrays;

final class TensorShape {
    private TensorShape() {
    }

    static int sizeOf(int[] s) {
        long n = 1;
        for (int d : s) {
            if (d < 0) throw new IllegalArgumentException("negative dimension");
            n *= d;
            if (n > Integer.MAX_VALUE) throw new IllegalArgumentException("too large");
        }
        return (int) n;
    }

    static int[] strides(int[] s) {
        int[] r = new int[s.length];
        int n = 1;
        for (int i = s.length - 1; i >= 0; i--) {
            r[i] = n;
            n = Math.multiplyExact(n, s[i]);
        }
        return r;
    }

    static boolean contiguous(int[] s, int[] st) {
        int n = 1;
        for (int i = s.length - 1; i >= 0; i--) {
            if (s[i] > 1 && st[i] != n) return false;
            n *= s[i];
        }
        return true;
    }

    static int[] broadcast(int[] a, int[] b) {
        int n = Math.max(a.length, b.length);
        int[] r = new int[n];
        for (int i = 0; i < n; i++) {
            int ai = i - (n - a.length), bi = i - (n - b.length), x = ai < 0 ? 1 : a[ai], y = bi < 0 ? 1 : b[bi];
            if (x != y && x != 1 && y != 1)
                throw new IllegalArgumentException("cannot broadcast " + Arrays.toString(a) + " and " + Arrays.toString(b));
            r[i] = x == 1 ? y : x;
        }
        return r;
    }
}
