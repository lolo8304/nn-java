package ch.lolo.tensor;

import java.util.function.Consumer;

final class TensorIterator {
    private TensorIterator() {
    }

    static void each(int[] s, Consumer<int[]> c) {
        if (s.length == 0) {
            c.accept(new int[0]);
            return;
        }
        for (int d : s) if (d == 0) return;
        int[] x = new int[s.length];
        while (true) {
            c.accept(x.clone());
            int d = s.length - 1;
            while (d >= 0 && ++x[d] == s[d]) {
                x[d] = 0;
                d--;
            }
            if (d < 0) return;
        }
    }
}
