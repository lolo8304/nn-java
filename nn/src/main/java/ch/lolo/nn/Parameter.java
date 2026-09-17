package ch.lolo.nn;

import ch.lolo.nn.autograd.Variable;
import ch.lolo.tensor.Tensor;

public final class Parameter extends Variable {
    public Parameter(Tensor value) {
        super(value, true);
    }

    private static void each(int[] s, java.util.function.Consumer<int[]> c) {
        if (s.length == 0) {
            c.accept(new int[0]);
            return;
        }
        int[] i = new int[s.length];
        while (true) {
            c.accept(i.clone());
            int a = s.length - 1;
            while (a >= 0 && ++i[a] >= s[a]) {
                i[a] = 0;
                a--;
            }
            if (a < 0) return;
        }
    }

    public void applyDelta(Tensor d) {
        int[] sh = value().shape();
        each(sh, i -> value().set(value().get(i) + d.get(i), i));
    }
}
