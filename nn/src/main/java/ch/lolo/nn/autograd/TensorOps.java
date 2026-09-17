package ch.lolo.nn.autograd;

import ch.lolo.tensor.Tensor;

final class TensorOps {
    static void each(int[] shape, java.util.function.Consumer<int[]> c) {
        if (shape.length == 0) {
            c.accept(new int[0]);
            return;
        }
        for (int d : shape) if (d == 0) return;
        int[] i = new int[shape.length];
        while (true) {
            c.accept(i.clone());
            int a = shape.length - 1;
            while (a >= 0 && ++i[a] >= shape[a]) {
                i[a] = 0;
                a--;
            }
            if (a < 0) return;
        }
    }

    static Tensor map(Tensor x, java.util.function.DoubleUnaryOperator f) {
        return x.map(f);
    }

    static Tensor unbroadcast(Tensor g, int[] target) {
        Tensor r = g;
        while (r.rank() > target.length) r = r.sum(0);
        for (int a = 0; a < target.length; a++)
            if (target[a] == 1 && r.shape()[a] != 1) {
                r = r.sum(a);
                r = r.reshape(withOne(r.shape(), a));
            }
        return r;
    }

    private static int[] withOne(int[] reduced, int axis) {
        int[] x = new int[reduced.length + 1];
        for (int i = 0, j = 0; i < x.length; i++) x[i] = i == axis ? 1 : reduced[j++];
        return x;
    }

    static Tensor transposeLast2(Tensor t) {
        int[] p = new int[t.rank()];
        for (int i = 0; i < p.length; i++) p[i] = i;
        if (p.length < 2) return t;
        int q = p.length - 1;
        p[q] = q - 1;
        p[q - 1] = q;
        return t.transpose(p);
    }

    static Tensor zerosLike(Tensor t) {
        return Tensor.zeros(t.shape());
    }

    static Tensor onesLike(Tensor t) {
        return Tensor.ones(t.shape());
    }

    static int[] insert(int[] x, int a, int v) {
        int[] r = new int[x.length + 1];
        for (int i = 0, j = 0; i < r.length; i++) r[i] = i == a ? v : x[j++];
        return r;
    }
}
