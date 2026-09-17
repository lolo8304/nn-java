package ch.lolo.nn.metrics;

import ch.lolo.tensor.Tensor;

public final class BinaryAccuracy {
    public double compute(Tensor p, Tensor y) {
        double[] a = p.toArray(), b = y.toArray();
        if (a.length != b.length) throw new IllegalArgumentException();
        int ok = 0;
        for (int i = 0; i < a.length; i++) if ((a[i] >= .5) == (b[i] >= .5)) ok++;
        return ok / (double) a.length;
    }
}
