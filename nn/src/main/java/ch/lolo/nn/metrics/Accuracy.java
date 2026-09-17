package ch.lolo.nn.metrics;

import ch.lolo.tensor.Tensor;

public final class Accuracy {
    public double compute(Tensor p, Tensor y) {
        if (p.rank() != 2 || y.rank() != 2) throw new IllegalArgumentException("expected [batch,classes]");
        int ok = 0;
        for (int i = 0; i < p.shape()[0]; i++) if (argmax(p.slice(0, i)) == argmax(y.slice(0, i))) ok++;
        return ok / (double) p.shape()[0];
    }

    private int argmax(Tensor t) {
        int best = 0;
        for (int i = 1; i < t.shape()[0]; i++) if (t.get(i) > t.get(best)) best = i;
        return best;
    }
}
