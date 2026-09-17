package ch.lolo.nn.layers;

import ch.lolo.nn.Layer;
import ch.lolo.nn.NN;
import ch.lolo.nn.autograd.Variable;
import ch.lolo.tensor.Tensor;

public final class Dropout implements Layer {
    private final double p;

    public Dropout(double p) {
        if (p < 0 || p >= 1) throw new IllegalArgumentException();
        this.p = p;
    }

    public double probability() {
        return p;
    }

    public Variable forward(Variable x, boolean training) {
        if (!training || p == 0) return x;
        double scale = 1 / (1 - p);
        Tensor mask = Tensor.generate(i -> NN.random().nextDouble() >= p ? scale : 0, x.value().shape());
        return x.multiply(Variable.of(mask));
    }

    public String type() {
        return "Dropout";
    }
}
