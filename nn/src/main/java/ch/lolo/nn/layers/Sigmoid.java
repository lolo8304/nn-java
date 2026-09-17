package ch.lolo.nn.layers;

import ch.lolo.nn.Layer;
import ch.lolo.nn.autograd.Variable;

public final class Sigmoid implements Layer {
    public Variable forward(Variable x, boolean t) {
        return x.sigmoid();
    }

    public String type() {
        return "Sigmoid";
    }
}
