package ch.lolo.nn.layers;

import ch.lolo.nn.Layer;
import ch.lolo.nn.autograd.Variable;

public final class Tanh implements Layer {
    public Variable forward(Variable x, boolean t) {
        return x.tanh();
    }

    public String type() {
        return "Tanh";
    }
}
