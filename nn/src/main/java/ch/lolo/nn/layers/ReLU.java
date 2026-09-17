package ch.lolo.nn.layers;

import ch.lolo.nn.Layer;
import ch.lolo.nn.autograd.Variable;

public final class ReLU implements Layer {
    public Variable forward(Variable x, boolean t) {
        return x.relu();
    }

    public String type() {
        return "ReLU";
    }
}
