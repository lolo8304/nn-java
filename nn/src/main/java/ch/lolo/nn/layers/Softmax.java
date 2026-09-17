package ch.lolo.nn.layers;

import ch.lolo.nn.Layer;
import ch.lolo.nn.autograd.Variable;

public record Softmax(int axis) implements Layer {
    public Softmax() {
        this(-1);
    }

    public Variable forward(Variable x, boolean t) {
        return x.softmax(axis);
    }

    public String type() {
        return "Softmax";
    }
}
