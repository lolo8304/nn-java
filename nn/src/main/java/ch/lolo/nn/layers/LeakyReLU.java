package ch.lolo.nn.layers;

import ch.lolo.nn.Layer;
import ch.lolo.nn.autograd.Variable;

public record LeakyReLU(double alpha) implements Layer {
    public LeakyReLU() {
        this(.01);
    }

    public Variable forward(Variable x, boolean t) {
        return x.leakyRelu(alpha);
    }

    public String type() {
        return "LeakyReLU";
    }
}
