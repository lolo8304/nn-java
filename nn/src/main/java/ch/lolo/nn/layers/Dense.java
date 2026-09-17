package ch.lolo.nn.layers;

import ch.lolo.nn.Layer;
import ch.lolo.nn.Parameter;
import ch.lolo.nn.autograd.Variable;
import ch.lolo.nn.init.Initializers;
import ch.lolo.tensor.Tensor;

import java.util.List;

public final class Dense implements Layer {
    private final int in, out;
    private final Parameter weight, bias;

    public Dense(int in, int out) {
        this(in, out, Initializers.xavier(in, out), Tensor.zeros(out));
    }

    public Dense(int in, int out, Tensor w, Tensor b) {
        this.in = in;
        this.out = out;
        weight = new Parameter(w);
        bias = new Parameter(b);
    }

    public int inputSize() {
        return in;
    }

    public int outputSize() {
        return out;
    }

    public Parameter weight() {
        return weight;
    }

    public Parameter bias() {
        return bias;
    }

    public Variable forward(Variable x, boolean training) {
        return x.matmul(weight).add(bias);
    }

    public List<Parameter> parameters() {
        return List.of(weight, bias);
    }

    public String type() {
        return "Dense";
    }
}
