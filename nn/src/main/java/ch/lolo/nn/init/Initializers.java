package ch.lolo.nn.init;

import ch.lolo.nn.NN;
import ch.lolo.tensor.Tensor;

public final class Initializers {
    private Initializers() {
    }

    public static Tensor xavier(int in, int out) {
        double lim = Math.sqrt(6.0 / (in + out));
        return Tensor.random(NN.random(), -lim, lim, in, out);
    }

    public static Tensor he(int in, int out) {
        double std = Math.sqrt(2.0 / in);
        return Tensor.generate(i -> NN.random().nextGaussian() * std, in, out);
    }

    public static Tensor zeros(int... s) {
        return Tensor.zeros(s);
    }

    public static Tensor uniform(double lo, double hi, int... s) {
        return Tensor.random(NN.random(), lo, hi, s);
    }

    public static Tensor normal(double mean, double std, int... s) {
        return Tensor.generate(i -> mean + NN.random().nextGaussian() * std, s);
    }
}
