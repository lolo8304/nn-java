package ch.lolo.nn.optim;

import ch.lolo.nn.Parameter;
import ch.lolo.tensor.Tensor;

import java.util.IdentityHashMap;
import java.util.List;

public final class SGD implements Optimizer {
    private double lr, momentum;
    private final IdentityHashMap<Parameter, Tensor> velocity = new IdentityHashMap<>();

    public SGD(double lr) {
        this(lr, 0);
    }

    public SGD(double lr, double m) {
        this.lr = lr;
        momentum = m;
    }

    private static void writeTensor(java.io.DataOutput o, Tensor t) throws java.io.IOException {
        o.writeInt(t.rank());
        for (int x : t.shape()) o.writeInt(x);
        for (double x : t.toArray()) o.writeDouble(x);
    }

    private static Tensor readTensor(java.io.DataInput in) throws java.io.IOException {
        int r = in.readInt();
        int[] s = new int[r];
        for (int i = 0; i < r; i++) s[i] = in.readInt();
        double[] a = new double[(int) java.util.Arrays.stream(s).asLongStream().reduce(1, (x, y) -> x * y)];
        for (int i = 0; i < a.length; i++) a[i] = in.readDouble();
        final int[] q = {0};
        return Tensor.generate(ix -> a[q[0]++], s);
    }

    public double learningRate() {
        return lr;
    }

    public double momentum() {
        return momentum;
    }

    public void step(List<Parameter> ps) {
        for (Parameter p : ps) {
            if (p.grad() == null) continue;
            Tensor v = velocity.computeIfAbsent(p, ignored -> Tensor.zeros(p.value().shape()));
            p.value().sgdStep(p.grad(), v, lr, momentum);
        }
    }

    public String type() {
        return "SGD";
    }

    public void writeState(java.io.DataOutput o, List<Parameter> ps) throws java.io.IOException {
        o.writeDouble(lr);
        o.writeDouble(momentum);
        for (Parameter p : ps) writeTensor(o, velocity.getOrDefault(p, Tensor.zeros(p.value().shape())));
    }

    public void readState(java.io.DataInput in, List<Parameter> ps) throws java.io.IOException {
        lr = in.readDouble();
        momentum = in.readDouble();
        velocity.clear();
        for (Parameter p : ps) velocity.put(p, readTensor(in));
    }
}
