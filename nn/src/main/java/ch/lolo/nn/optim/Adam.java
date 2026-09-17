package ch.lolo.nn.optim;

import ch.lolo.nn.Parameter;
import ch.lolo.tensor.Tensor;

import java.util.IdentityHashMap;
import java.util.List;

public final class Adam implements Optimizer {
    private final double lr, b1, b2, eps;
    private final IdentityHashMap<Parameter, Tensor> m = new IdentityHashMap<>(), v = new IdentityHashMap<>();
    private int t;

    public Adam(double lr) {
        this(lr, .9, .999, 1e-8);
    }

    public Adam(double lr, double b1, double b2, double eps) {
        this.lr = lr;
        this.b1 = b1;
        this.b2 = b2;
        this.eps = eps;
    }

    private static void writeTensor(java.io.DataOutput o, Tensor t) throws java.io.IOException {
        o.writeInt(t.rank());
        for (int x : t.shape()) o.writeInt(x);
        for (double x : t.toArray()) o.writeDouble(x);
    }

    private static Tensor readTensor(java.io.DataInput in) throws java.io.IOException {
        int r = in.readInt();
        int[] s = new int[r];
        int n = 1;
        for (int i = 0; i < r; i++) {
            s[i] = in.readInt();
            n *= s[i];
        }
        double[] a = new double[n];
        for (int i = 0; i < n; i++) a[i] = in.readDouble();
        final int[] q = {0};
        return Tensor.generate(ix -> a[q[0]++], s);
    }

    public double learningRate() {
        return lr;
    }

    public int stepCount() {
        return t;
    }

    public void step(List<Parameter> ps) {
        t++;
        for (Parameter p : ps) {
            Tensor g = p.grad();
            if (g == null) continue;
            Tensor mm = m.getOrDefault(p, Tensor.zeros(g.shape())).multiply(b1).add(g.multiply(1 - b1));
            Tensor vv = v.getOrDefault(p, Tensor.zeros(g.shape())).multiply(b2).add(g.pow(2).multiply(1 - b2));
            m.put(p, mm);
            v.put(p, vv);
            double c1 = 1 - Math.pow(b1, t), c2 = 1 - Math.pow(b2, t);
            Tensor delta = Tensor.generate(i -> -lr * (mm.get(i) / c1) / (Math.sqrt(vv.get(i) / c2) + eps), g.shape());
            p.applyDelta(delta);
        }
    }

    public String type() {
        return "Adam";
    }

    public void writeState(java.io.DataOutput o, List<Parameter> ps) throws java.io.IOException {
        o.writeDouble(lr);
        o.writeDouble(b1);
        o.writeDouble(b2);
        o.writeDouble(eps);
        o.writeInt(t);
        for (Parameter p : ps) {
            writeTensor(o, m.getOrDefault(p, Tensor.zeros(p.value().shape())));
            writeTensor(o, v.getOrDefault(p, Tensor.zeros(p.value().shape())));
        }
    }

    public void readState(java.io.DataInput in, List<Parameter> ps) throws java.io.IOException {
        in.readDouble();
        in.readDouble();
        in.readDouble();
        in.readDouble();
        t = in.readInt();
        for (Parameter p : ps) {
            m.put(p, readTensor(in));
            v.put(p, readTensor(in));
        }
    }
}
