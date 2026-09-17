package ch.lolo.nn.autograd;

import ch.lolo.tensor.Tensor;

import java.util.*;
import java.util.function.Consumer;

public class Variable {
    private final Tensor value;
    private final boolean requiresGrad;
    private final List<Variable> parents;
    private final Consumer<Tensor> backwardFn;
    private Tensor grad;

    public Variable(Tensor value, boolean requiresGrad) {
        this(value, requiresGrad, List.of(), null);
    }

    private Variable(Tensor v, boolean r, List<Variable> p, Consumer<Tensor> b) {
        value = v;
        requiresGrad = r;
        parents = p;
        backwardFn = b;
    }

    public static Variable of(Tensor t) {
        return new Variable(t, false);
    }

    public static Variable parameter(Tensor t) {
        return new Variable(t, true);
    }

    private boolean recordsGrad() {
        return requiresGrad && GradMode.isEnabled();
    }

    private boolean recordsGrad(Variable other) {
        return (requiresGrad || other.requiresGrad) && GradMode.isEnabled();
    }

    // Callers must check recordsGrad before constructing the parent list and closure.
    private static Variable node(Tensor v, List<Variable> p, Consumer<Tensor> b) {
        return new Variable(v, true, p, b);
    }

    private static void build(Variable v, Set<Variable> s, List<Variable> o) {
        if (!s.add(v)) return;
        for (var p : v.parents) build(p, s, o);
        o.add(v);
    }

    public Tensor value() {
        return value;
    }

    public Tensor grad() {
        return grad;
    }

    public boolean requiresGrad() {
        return requiresGrad;
    }

    public void zeroGrad() {
        grad = null;
    }

    private void addGrad(Tensor g) {
        if (requiresGrad) grad = grad == null ? g.copy() : grad.add(g);
    }

    public Variable add(Variable b) {
        Variable a = this;
        Tensor out = value.add(b.value);
        if (!recordsGrad(b)) return of(out);
        return node(out, List.of(a, b), g -> {
            if (a.requiresGrad) a.addGrad(TensorOps.unbroadcast(g, a.value.shape()));
            if (b.requiresGrad) b.addGrad(TensorOps.unbroadcast(g, b.value.shape()));
        });
    }

    public Variable add(Tensor b) {
        return add(of(b));
    }

    public Variable add(double x) {
        return add(of(Tensor.scalar(x)));
    }

    public Variable subtract(Variable b) {
        Variable a = this;
        Tensor out = value.subtract(b.value);
        if (!recordsGrad(b)) return of(out);
        return node(out, List.of(a, b), g -> {
            if (a.requiresGrad) a.addGrad(TensorOps.unbroadcast(g, a.value.shape()));
            if (b.requiresGrad) b.addGrad(TensorOps.unbroadcast(g.negate(), b.value.shape()));
        });
    }

    public Variable multiply(Variable b) {
        Variable a = this;
        Tensor out = value.multiply(b.value);
        if (!recordsGrad(b)) return of(out);
        return node(out, List.of(a, b), g -> {
            if (a.requiresGrad) a.addGrad(TensorOps.unbroadcast(g.multiply(b.value), a.value.shape()));
            if (b.requiresGrad) b.addGrad(TensorOps.unbroadcast(g.multiply(a.value), b.value.shape()));
        });
    }

    public Variable multiply(double x) {
        return multiply(of(Tensor.scalar(x)));
    }

    public Variable divide(Variable b) {
        Variable a = this;
        Tensor out = value.divide(b.value);
        if (!recordsGrad(b)) return of(out);
        return node(out, List.of(a, b), g -> {
            if (a.requiresGrad) a.addGrad(TensorOps.unbroadcast(g.divide(b.value), a.value.shape()));
            if (b.requiresGrad) b.addGrad(TensorOps.unbroadcast(g.multiply(a.value).divide(b.value.pow(2)).negate(), b.value.shape()));
        });
    }

    public Variable matmul(Variable b) {
        Variable a = this;
        Tensor out = value.matmul(b.value);
        if (!recordsGrad(b)) return of(out);
        return node(out, List.of(a, b), g -> {
            if (a.value.rank() == 2 && b.value.rank() == 2) {
                if (a.requiresGrad) a.addGrad(g.matmul(b.value.transpose()));
                if (b.requiresGrad) b.addGrad(a.value.transpose().matmul(g));
            } else if (a.value.rank() == 1 && b.value.rank() == 1) {
                if (a.requiresGrad) a.addGrad(b.value.multiply(g.scalar()));
                if (b.requiresGrad) b.addGrad(a.value.multiply(g.scalar()));
            } else { // general batched/vector fallback: numerical-shape algebra for >=2D
                if (a.value.rank() >= 2 && b.value.rank() >= 2) {
                    if (a.requiresGrad) a.addGrad(TensorOps.unbroadcast(g.matmul(TensorOps.transposeLast2(b.value)), a.value.shape()));
                    if (b.requiresGrad) b.addGrad(TensorOps.unbroadcast(TensorOps.transposeLast2(a.value).matmul(g), b.value.shape()));
                } else throw new UnsupportedOperationException("autograd matmul vector/matrix mixed not yet supported");
            }
        });
    }

    public Variable sum() {
        Variable a = this;
        Tensor out = value.sum();
        if (!recordsGrad()) return of(out);
        return node(out, List.of(a), g -> a.addGrad(TensorOps.onesLike(a.value).multiply(g.scalar())));
    }

    public Variable mean() {
        Variable a = this;
        Tensor out = value.mean();
        if (!recordsGrad()) return of(out);
        return node(out, List.of(a), g -> a.addGrad(TensorOps.onesLike(a.value).multiply(g.scalar() / a.value.size())));
    }

    public Variable pow(double p) {
        Variable a = this;
        Tensor out = value.pow(p);
        if (!recordsGrad()) return of(out);
        return node(out, List.of(a), g -> a.addGrad(g.multiply(a.value.pow(p - 1)).multiply(p)));
    }

    public Variable log() {
        Variable a = this;
        Tensor y = TensorOps.map(value, Math::log);
        if (!recordsGrad()) return of(y);
        return node(y, List.of(a), g -> a.addGrad(g.divide(a.value)));
    }

    public Variable relu() {
        Variable a = this;
        Tensor y = value.relu();
        if (!recordsGrad()) return of(y);
        return node(y, List.of(a), g -> a.addGrad(g.multiply(Tensor.generate(i -> a.value.get(i) > 0 ? 1 : 0, a.value.shape()))));
    }

    public Variable sigmoid() {
        Variable a = this;
        Tensor y = value.sigmoid();
        if (!recordsGrad()) return of(y);
        return node(y, List.of(a), g -> a.addGrad(g.multiply(y.multiply(Tensor.ones(y.shape()).subtract(y)))));
    }

    public Variable tanh() {
        Variable a = this;
        Tensor y = TensorOps.map(value, Math::tanh);
        if (!recordsGrad()) return of(y);
        return node(y, List.of(a), g -> a.addGrad(g.multiply(Tensor.ones(y.shape()).subtract(y.pow(2)))));
    }

    public Variable leakyRelu(double alpha) {
        Variable a = this;
        Tensor y = TensorOps.map(value, x -> x >= 0 ? x : alpha * x);
        if (!recordsGrad()) return of(y);
        return node(y, List.of(a), g -> a.addGrad(g.multiply(Tensor.generate(i -> a.value.get(i) >= 0 ? 1 : alpha, a.value.shape()))));
    }

    public Variable reshape(int... s) {
        Variable a = this;
        Tensor y = value.reshape(s);
        if (!recordsGrad()) return of(y);
        return node(y, List.of(a), g -> a.addGrad(g.reshape(a.value.shape())));
    }

    public Variable flatten() {
        return reshape((int) value.size());
    }

    public Variable transpose() {
        Variable a = this;
        Tensor y = value.transpose();
        if (!recordsGrad()) return of(y);
        return node(y, List.of(a), g -> a.addGrad(g.transpose()));
    }

    public Variable softmax(int axis) {
        Variable a = this;
        Tensor y = value.softmax(axis);
        if (!recordsGrad()) return of(y);
        return node(y, List.of(a), g -> {
            int ax = axis < 0 ? axis + y.rank() : axis;
            Tensor dx = Tensor.zeros(y.shape());
            int[] sh = y.shape();
            int n = sh[ax];
            int[] outer = new int[sh.length - 1];
            for (int i = 0, j = 0; i < sh.length; i++) if (i != ax) outer[j++] = sh[i];
            TensorOps.each(outer, o -> {
                double dot = 0;
                int[] index = TensorOps.insert(o, ax, 0);
                for (int j = 0; j < n; j++) {
                    index[ax] = j;
                    dot += g.get(index) * y.get(index);
                }
                for (int i = 0; i < n; i++) {
                    index[ax] = i;
                    dx.set(y.get(index) * (g.get(index) - dot), index);
                }
            });
            a.addGrad(dx);
        });
    }

    /**
     * Fused, unsmoothed cross entropy over the last axis. Targets are finite,
     * nonnegative weights of exactly the same shape; they need not sum to one.
     * Sums all positions and divides by the first dimension (or one for a vector).
     */
    public Variable crossEntropy(Tensor target) {
        int[] shape = value.shape();
        if (shape.length == 0 || value.size() == 0 || !Arrays.equals(shape, target.shape()))
            throw new IllegalArgumentException("cross entropy requires matching nonempty nonscalar shapes");
        int classes = shape[shape.length - 1];
        int batch = shape.length > 1 ? shape[0] : 1;
        double[] logits = value.toArray();
        double[] weights = target.toArray();
        boolean record = recordsGrad();
        double[] derivative = record ? new double[logits.length] : null;
        double loss = 0;
        for (int row = 0; row < logits.length; row += classes) {
            double max = Double.NEGATIVE_INFINITY;
            double mass = 0;
            for (int j = row; j < row + classes; j++) {
                if (!Double.isFinite(logits[j]) || !Double.isFinite(weights[j]) || weights[j] < 0)
                    throw new IllegalArgumentException("cross entropy requires finite logits and finite nonnegative targets");
                max = Math.max(max, logits[j]);
                mass += weights[j];
            }
            if (!Double.isFinite(mass))
                throw new IllegalArgumentException("cross entropy target sum must be finite");
            double sum = 0;
            for (int j = row; j < row + classes; j++) sum += Math.exp(logits[j] - max);
            double logSum = Math.log(sum);
            for (int j = row; j < row + classes; j++) {
                // Keep the log-sum-exp shifted: adding max first loses small losses
                // for large common offsets. Skip zero weights to avoid 0 * infinity.
                if (weights[j] != 0) loss += (weights[j] / batch) * (logSum + (max - logits[j]));
                if (record) derivative[j] = (Math.exp(logits[j] - max) / sum * mass - weights[j]) / batch;
            }
        }
        Tensor out = Tensor.scalar(loss);
        if (!record) return of(out);
        // Snapshot the adjoint so later target/logit mutation cannot change backward.
        Tensor dx = Tensor.of(derivative).reshape(shape);
        return node(out, List.of(this), g -> addGrad(dx.multiply(g.scalar())));
    }

    public void backward() {
        if (!value.isScalar()) throw new IllegalStateException("backward() requires scalar output");
        backward(Tensor.scalar(1));
    }

    public void backward(Tensor seed) {
        if (!Arrays.equals(seed.shape(), value.shape()))
            throw new IllegalArgumentException("gradient seed shape differs from output");
        List<Variable> topo = new ArrayList<>();
        Set<Variable> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        build(this, seen, topo);
        // Intermediate adjoints belong to this traversal; only leaf gradients accumulate.
        for (Variable v : topo) if (!v.parents.isEmpty()) v.grad = null;
        addGrad(seed);
        for (int i = topo.size() - 1; i >= 0; i--) {
            Variable v = topo.get(i);
            if (v.grad != null && v.backwardFn != null) v.backwardFn.accept(v.grad);
        }
    }
}
