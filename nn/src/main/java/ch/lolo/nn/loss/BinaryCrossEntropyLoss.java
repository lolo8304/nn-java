package ch.lolo.nn.loss;

import ch.lolo.nn.autograd.Variable;
import ch.lolo.tensor.Tensor;

public final class BinaryCrossEntropyLoss implements Loss {
    private static final double E = 1e-12;

    public Variable compute(Variable p, Tensor y) {
        Variable yy = Variable.of(y), one = Variable.of(Tensor.ones(y.shape()));
        Variable q = p.multiply(1 - 2 * E).add(E);
        return yy.multiply(q.log()).add(one.subtract(yy).multiply(one.subtract(q).log())).mean().multiply(-1);
    }

    public String type() {
        return "BinaryCrossEntropy";
    }
}
