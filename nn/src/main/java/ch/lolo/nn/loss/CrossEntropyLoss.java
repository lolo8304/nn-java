package ch.lolo.nn.loss;

import ch.lolo.nn.autograd.Variable;
import ch.lolo.tensor.Tensor;

public final class CrossEntropyLoss implements Loss {
    private static final double E = 1e-12;

    public Variable compute(Variable logits, Tensor oneHot) {
        Variable p = logits.softmax(-1).multiply(1 - 2 * E).add(E);
        Variable per = Variable.of(oneHot).multiply(p.log()).sum().multiply(-1);
        int batch = logits.value().rank() > 1 ? logits.value().shape()[0] : 1;
        return per.multiply(1.0 / batch);
    }

    public String type() {
        return "CrossEntropy";
    }
}
