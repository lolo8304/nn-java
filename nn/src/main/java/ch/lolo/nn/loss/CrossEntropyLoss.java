package ch.lolo.nn.loss;

import ch.lolo.nn.autograd.Variable;
import ch.lolo.tensor.Tensor;

public final class CrossEntropyLoss implements Loss {
    /** Standard logits cross entropy; see Variable.crossEntropy for target/reduction rules. */
    public Variable compute(Variable logits, Tensor target) {
        return logits.crossEntropy(target);
    }

    public String type() {
        return "CrossEntropy";
    }
}
