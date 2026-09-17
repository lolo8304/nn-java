package ch.lolo.nn.loss;

import ch.lolo.nn.autograd.Variable;
import ch.lolo.tensor.Tensor;

public interface Loss {
    Variable compute(Variable prediction, Tensor target);

    String type();
}
