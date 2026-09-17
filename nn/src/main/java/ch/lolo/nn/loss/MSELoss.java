package ch.lolo.nn.loss;

import ch.lolo.nn.autograd.Variable;
import ch.lolo.tensor.Tensor;

public final class MSELoss implements Loss {
    public Variable compute(Variable p, Tensor y) {
        return p.subtract(Variable.of(y)).pow(2).mean();
    }

    public String type() {
        return "MSE";
    }
}
