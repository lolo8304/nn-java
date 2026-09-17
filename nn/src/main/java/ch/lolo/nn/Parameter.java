package ch.lolo.nn;

import ch.lolo.nn.autograd.Variable;
import ch.lolo.tensor.Tensor;

public final class Parameter extends Variable {
    public Parameter(Tensor value) {
        super(value, true);
    }

    /** Adds a same-shaped delta in place, safely handling overlapping views. */
    public void applyDelta(Tensor d) {
        if (!java.util.Arrays.equals(value().shape(), d.shape()))
            throw new IllegalArgumentException("delta shape differs from parameter");
        value().addInto(d, value());
    }
}
