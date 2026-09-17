package ch.lolo.nn.optim;

import ch.lolo.nn.Parameter;

import java.util.List;

public interface Optimizer {
    void step(List<Parameter> p);

    default void zeroGrad(List<Parameter> p) {
        p.forEach(Parameter::zeroGrad);
    }

    String type();

    default void writeState(java.io.DataOutput out, List<Parameter> p) throws java.io.IOException {
    }

    default void readState(java.io.DataInput in, List<Parameter> p) throws java.io.IOException {
    }
}
