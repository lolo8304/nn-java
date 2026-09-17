package ch.lolo.nn;

import ch.lolo.nn.autograd.Variable;

import java.util.List;

public interface Layer {
    Variable forward(Variable input, boolean training);

    default List<Parameter> parameters() {
        return List.of();
    }

    String type();
}
