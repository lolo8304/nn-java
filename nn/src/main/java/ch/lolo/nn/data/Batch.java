package ch.lolo.nn.data;

import ch.lolo.tensor.Tensor;

public record Batch(Tensor inputs, Tensor targets) {
}
