package ch.lolo.nn.data;

import ch.lolo.tensor.Tensor;

public record Sample(Tensor input, Tensor target) {
}
