package ch.lolo.nn;

import ch.lolo.nn.autograd.Variable;
import ch.lolo.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class AutogradTest {
    @Test
    void squareGradient() {
        var x = Variable.parameter(Tensor.scalar(3));
        x.pow(2).backward();
        assertEquals(6, x.grad().scalar(), 1e-10);
    }

    @Test
    void matmulGradient() {
        var x = Variable.parameter(Tensor.of(new double[][]{{1, 2}}));
        var w = Variable.parameter(Tensor.of(new double[][]{{3}, {4}}));
        x.matmul(w).sum().backward();
        assertArrayEquals(new double[]{3, 4}, x.grad().toArray(), 1e-10);
        assertArrayEquals(new double[]{1, 2}, w.grad().toArray(), 1e-10);
    }
}
