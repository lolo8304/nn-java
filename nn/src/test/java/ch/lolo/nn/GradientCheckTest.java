package ch.lolo.nn;

import ch.lolo.nn.autograd.Variable;
import ch.lolo.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class GradientCheckTest {
    @Test
    void sigmoidFiniteDifference() {
        double x = .37, h = 1e-6;
        var v = Variable.parameter(Tensor.scalar(x));
        v.sigmoid().backward();
        double numeric = (sig(x + h) - sig(x - h)) / (2 * h);
        assertEquals(numeric, v.grad().scalar(), 1e-6);
    }

    private double sig(double x) {
        return 1 / (1 + Math.exp(-x));
    }
}
