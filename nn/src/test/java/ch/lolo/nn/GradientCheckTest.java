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

    @Test
    void contiguousUnaryKernelsFiniteDifference() {
        Tensor input = Tensor.random(52L, .2, 2.0, 129);
        for (int kind = 0; kind < 4; kind++) {
            var x = Variable.parameter(input);
            unary(kind, x).sum().backward();
            for (int i : new int[]{0, 1, 64, 127, 128}) {
                Tensor plus = input.copy(), minus = input.copy();
                plus.set(input.get(i) + 1e-6, i);
                minus.set(input.get(i) - 1e-6, i);
                double numeric = (unary(kind, Variable.of(plus)).value().sum().scalar()
                        - unary(kind, Variable.of(minus)).value().sum().scalar()) / 2e-6;
                assertEquals(numeric, x.grad().get(i), 1e-7);
            }
        }
    }

    private Variable unary(int kind, Variable x) {
        return switch (kind) {
            case 0 -> x.log(); case 1 -> x.tanh(); case 2 -> x.sigmoid(); default -> x.pow(2);
        };
    }

    private double sig(double x) {
        return 1 / (1 + Math.exp(-x));
    }
}
