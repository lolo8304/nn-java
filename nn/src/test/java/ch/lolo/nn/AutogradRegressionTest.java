package ch.lolo.nn;

import ch.lolo.nn.autograd.Variable;
import ch.lolo.tensor.Tensor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class AutogradRegressionTest {
    @Test void broadcastGradientRestoresSingletonAxes() {
        var bias = Variable.parameter(Tensor.ones(1, 3, 1));
        bias.multiply(Variable.of(Tensor.ones(2, 3, 4))).sum().backward();
        assertArrayEquals(new int[]{1, 3, 1}, bias.grad().shape());
        assertArrayEquals(new double[]{8, 8, 8}, bias.grad().toArray());
    }
    @Test void emptyBroadcastGradient() {
        var bias = Variable.parameter(Tensor.ones(1, 3));
        bias.add(Tensor.zeros(0, 3)).sum().backward();
        assertArrayEquals(new double[]{0, 0, 0}, bias.grad().toArray());
    }
    @Test void repeatedBackwardAccumulatesOnlyLeafGradients() {
        var x = Variable.parameter(Tensor.scalar(3));
        var y = x.pow(2).multiply(2);
        y.backward(); y.backward();
        assertEquals(24, x.grad().scalar());
        x.zeroGrad(); y.backward();
        assertEquals(12, x.grad().scalar());
    }
    @Test void batchedMatmulBroadcastGradientsMatchFiniteDifferences() {
        Tensor left = Tensor.random(12L, 2, 2, 3), right = Tensor.random(13L, 1, 3, 2);
        var a = Variable.parameter(left);
        var b = Variable.parameter(right);
        a.matmul(b).pow(2).mean().backward();
        for (int which = 0; which < 2; which++) {
            Tensor input = which == 0 ? left : right;
            Tensor grad = which == 0 ? a.grad() : b.grad();
            assertArrayEquals(input.shape(), grad.shape());
            for (int i = 0; i < input.shape()[0]; i++)
                for (int j = 0; j < input.shape()[1]; j++)
                    for (int k = 0; k < input.shape()[2]; k++) {
                        double h = 1e-6, original = input.get(i, j, k);
                        input.set(original + h, i, j, k);
                        double plus = left.matmul(right).pow(2).mean().scalar();
                        input.set(original - h, i, j, k);
                        double minus = left.matmul(right).pow(2).mean().scalar();
                        input.set(original, i, j, k);
                        assertEquals((plus - minus) / (2 * h), grad.get(i, j, k), 1e-8);
                    }
        }
    }
    @Test void wrongSeedShapeRejected() {
        var x = Variable.parameter(Tensor.ones(2, 3));
        assertThrows(IllegalArgumentException.class, () -> x.relu().backward(Tensor.ones(3)));
    }
    @Test void softmaxGradientsMatchFiniteDifferencesForEveryAxisAndViews() {
        Tensor x = Tensor.of(new double[][]{{.2, -.4, .7}, {1.2, .1, -.8}}).transpose();
        Tensor seed = Tensor.of(new double[][]{{.3, .8}, {-.2, .6}, {.1, -.5}});
        for (int axis : new int[]{0, 1, -1}) {
            var v = Variable.parameter(x);
            v.softmax(axis).backward(seed);
            for (int i = 0; i < 3; i++) for (int j = 0; j < 2; j++) {
                double h = 1e-6;
                Tensor plus = x.copy(), minus = x.copy();
                plus.set(x.get(i, j) + h, i, j); minus.set(x.get(i, j) - h, i, j);
                double numeric = (plus.softmax(axis).multiply(seed).sum().scalar()
                        - minus.softmax(axis).multiply(seed).sum().scalar()) / (2 * h);
                assertEquals(numeric, v.grad().get(i, j), 1e-8);
            }
        }
    }
}
