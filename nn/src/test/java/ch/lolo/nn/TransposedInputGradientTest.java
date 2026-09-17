package ch.lolo.nn;

import ch.lolo.nn.autograd.Variable;
import ch.lolo.nn.optim.SGD;
import ch.lolo.tensor.Tensor;
import org.junit.jupiter.api.Test;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class TransposedInputGradientTest {
    @Test
    void inputGradientsMatchFiniteDifferencesBeforeAndAfterWeightUpdates() {
        // Offset, noncontiguous weights and inputs, with odd reduction/output dimensions.
        Tensor x = Tensor.random(41L, -1.0, 1.0, 17, 2, 5).slice(1, 1).transpose();
        var weights = new Parameter(Tensor.random(42L, -1.0, 1.0, 17, 2, 9).slice(1, 1));
        var optimizer = new SGD(.05);
        for (int step = 0; step < 2; step++) {
            weights.zeroGrad();
            var input = Variable.parameter(x);
            input.matmul(weights).pow(2).mean().backward();
            for (int row = 0; row < 5; row++) for (int col = 0; col < 17; col++) {
                double original = x.get(row, col), h = 1e-6;
                x.set(original + h, row, col);
                double plus = x.matmul(weights.value()).pow(2).mean().scalar();
                x.set(original - h, row, col);
                double minus = x.matmul(weights.value()).pow(2).mean().scalar();
                x.set(original, row, col);
                assertEquals((plus - minus) / (2 * h), input.grad().get(row, col), 1e-8);
            }
            optimizer.step(List.of(weights));
        }
    }

    @Test
    void blockedProductsMatchInputAndWeightFiniteDifferences() {
        // All three products cross the blocking threshold, including both transposes.
        Tensor x = Tensor.random(41L, -0.5, 0.5, 257, 257);
        var weights = new Parameter(Tensor.random(42L, -0.5, 0.5, 257, 129));
        var input = Variable.parameter(x);
        input.matmul(weights).pow(2).mean().backward();
        for (boolean checkInput : new boolean[]{true, false}) {
            Tensor value = checkInput ? x : weights.value();
            Tensor gradient = checkInput ? input.grad() : weights.grad();
            for (int[] index : new int[][]{{0, 0}, {64, 64},
                    {value.shape()[0] - 1, value.shape()[1] - 1}}) {
                int row = index[0], col = index[1];
                double original = value.get(row, col), h = 1e-5;
                value.set(original + h, row, col);
                double plus = x.matmul(weights.value()).pow(2).mean().scalar();
                value.set(original - h, row, col);
                double minus = x.matmul(weights.value()).pow(2).mean().scalar();
                value.set(original, row, col);
                assertEquals((plus - minus) / (2 * h), gradient.get(row, col), 1e-8);
            }
        }
    }

}
