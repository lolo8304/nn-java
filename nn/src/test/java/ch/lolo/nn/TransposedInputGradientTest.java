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
}
