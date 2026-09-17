package ch.lolo.nn;

import ch.lolo.nn.layers.Dense;
import ch.lolo.nn.loss.CrossEntropyLoss;
import ch.lolo.nn.optim.Adam;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.Path;
import ch.lolo.nn.data.TensorDataset;
import ch.lolo.nn.layers.Dropout;
import ch.lolo.nn.layers.ReLU;
import ch.lolo.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ValidationTrainingTest {
    @Test
    void earlyStoppingStopsAfterPatienceAndRetainsBestCheckpoint(@TempDir Path directory) throws Exception {
        var data = new TensorDataset(Tensor.ones(20, 2), Tensor.generate(i -> i[1] == 0 ? 1 : 0, 20, 2));
        var model = Sequential.of(new Dense(2, 2)).compile(new ch.lolo.nn.optim.SGD(0), new CrossEntropyLoss());
        var checkpoint = directory.resolve("best.nn");
        var history = model.fitWithValidation(data, 20, 8, false, .2, 42, checkpoint, 2);
        assertEquals(3, history.loss().size()); // First best epoch, followed by two unchanged epochs.
        assertEquals(1, history.bestEpoch());
        assertEquals(history.bestValidationLoss(), Sequential.load(checkpoint).evaluate(data, 8).loss(), 1e-12);
        var full = model.fitWithValidation(data, 4, 8, false, .2, 42, checkpoint, 0);
        assertEquals(4, full.loss().size());
        assertThrows(IllegalArgumentException.class,
                () -> model.fitWithValidation(data, 4, 8, false, .2, 42, checkpoint, -1));
    }

    @Test
    void evaluationWeightsPartialBatchAndDisablesDropout() {
        // 256 correct predictions followed by one incorrect prediction.
        var x = Tensor.generate(i -> i[1] == 0 ? 2 : 0, 257, 2);
        var y = Tensor.generate(i -> i[1] == (i[0] == 256 ? 1 : 0) ? 1 : 0, 257, 2);
        var model = Sequential.of(new Dropout(.9), new ReLU()).compile(new Adam(.001), new CrossEntropyLoss());
        var data = new TensorDataset(x, y);
        var result = model.evaluate(data, 256);
        assertEquals(256.0 / 257, result.accuracy(), 1e-12);
        assertEquals(Math.log1p(Math.exp(-2)) + 2.0 / 257, result.loss(), 1e-10);
        assertEquals(result, model.evaluate(data, 256));
    }

    @Test
    void savesBestValidationCheckpointAndKeepsOptimizerState(@TempDir Path directory) throws Exception {
        NN.seed(42);
        var data = new TensorDataset(
                Tensor.generate(i -> i[0] % 2 == i[1] ? 1 : 0, 20, 2),
                Tensor.generate(i -> i[0] % 2 == i[1] ? 1 : 0, 20, 2));
        var adam = new Adam(.01);
        var model = Sequential.of(new Dense(2, 2)).compile(adam, new CrossEntropyLoss());
        var checkpoint = directory.resolve("best.nn");
        var history = model.fitWithValidation(data, 3, 5, true, .2, 42, checkpoint);
        assertEquals(3, history.loss().size());
        assertEquals(3, history.validationLoss().size());
        assertEquals(3, history.validationAccuracy().size());
        assertEquals(12, adam.stepCount()); // 16 training samples, four batches per epoch.
        double minimum = history.validationLoss().stream().mapToDouble(Double::doubleValue).min().orElseThrow();
        assertEquals(minimum, history.bestValidationLoss());
        assertEquals(history.validationLoss().indexOf(minimum) + 1, history.bestEpoch());
        var saved = Sequential.load(checkpoint);
        var validation = data.splitTraining(.2, 42)[1];
        assertEquals(minimum, saved.evaluate(validation, 5).loss(), 1e-12);
        assertEquals(history.bestEpoch() * 4, ((Adam) saved.optimizer()).stepCount());
        var before = model.predict(Tensor.ones(1, 2)).toArray();
        model.evaluate(validation, 5);
        assertArrayEquals(before, model.predict(Tensor.ones(1, 2)).toArray());
        assertEquals(12, adam.stepCount());
    }

    @Test
    void rejectsInvalidEpochsBeforeTraining(@TempDir Path directory) {
        var model = Sequential.of(new Dense(2, 2)).compile(new Adam(.001), new CrossEntropyLoss());
        var data = new TensorDataset(Tensor.ones(10, 2), Tensor.ones(10, 2));
        assertThrows(IllegalArgumentException.class,
                () -> model.fitWithValidation(data, 0, 2, true, .1, 42, directory.resolve("best.nn")));
    }
}
