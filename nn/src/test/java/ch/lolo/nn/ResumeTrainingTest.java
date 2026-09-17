package ch.lolo.nn;

import ch.lolo.nn.layers.Dense;
import ch.lolo.nn.layers.Tanh;
import ch.lolo.nn.loss.CrossEntropyLoss;
import ch.lolo.nn.optim.*;
import ch.lolo.tensor.Tensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.Path;
import java.nio.file.Files;
import static org.junit.jupiter.api.Assertions.*;

class ResumeTrainingTest {
    @Test
    void adamResumeMatchesUninterruptedTraining(@TempDir Path directory) throws Exception {
        verify(new Adam(.007, .8, .95, 1e-5), directory);
    }

    @Test
    void sgdResumeMatchesUninterruptedTraining(@TempDir Path directory) throws Exception {
        verify(new SGD(.03, .7), directory);
    }

    private void verify(Optimizer optimizer, Path directory) throws Exception {
        NN.seed(42);
        var model = Sequential.of(new Dense(2, 3), new Tanh(), new Dense(3, 2))
                .compile(optimizer, new CrossEntropyLoss());
        var data = new ch.lolo.nn.data.TensorDataset(
                Tensor.of(new double[][]{{1, 0}, {0, 1}, {1, 1}, {.2, .8}}),
                Tensor.of(new double[][]{{1, 0}, {0, 1}, {1, 0}, {0, 1}}));
        model.fit(data, 2, 2, false);
        var checkpoint = directory.resolve("before.nn");
        model.save(checkpoint);
        var resumed = Sequential.load(checkpoint);
        // A byte-identical round trip includes every optimizer hyperparameter and moment.
        var roundTrip = directory.resolve("roundtrip.nn");
        resumed.save(roundTrip);
        assertArrayEquals(Files.readAllBytes(checkpoint), Files.readAllBytes(roundTrip));
        model.fit(data, 3, 2, false);
        resumed.fit(data, 3, 2, false);
        for (int i = 0; i < model.parameters().size(); i++) {
            assertArrayEquals(model.parameters().get(i).value().toArray(), resumed.parameters().get(i).value().toArray(), 0);
        }
        var uninterruptedPath = directory.resolve("uninterrupted.nn");
        var resumedPath = directory.resolve("resumed.nn");
        model.save(uninterruptedPath);
        resumed.save(resumedPath);
        assertArrayEquals(Files.readAllBytes(uninterruptedPath), Files.readAllBytes(resumedPath));
    }
}
