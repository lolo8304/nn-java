package ch.lolo.examples;

import ch.lolo.nn.NN;
import ch.lolo.nn.Sequential;
import ch.lolo.nn.data.MnistDataset;
import ch.lolo.nn.layers.Dense;
import ch.lolo.nn.layers.Dropout;
import ch.lolo.nn.layers.ReLU;
import ch.lolo.nn.loss.CrossEntropyLoss;
import ch.lolo.nn.optim.Adam;

import java.nio.file.Path;

public class MnistExample {
    static void main(String[] a) throws Exception {
        var directory = Path.of(a.length > 0 ? a[0] : "mnist");
        NN.seed(42);
        // MNIST already supplies separate training and test sets.
        var trainingData = new MnistDataset(
                directory.resolve("train-images-idx3-ubyte"),
                directory.resolve("train-labels-idx1-ubyte"));
        var m = Sequential.of(
                new Dense(784, 128),
                new ReLU(),
                new Dropout(.2),
                new Dense(128, 10)
        ).compile(
                new Adam(.001),
                new CrossEntropyLoss()
        );
        var modelPath = Path.of("mnist.nn");
        var history = m.fitWithValidation(trainingData, 50, 256, true, .1, 42, modelPath, 5);

        // Select the checkpoint using validation only, then evaluate the test set once.
        var best = Sequential.load(modelPath);
        var test = new MnistDataset(
                directory.resolve("t10k-images-idx3-ubyte"),
                directory.resolve("t10k-labels-idx1-ubyte"));
        var result = best.evaluate(test, 256);
        System.out.printf("Selected epoch %d (validation loss=%.6f), saved to %s%n",
                history.bestEpoch(), history.bestValidationLoss(), modelPath);
        System.out.printf("Final held-out test: loss=%.6f accuracy=%.2f%% (%d samples)%n",
                result.loss(), 100 * result.accuracy(), test.size());
    }

}
