package ch.lolo.examples;

import ch.lolo.nn.NN;
import ch.lolo.nn.Sequential;
import ch.lolo.nn.data.MnistDataset;
import ch.lolo.nn.data.AugmentedImageDataset;
import ch.lolo.nn.layers.Dense;
import ch.lolo.nn.layers.Dropout;
import ch.lolo.nn.layers.ReLU;
import ch.lolo.nn.loss.CrossEntropyLoss;
import ch.lolo.nn.optim.Adam;

import java.nio.file.Path;
import java.nio.file.Files;

public class MnistExample {
    static void main(String[] a) throws Exception {
        var options = Options.parse(a);
        var directory = options.directory();
        NN.seed(42);
        var m = options.resume() == null ? newModel() : Sequential.load(options.resume());
        validateModel(m);
        if (options.resume() != null) {
            System.out.printf("Resuming %s with saved %s optimizer; up to %d additional epochs%n",
                    options.resume(), m.optimizer().type(), options.epochs());
        }
        // MNIST already supplies separate training and test sets.
        var trainingData = new MnistDataset(
                directory.resolve("train-images-idx3-ubyte"),
                directory.resolve("train-labels-idx1-ubyte"));
        var modelPath = options.output();
        var history = m.fitWithValidation(trainingData, options.epochs(), 256, true, .1, 42, modelPath, 5,
                train -> new AugmentedImageDataset(train, 28, 28, 3, .85, 1.15, 10, 43));

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

    private static Sequential newModel() {
        return Sequential.of(new Dense(784, 128), new ReLU(), new Dropout(.2), new Dense(128, 10))
                .compile(new Adam(.001), new CrossEntropyLoss());
    }

    static void validateModel(Sequential model) {
        var layers = model.layers();
        if (layers.size() != 4
                || !(layers.get(0) instanceof Dense first) || first.inputSize() != 784 || first.outputSize() != 128
                || !(layers.get(1) instanceof ReLU)
                || !(layers.get(2) instanceof Dropout dropout) || dropout.probability() != .2
                || !(layers.get(3) instanceof Dense last) || last.inputSize() != 128 || last.outputSize() != 10) {
            throw new IllegalArgumentException("Expected MNIST architecture: Dense(784,128), ReLU, Dropout(.2), Dense(128,10)");
        }
        if (model.optimizer() == null || !(model.loss() instanceof CrossEntropyLoss)) {
            throw new IllegalArgumentException("Resume requires a saved optimizer and CrossEntropyLoss");
        }
    }

    record Options(Path directory, Path resume, Path output, int epochs) {
        static Options parse(String[] args) throws java.io.IOException {
            Path directory = Path.of("mnist"), resume = null, output = null;
            Integer epochs = null;
            boolean hasDirectory = false;
            for (int i = 0; i < args.length; i++) {
                switch (args[i]) {
                    case "--resume", "--output", "--epochs" -> {
                        String option = args[i];
                        if (++i == args.length) throw new IllegalArgumentException("Missing value for " + option);
                        switch (option) {
                            case "--resume" -> resume = Path.of(args[i]);
                            case "--output" -> output = Path.of(args[i]);
                            case "--epochs" -> epochs = Integer.parseInt(args[i]);
                        }
                    }
                    default -> {
                        if (args[i].startsWith("--") || hasDirectory)
                            throw new IllegalArgumentException("Usage: [data-directory] [--resume model.nn] [--output result.nn] [--epochs N]");
                        directory = Path.of(args[i]);
                        hasDirectory = true;
                    }
                }
            }
            if (epochs == null) epochs = resume == null ? 50 : 20;
            if (epochs < 1) throw new IllegalArgumentException("Epochs must be positive");
            if (output == null) output = Path.of(resume == null ? "mnist.nn" : "mnist-finetuned.nn");
            if (resume != null && (resume.toAbsolutePath().normalize().equals(output.toAbsolutePath().normalize())
                    || (Files.exists(resume) && Files.exists(output) && Files.isSameFile(resume, output)))) {
                throw new IllegalArgumentException("Choose a different --output path to preserve the original checkpoint");
            }
            return new Options(directory, resume, output, epochs);
        }
    }
}
