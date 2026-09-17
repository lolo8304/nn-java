package ch.lolo.nn;

import ch.lolo.nn.autograd.Variable;
import ch.lolo.nn.data.Batch;
import ch.lolo.nn.data.DataLoader;
import ch.lolo.nn.data.Dataset;
import ch.lolo.nn.data.TensorDataset;
import ch.lolo.nn.loss.Loss;
import ch.lolo.nn.metrics.Accuracy;
import ch.lolo.nn.optim.Optimizer;
import ch.lolo.tensor.Tensor;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public final class Sequential {
    private final List<Layer> layers;
    private Optimizer optimizer;
    private Loss loss;

    public Sequential(List<Layer> l) {
        layers = new ArrayList<>(l);
    }

    public static Sequential of(Layer... l) {
        return new Sequential(List.of(l));
    }

    public static Sequential load(Path p) throws IOException {
        return ch.lolo.nn.io.ModelIO.load(p);
    }

    public List<Layer> layers() {
        return List.copyOf(layers);
    }

    public List<Parameter> parameters() {
        return layers.stream().flatMap(x -> x.parameters().stream()).toList();
    }

    public Sequential compile(Optimizer o, Loss l) {
        optimizer = o;
        loss = l;
        return this;
    }

    public Variable forward(Variable x, boolean training) {
        Variable v = x;
        for (Layer l : layers) v = l.forward(v, training);
        return v;
    }

    public Tensor predict(Tensor x) {
        return forward(Variable.of(x), false).value();
    }

    public Tensor predictClasses(Tensor x) {
        Tensor p = predict(x);
        if (p.rank() != 2) throw new IllegalArgumentException("expected [batch,classes]");
        return Tensor.generate(i -> {
            Tensor row = p.slice(0, i[0]);
            int best = 0;
            for (int j = 1; j < row.shape()[0]; j++) if (row.get(j) > row.get(best)) best = j;
            return best;
        }, p.shape()[0]);
    }

    public TrainingHistory fit(Dataset ds, int epochs, int batchSize, boolean shuffle) {
        checkTraining(epochs, batchSize);
        if (ds.size() == 0) throw new IllegalArgumentException("Cannot train on an empty dataset");
        TrainingHistory history = new TrainingHistory();
        for (int epoch = 1; epoch <= epochs; epoch++) {
            var metrics = trainEpoch(ds, batchSize, shuffle);
            history.add(metrics.loss(), metrics.accuracy());
            System.out.printf("Epoch %d/%d loss=%.6f accuracy=%.4f%n",
                    epoch, epochs, metrics.loss(), metrics.accuracy());
        }
        return history;
    }

    /**
     * Splits training data once and evaluates validation data after every epoch.
     * Saves the lowest finite validation-loss checkpoint, including optimizer state.
     * This model retains the final epoch's state; load bestModelPath to use the best epoch.
     * Test data must be kept separate and evaluated only after model selection.
     *
     * @param data training data, excluding the final test set
     * @param epochs number of training epochs (positive)
     * @param batchSize maximum batch size (positive)
     * @param shuffle whether to shuffle the training subset each epoch
     * @param validationFraction fraction reserved for validation
     * @param splitSeed seed for the reproducible training/validation split
     * @param bestModelPath destination for the best model checkpoint
     * @return training and validation metrics, with a one-based best epoch
     * @throws IOException if saving the checkpoint fails
     */
    public TrainingHistory fitWithValidation(Dataset data, int epochs, int batchSize, boolean shuffle,
                                             double validationFraction, long splitSeed,
                                             Path bestModelPath) throws IOException {
        return fitWithValidation(data, epochs, batchSize, shuffle, validationFraction, splitSeed, bestModelPath, 0);
    }

    /**
     * Fits with validation and optional early stopping. A positive patience stops after that
     * many consecutive epochs without a strictly lower finite validation loss. Zero disables
     * early stopping. The saved checkpoint remains the best model; this instance retains
     * the last trained epoch. All other arguments match the overload without patience.
     */
    public TrainingHistory fitWithValidation(Dataset data, int epochs, int batchSize, boolean shuffle,
                                             double validationFraction, long splitSeed,
                                             Path bestModelPath, int patience) throws IOException {
        return fitWithValidation(data, epochs, batchSize, shuffle, validationFraction, splitSeed,
                bestModelPath, patience, java.util.function.UnaryOperator.identity());
    }

    /**
     * Applies trainingTransform once, after splitting, to the training subset only.
     * Use a dataset view that generates fresh augmentation on get(); validation stays untouched.
     */
    public TrainingHistory fitWithValidation(Dataset data, int epochs, int batchSize, boolean shuffle,
                                             double validationFraction, long splitSeed,
                                             Path bestModelPath, int patience,
                                             java.util.function.UnaryOperator<Dataset> trainingTransform) throws IOException {
        checkTraining(epochs, batchSize);
        if (patience < 0) throw new IllegalArgumentException("Patience must be nonnegative");
        java.util.Objects.requireNonNull(bestModelPath, "bestModelPath");
        var split = data.splitTraining(validationFraction, splitSeed);
        var train = java.util.Objects.requireNonNull(trainingTransform, "trainingTransform").apply(split[0]);
        if (train == null || train.size() != split[0].size())
            throw new IllegalArgumentException("Training transform must preserve sample count");
        var validation = split[1];
        System.out.printf("Training samples: %d; validation samples: %d%n", train.size(), validation.size());
        var history = new TrainingHistory();
        for (int epoch = 1; epoch <= epochs; epoch++) {
            var training = trainEpoch(train, batchSize, shuffle);
            var metrics = evaluate(validation, batchSize);
            history.add(training.loss(), training.accuracy());
            history.addValidation(metrics.loss(), metrics.accuracy());
            System.out.printf("Epoch %d/%d loss=%.6f accuracy=%.4f val_loss=%.6f val_accuracy=%.4f%n",
                    epoch, epochs, training.loss(), training.accuracy(), metrics.loss(), metrics.accuracy());
            if (history.bestEpoch() == epoch) save(bestModelPath);
            if (patience > 0 && epoch - history.bestEpoch() >= patience) {
                System.out.printf("Early stopping after epoch %d; best epoch: %d%n", epoch, history.bestEpoch());
                break;
            }
        }
        if (history.bestEpoch() == 0) throw new IllegalStateException("No finite validation loss; check training");
        return history;
    }

    private void checkTraining(int epochs, int batchSize) {
        if (optimizer == null || loss == null) throw new IllegalStateException("compile first");
        if (epochs < 1 || batchSize < 1) throw new IllegalArgumentException("Epochs and batch size must be positive");
    }

    private Evaluation trainEpoch(Dataset data, int batchSize, boolean shuffle) {
        double totalLoss = 0, totalAccuracy = 0;
        int total = 0;
        List<Parameter> parameters = parameters();
        for (Batch batch : new DataLoader(data, batchSize, shuffle)) {
            optimizer.zeroGrad(parameters);
            var out = forward(Variable.of(batch.inputs()), true);
            var value = loss.compute(out, batch.targets());
            value.backward();
            optimizer.step(parameters);
            int count = batch.inputs().shape()[0];
            totalLoss += value.value().scalar() * count;
            totalAccuracy += classificationAccuracy(out.value(), batch.targets()) * count;
            total += count;
        }
        return new Evaluation(totalLoss / total, totalAccuracy / total);
    }

    /** Sample-weighted loss and classification accuracy; accuracy is NaN for non-classification shapes. */
    public record Evaluation(double loss, double accuracy) {}

    /** Evaluates using the compiled loss with dropout disabled and without updating weights or optimizer state. */
    public Evaluation evaluate(Dataset data, int batchSize) {
        if (loss == null) throw new IllegalStateException("compile first");
        if (data.size() == 0) throw new IllegalArgumentException("Cannot evaluate an empty dataset");
        double totalLoss = 0, totalAccuracy = 0;
        int total = 0;
        for (Batch batch : new DataLoader(data, batchSize, false)) {
            var predictions = predict(batch.inputs());
            int count = batch.inputs().shape()[0];
            totalLoss += loss.compute(Variable.of(predictions), batch.targets()).value().scalar() * count;
            totalAccuracy += classificationAccuracy(predictions, batch.targets()) * count;
            total += count;
        }
        return new Evaluation(totalLoss / total, totalAccuracy / total);
    }

    private static double classificationAccuracy(Tensor predictions, Tensor targets) {
        if (predictions.rank() != 2 || targets.rank() != 2 || predictions.shape()[1] < 2
                || predictions.shape()[1] != targets.shape()[1]) return Double.NaN;
        return new Accuracy().compute(predictions, targets);
    }

    public TrainingHistory fit(Tensor x, Tensor y, int epochs, int batchSize) {
        return fit(new TensorDataset(x, y), epochs, batchSize, true);
    }

    public void save(Path p) throws IOException {
        ch.lolo.nn.io.ModelIO.save(this, p);
    }

    public Optimizer optimizer() {
        return optimizer;
    }

    public Loss loss() {
        return loss;
    }
}
