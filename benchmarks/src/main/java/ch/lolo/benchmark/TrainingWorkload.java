package ch.lolo.benchmark;

import ch.lolo.nn.*;
import ch.lolo.nn.autograd.Variable;
import ch.lolo.nn.data.*;
import ch.lolo.nn.layers.*;
import ch.lolo.nn.loss.*;
import ch.lolo.nn.optim.*;
import ch.lolo.tensor.Tensor;
import java.util.List;

/** Shared repeatable workload; performs real loading, forward, loss, backward and updates. */
final class TrainingWorkload {
    final TensorDataset dataset;
    final Sequential model;
    final List<Parameter> parameters;
    final Optimizer optimizer;
    final Loss loss;
    final int batchSize;

    TrainingWorkload(int samples, int features, int batchSize, long seed) {
        this(samples, features, batchSize, seed, new MSELoss(), false);
    }

    TrainingWorkload(int samples, int features, int batchSize, long seed, Loss loss, boolean classification) {
        this.loss = loss;
        this.batchSize = batchSize;
        Tensor x = Tensor.random(seed, -1.0, 1.0, samples, features);
        Tensor teacher = Tensor.random(seed + 1, -.1, .1, features, 10);
        Tensor y = x.matmul(teacher).sigmoid();
        if (classification) {
            Tensor labels = Tensor.zeros(samples, 10);
            for (int i = 0; i < samples; i++) {
                int best = 0;
                for (int j = 1; j < 10; j++) if (y.get(i, j) > y.get(i, best)) best = j;
                labels.set(1, i, best);
            }
            y = labels;
        }
        dataset = new TensorDataset(x, y);
        NN.seed(seed);
        optimizer = new Adam(.001);
        model = Sequential.of(new Dense(features, 64), new ReLU(), new Dense(64, 10))
                .compile(optimizer, loss);
        parameters = model.parameters();
    }

    double epoch() {
        double totalLoss = 0;
        for (Batch batch : new DataLoader(dataset, batchSize, true)) {
            optimizer.zeroGrad(parameters);
            var predictions = model.forward(Variable.of(batch.inputs()), true);
            var value = loss.compute(predictions, batch.targets());
            value.backward();
            optimizer.step(parameters);
            totalLoss += value.value().scalar() * batch.inputs().shape()[0];
        }
        return totalLoss / dataset.size();
    }
}
