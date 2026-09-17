# java-nn

A small, educational neural-network framework written in pure Java on top of the companion `tensor` module. It is deliberately transparent: no native code, no external numerical library, no hidden training engine.

## Modules

- `tensor` — arbitrary-rank `double` tensors, mutable views, broadcasting, batched matmul, reductions and activations.
- `nn` — reverse-mode autograd, sequential layers, losses, optimizers, datasets/loaders, metrics and model persistence.
- `examples` — XOR, regression, classification and MNIST examples.

The Gradle build targets **Java 26**. The source intentionally stays simple enough to compile on Java 21+ as well.

## Quick start

```java
NN.seed(42);

Tensor x = Tensor.of(new double[][] {
    {0,0}, {0,1}, {1,0}, {1,1}
});
Tensor y = Tensor.of(new double[][] {
    {0}, {1}, {1}, {0}
});

Sequential model = Sequential.of(
    new Dense(2, 8),
    new Tanh(),
    new Dense(8, 1),
    new Sigmoid()
).compile(
    new Adam(0.03),
    new BinaryCrossEntropyLoss()
);

TrainingHistory history = model.fit(x, y, 500, 4);
Tensor prediction = model.predict(x);
model.save(Path.of("xor.nn"));
```

## Autograd

```java
Variable x = Variable.parameter(Tensor.scalar(3));
Variable y = x.pow(2);
y.backward();
System.out.println(x.grad().scalar()); // 6.0
```

Gradients accumulate until `zeroGrad()` is called. `Optimizer.zeroGrad(parameters)` handles this during training.

## Layers

`Dense`, `Flatten`, `Dropout`, `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `Softmax`.

## Losses

`MSELoss`, `BinaryCrossEntropyLoss`, `CrossEntropyLoss`. Cross entropy expects one-hot targets and logits; it applies softmax internally.

## Optimizers

`SGD` (optional momentum) and `Adam`.

## Data

`Dataset`, `Sample`, `TensorDataset`, `DataLoader`, and a pure-Java IDX `MnistDataset`. MNIST files are supplied by the caller; the library does not download data implicitly.

Any dataset supports a reproducible training/validation split:

```java
var split = dataset.splitTraining(0.1, 42);
var train = split[0];
var validation = split[1];
```

The validation fraction is rounded to a sample count; both subsets must be nonempty.
These are shuffled views of the original dataset, so samples are not copied.

To manage the split, epoch loop, validation metrics, and best checkpoint together:

```java
var checkpoint = Path.of("mnist.nn");
var history = model.fitWithValidation(trainingData, 50, 256, true, 0.1, 42, checkpoint);
var best = Sequential.load(checkpoint);
var testMetrics = best.evaluate(testData, 256);
```

`fitWithValidation` splits once, preserves optimizer state across epochs, and saves
the lowest finite validation-loss checkpoint. `history` exposes training and validation
loss/accuracy, `bestEpoch()` (one-based), and `bestValidationLoss()`. The original model
retains its final epoch state; load the saved checkpoint for the selected model.
`evaluate` uses the compiled loss and sample-weighted metrics with dropout disabled.
Classification accuracy is `NaN` when output/target shapes do not support multiclass accuracy.
Keep test data outside the fitting call and evaluate it only after selecting the model.

An optional final `patience` argument enables early stopping:

```java
var history = model.fitWithValidation(trainingData, 50, 256, true, 0.1, 42, checkpoint, 5);
```

This stops after five consecutive epochs without a strictly lower validation loss.
Omit patience or pass `0` to run every epoch. Reload the checkpoint to use the best model.
Batch loading fetches each sample once per batch; 2D matrix multiplication uses direct
array access for both ordinary matrices and transposed/sliced views.

## Persistence

`.nn` files use the versioned `JNN1` binary format. Architecture, layer parameters, optimizer type/configuration and optimizer state are stored so training can continue after loading.

## Build

```bash
gradle test
gradle build
gradle :examples:runXor
gradle :examples:runMnist
```

The MNIST example reads `train-images-idx3-ubyte` and `train-labels-idx1-ubyte`
from `mnist/`, shuffling with split seed 42 to reserve 10% for validation
(54,000 training / 6,000 validation samples with standard MNIST). It trains for
up to 50 epochs with batch size 256 and early-stopping patience 5, reports validation loss and accuracy after each epoch,
and saves the checkpoint with the lowest validation loss as `mnist.nn` in the project root.
After training, it reloads that checkpoint and evaluates once on the untouched
`t10k-images-idx3-ubyte` and `t10k-labels-idx1-ubyte` test set.
To use another data directory,
pass `--args="/path/to/mnist"` to `:examples:runMnist`.

With a local Gradle installation you can generate the standard wrapper once with `gradle wrapper` and thereafter use `./gradlew`.

## Design boundary

V1 intentionally supports sequential networks only. It has a real reverse-mode autograd engine, but does not attempt arbitrary graph-model composition, convolutions, recurrent layers, GPU execution, mixed precision or parallel training.
