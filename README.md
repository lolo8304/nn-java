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

For training-only image augmentation, pass a dataset wrapper factory after patience:

```java
var history = model.fitWithValidation(trainingData, 50, 256, true, 0.1, 42, checkpoint, 5,
        train -> new AugmentedImageDataset(train, 28, 28, 3, .85, 1.15, 10, 43));
```

The factory receives only the training subset, after splitting. This example uses fresh
random shifts of up to 3 pixels, uniform scaling from 85–115%, and rotations up to 10 degrees.
Images remain 28×28 with black padding and bilinear interpolation; labels are unchanged.
Transforms that would clip the digit are retried, falling back to the original if necessary.
The augmentation has its own seeded random generator. Validation and test data are unchanged.
Do not wrap the full dataset before splitting, which would also augment validation samples.

## Persistence

`.nn` files use the versioned `JNN1` binary format. Architecture, layer parameters, optimizer type/configuration and optimizer state are stored so training can continue after loading.

Load a trained model with `Sequential.load(path)` and call `fit` or `fitWithValidation`
on it to continue training. Avoid calling `compile` again when retaining the saved
optimizer: it restores Adam's learning rate, betas, epsilon, step count and moments,
or SGD's learning rate, momentum and velocity. Existing JNN1 files remain compatible.
Random-generator state and previous validation/early-stopping history are not stored,
so resuming stochastic training is not a bit-for-bit continuation of its random sequence.

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
Training uses the image augmentation settings above.
After training, it reloads that checkpoint and evaluates once on the untouched
`t10k-images-idx3-ubyte` and `t10k-labels-idx1-ubyte` test set.
To use another data directory,
pass `--args="/path/to/mnist"` to `:examples:runMnist`.

Resume an existing MNIST checkpoint for up to 20 additional epochs:

```bash
./gradlew :examples:runMnist --args="--resume mnist.nn"
```

This saves the best checkpoint from the new run as `mnist-finetuned.nn`, leaving
the original unchanged. Override the data directory, output path, or additional epochs:

```bash
./gradlew :examples:runMnist --args="mnist --resume mnist.nn --output mnist-augmented.nn --epochs 10"
```

The example checks the saved architecture and loss before training. It keeps the same
validation split seed (42), augmentation, and early-stopping patience (5). Reuse the same
training files and ordering to preserve the validation split. Early-stopping history starts
fresh; the new checkpoint is the best from the resumed run, not necessarily better than the
original. A fresh run without `--resume` still trains for up to 50 epochs.

With a local Gradle installation you can generate the standard wrapper once with `gradle wrapper` and thereafter use `./gradlew`.

## Design boundary

V1 intentionally supports sequential networks only. It has a real reverse-mode autograd engine, but does not attempt arbitrary graph-model composition, convolutions, recurrent layers, GPU execution, mixed precision or parallel training.
