# java-nn

A small, educational neural-network framework written in pure Java on top of the companion `tensor` module. It is deliberately transparent: no native code, no external numerical library, no hidden training engine.

## Modules

- `tensor` — arbitrary-rank `double` tensors, mutable views, broadcasting, batched matmul, reductions and activations.
- `nn` — reverse-mode autograd, sequential layers, losses, optimizers, datasets/loaders, metrics and model persistence.
- `examples` — XOR, regression, classification and MNIST examples.

The Gradle build targets **Java 26** in every module. No preview features are required.
The default Vector backend uses `jdk.incubator.vector`, included in the JDK.
Gradle supplies its runtime flag automatically. The explicit pure Java backend
needs no incubator module at runtime.

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

Leaf gradients accumulate until `zeroGrad()` is called. Intermediate gradients are recomputed on each backward pass. `Optimizer.zeroGrad(parameters)` handles leaf resets during training.

`predict`, `predictClasses`, and `evaluate` disable autograd recording automatically.
For custom computations, use the exception-safe, thread-local scope:

```java
Variable result = GradMode.noGrad(() -> x.pow(2)); // import ch.lolo.nn.autograd.GradMode
// result.requiresGrad() is false; x remains trainable and its gradient is unchanged.
```

Nested scopes restore the previous mode, including when an exception is thrown.
Operations with no gradient-requiring inputs also omit graph bookkeeping.
Recording is separate from layer training mode: `forward(input, false)` still
supports differentiation, while `GradMode.noGrad(() -> model.forward(input, true))`
still applies training-time dropout. Explicit parameter creation is unaffected,
and existing recorded graphs can still be differentiated inside a no-grad scope.

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

## Performance and Java 26

See the [optimization roadmap](OPTIMIZATION_ROADMAP.md) for prioritized future
tensor and neural-network work, dependencies, and validation criteria.

Contiguous elementwise operations, copies and array exports use direct array loops;
broadcasting computes addresses without allocating an index array per element.
Axis sums and softmax use strides directly, including on sliced/transposed views.
Softmax backward uses `y * (g - sum(g * y))`, linear in the number of classes.
Backpropagation skips derivatives for constant operands, and optimizers allocate
initial moment/velocity tensors only when first needed.

Run the repeated JMH benchmarks and complete training profile on Java 26:

```bash
./gradlew :benchmarks:kernel
./gradlew :benchmarks:training
./gradlew :benchmarks:profile
```

Append `-PtensorBackend=java` for the Java backend. See the
[benchmark guide](benchmarks/GUIDE.md) for warmup, forks, workload filters,
JSON results, phase allocation/time reports, and JFR recordings.
`:examples:runKernelBenchmark` remains the short legacy smoke harness.

Install JDK 26 before building. The shared Gradle toolchain selects Java 26 for
compilation, tests, and all example/benchmark run tasks, even when the shell's
`java` command points to an older JDK. The benchmark uses a 256–512 MiB heap.

The JMH tasks use two forks, time-based warmup, repeated measurements, and GC profiling.
Historical smoke-harness results are not directly comparable to these measurements. See [latest Java/Vector benchmark results](benchmarks/README.md#current-java-26-backend-measurements)
for timings, allocation counts, the user-reported comparison, and the subsequent
default-Vector verification run.

### Reusable tensor output buffers

Arithmetic, scalar arithmetic, copying and ReLU can write into an existing tensor:

```java
Tensor out = Tensor.zeros(x.shape());
x.multiplyInto(y, out);
out.addInto(bias, out);
out.reluInto(out);
x.copyInto(out);
```

`addInto`, `subtractInto`, `multiplyInto`, and `divideInto` accept either a tensor
(with ordinary broadcasting) or a scalar, followed by the destination. Every method
returns that destination; its shape must exactly match the result. Existing
allocating methods keep returning independent storage.

In-place updates, offset views and noncontiguous destinations are supported.
Inputs have snapshot semantics: a differently mapped overlapping input is copied
before any writes; contiguous `copyInto` uses overlap-safe `System.arraycopy`.
Separate storage and exact-layout in-place updates avoid full-sized temporary
buffers, though small shape/index allocations may remain. Conservative overlap
detection can copy interleaved views even when their individual elements are disjoint.
Shape errors are rejected before destination writes.

These are explicitly mutating operations: do not overwrite tensors that a live
autograd graph still needs for backward. Optimizer fusion and automatic gradient
buffer reuse remain separate future changes. Matmul destination buffers are not
part of this initial API.

### Global execution backend

Select the backend once in application code, before training or prediction:

```java
import ch.lolo.tensor.Tensor;
import ch.lolo.tensor.TensorBackend;

Tensor.setBackend(TensorBackend.VECTOR); // default: SIMD kernels with Java fallbacks
Tensor.setBackend(TensorBackend.JAVA);   // explicit override: ordinary Java kernels
System.out.println(Tensor.backend());
```

This is process-wide and applies to existing tensors and all neural networks; it is
not saved in model checkpoints. Configure it before computation begins, and avoid
switching while other threads are computing. Ordinary Java kernels may also be
auto-vectorized by the JVM; JAVA means no explicit Vector API calls.

For Gradle example and benchmark runs, select the backend with a project property.
Gradle supplies both the runtime module flag and the startup property automatically:

```bash
./gradlew :examples:runKernelBenchmark                       # VECTOR (default)
./gradlew :examples:runKernelBenchmark -PtensorBackend=java   # JAVA
./gradlew :examples:runMnist                                 # VECTOR (default)
```

For your own application, launch with `--add-modules jdk.incubator.vector`.
VECTOR is selected by default; `-Dtensor.backend=vector` is optional. To run
without the module, use `-Dtensor.backend=java` at startup. This startup override
is necessary before calling any Tensor method in a JVM without the module.
Generated example distribution launchers also include the selected backend's flags.
Switching back to JAVA in code does not unload the module. Selecting VECTOR
without the module fails with a clear error; a failed runtime switch leaves the
current backend unchanged. `Tensor.isVectorAvailable()` checks whether the module
is resolved, not whether it will be faster on your CPU.

The [Java 26 Vector API](https://docs.oracle.com/en/java/javase/26/docs/api/jdk.incubator.vector/jdk/incubator/vector/package-summary.html)
remains incubating. Gradle enables it for compilation of the tensor module; no
`--enable-preview` flag is needed. Direct `javac` builds of that module must also
supply `--add-modules jdk.incubator.vector`.

Vector kernels cover contiguous same-shape add/subtract/multiply/divide, scalar
arithmetic in either operand order, contiguous 2D matrices with a 1D right-hand
bias, ReLU, and 2D matmul when right-hand rows are contiguous. Larger 2D products
with transposed/strided right operands use fresh 64-column packing panels; smaller
products keep the scalar path. Matmul supports strided operands and offset views,
and packing is never cached across mutations. Kernels use the CPU's preferred
vector width with scalar tails. Other
operations, including batched matmul, reductions, arbitrary `map` callbacks, and
transcendental functions, retain the Java implementation. Copies use `System.arraycopy`.
Matmul preserves the reduction order and uses separate multiply/add operations,
not fused multiply-add. No speedup is guaranteed; benchmark your workload.

`./gradlew build` runs the default Vector suite and `javaTest` in separate JVMs.
With `-PtensorBackend=java`, it runs the Java suite and `vectorTest` instead.
Tests include backend parity, tail lengths, views, empty tensors, NaNs/infinities,
and standalone JVM checks for module availability. Run `./gradlew vectorTest` for
only the Vector backend suites, or `./gradlew javaTest` for pure Java.

For much larger matrix workloads, a native BLAS backend through the
[Foreign Function & Memory API](https://docs.oracle.com/en/java/javase/26/docs/api/java.base/java/lang/foreign/package-summary.html)
is another option, at the cost of a native dependency and memory management.
A native backend is not implemented here.

Remaining boundaries: mixed vector/matrix matmul supports forward execution but not
autograd; cross entropy currently smooths softmax probabilities rather than using a
fused log-sum-exp loss. Prediction and evaluation skip autograd recording;
explicit forward calls retain independent control of recording and training mode.
These are opportunities for a separate API/numerical change.
