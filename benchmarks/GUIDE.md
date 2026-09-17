# Benchmarking and profiling

Use the dedicated `benchmarks` module for performance decisions. It uses
[OpenJDK JMH](https://github.com/openjdk/jmh) 1.37 with generated benchmark harnesses.
The old `:examples:runKernelBenchmark` remains a quick smoke comparison; its
100-operation warmup is not the reference for new performance claims.

## Repeated kernel measurements

```bash
./gradlew :benchmarks:kernel
./gradlew :benchmarks:kernel -PtensorBackend=java
```

Defaults: two independent JVM forks, three one-second warmup iterations, five
one-second measurement iterations per fork, one benchmark thread, a 256–512 MiB
heap, and the GC profiler. The fork receives the selected backend and module flags
explicitly. JMH prints the exact JVM and configuration, timing uncertainty, normalized
allocation (B/op), allocation rate, and GC count/time. Incubator warnings and JMH
1.37's Java 26 Unsafe deprecation warning do not indicate a benchmark failure.

The default sweep tests 32×128 and 32×784 tensors. It includes allocating and reusable
elementwise/copy operations, reductions, strided maps, matmul, transposed-right
matmul, softmax backward, and dense backward with both input and weight gradients.
This dense workload is broader than the old smoke benchmark, which only computed
weight gradients. A complete default sweep takes several minutes.

Filter benchmarks or override parameters and warmup using normal JMH arguments:

```bash
./gradlew :benchmarks:kernel --args='.*KernelBenchmarks.(elementwise|elementwiseInto|copy|copyInto)$ -p features=128'
./gradlew :benchmarks:kernel --args='.*matmul.* -p features=127,128,784 -p batch=1,32,256 -wi 5 -i 8 -w 2s -r 2s -f 3'
```

Short smoke check of harness wiring only:

```bash
./gradlew :benchmarks:kernel --args='.*elementwiseInto -p features=128 -f 1 -wi 1 -i 1 -w 100ms -r 100ms'
```

JSON results default to `benchmarks/build/results/kernel-vector.json` or
`kernel-java.json`, and are overwritten on another run of that kind/backend.
Use `-rff /absolute/path/result.json` to retain separate runs. Keep the raw JSON
with any reported comparison; use identical filters, parameters and JVM settings
for both backends. CLI JVM argument overrides must include the appropriate module
and `tensor.backend` property themselves.

## Complete training epochs

```bash
./gradlew :benchmarks:training
./gradlew :benchmarks:training -PtensorBackend=java
```

`trainingEpoch` measures a complete training epoch; `loadingEpoch` assembles every
shuffled batch and consumes it through a JMH Blackhole, without training. Both
use the same dataset setup. Filter with `.*TrainingBenchmarks.trainingEpoch` or
`.*TrainingBenchmarks.loadingEpoch` to run only one.

Each measured operation is a complete epoch over 1,024 synthetic samples with a
batch size of 32. The default feature sweep is 128 and 784. A seeded teacher creates
fixed regression targets outside the timed epoch; the trainable model is
Dense(features,64) → ReLU → Dense(64,10), trained with MSE and Adam.

The epoch includes shuffling, batch assembly, gradient reset, forward, loss, backward,
and optimizer updates. It excludes model/data construction, console logging, image
augmentation, validation, metrics, and checkpoint IO. The model/optimizer and RNG are
reset at each JMH iteration and train continuously across operations within that
iteration; nonfinite loss fails the run. This is a complete synthetic training loop,
not a claim about full MNIST application performance. GC profiling can also observe
iteration-setup allocation, so treat it as approximate epoch allocation.

Timing is ms/epoch; samples/second = `samples * 1000 / msPerEpoch`.
Results go to `benchmarks/build/results/training-vector.json` or `training-java.json`.

```bash
./gradlew :benchmarks:training --args='-p features=128 -p samples=1024 -p batch=32'
```

## Find where training spends time and allocates

```bash
./gradlew :benchmarks:profile
./gradlew :benchmarks:profile -PtensorBackend=java
```

The profile uses the same synthetic architecture and reports three independent
seeded training runs in one JVM, each with five warmup epochs and five measured
epochs. It reports ms/epoch, samples/second, current-thread bytes/epoch, GC
count/time, final loss, and min/median/max epoch time across runs. Phase counters
separate loading/shuffling, gradient reset, forward/loss, backward, and optimizer.
All runs use the same seed and starting model. Arguments are positional and positive:

```bash
# warmupEpochs runs measuredEpochs features samples
./gradlew :benchmarks:profile --args='10 5 10 784 1024'
```

Phase times include any GC pauses occurring during that phase.
Phase timers perturb execution; use them to locate bottlenecks, then confirm changes
with the uninstrumented JMH training workload. Phase percentages exclude loop and
instrumentation overhead. Byte counters cover the current thread; GC counters cover
the JVM. Neither is peak heap usage. This diagnostic does not make a statistical
confidence claim from the three same-JVM runs.

For call stacks, allocation sites, GC pauses and heap observations, record JFR:

```bash
mkdir -p benchmarks/build/results
./gradlew :benchmarks:profile -PprofileJfr=benchmarks/build/results/training-vector.jfr
```

Open the recording in a JFR viewer such as JDK Mission Control. Profiling samples
and post-GC heap observations are not an exact peak-live-memory measurement.
Keep profiling recordings separate from clean timing comparisons and avoid running
other benchmarks or training processes at the same time.

## Inference

```bash
./gradlew :benchmarks:inference
./gradlew :benchmarks:inference -PtensorBackend=java
```

Each operation predicts one batch through Dense(features,64), ReLU, Dropout(.2),
and Dense(64,10). Prediction disables dropout. The sweep covers 128 and 784 features
with batch sizes 1 and 32. Seeded model/input construction occurs once per trial,
outside timing. JMH consumes the returned tensor; this includes the public prediction
path and all forward allocations, but excludes loading, metrics, and checkpoint IO.
Defaults match the kernel/training forks, warmup, measurements, heap and GC profiler.
Time is microseconds per prediction batch; normalized allocation is bytes per batch.
Results default to `benchmarks/build/results/inference-vector.json` or
`inference-java.json`. Use `-rff` to preserve comparisons.

## Transposed-right matrix multiplication

```bash
./gradlew :benchmarks:kernel --args='.*TransposedMatmulBenchmarks.matmul$'
./gradlew :benchmarks:kernel --args='.*TransposedMatmulBenchmarks.(packed|panelPacked|packOnly|panelPackOnly)$'
```

Shapes are `rows x reduction x outputColumns`, for example `32x64x784` models
`g.matmul(weights.transpose())` with 32 samples, 64 outputs and 784 inputs.
Override with `-p shape=4x16x16,32x10x64`. Public `matmul` includes dispatch, tensor
allocation and any packing; `packed` and `panelPacked` compare raw whole-matrix
and 64-column panel packing plus multiplication. `packOnly` and `panelPackOnly`
isolate packing, including scratch allocation. The latter consumes each reused
panel through a Blackhole. Raw candidates omit tensor wrappers and shape checks.
`direct` retains an experimental gather kernel that was rejected on the measured
M4 Pro/JDK combination. All raw kernel candidates require the VECTOR backend;
public `matmul` can also run with `-PtensorBackend=java`.

These benchmarks report microseconds per multiplication (or packing operation)
and bytes/op through the GC profiler. Use the usual `-rff` option to preserve
results. Complete training epochs remain the check for application-level benefit.

## Reusable gradient storage

```bash
./gradlew :benchmarks:kernel --args='.*GradientStorageBenchmarks.backward$'
./gradlew :benchmarks:kernel -PtensorBackend=java --args='.*GradientStorageBenchmarks.backward$'
```

Measures reset plus repeated backward through a shared 4,096-element addition graph.
Graph construction and seed creation are outside timing. Read the
[OPT-10 results](GRADIENT_STORAGE_RESULTS.md) for ownership semantics, full-epoch
comparisons and raw allocation/timing data.

## OPT-11 kernel coverage

```bash
./gradlew :benchmarks:kernel --args='.*CoverageBenchmarks.run'
./gradlew :benchmarks:kernel -PtensorBackend=java --args='.*CoverageBenchmarks.run'
./gradlew :benchmarks:kernel --args='.*SigmoidCoverageBenchmarks.*'
```

`CoverageBenchmarks` covers reductions, higher-rank singleton broadcasting,
strided map/copy, broadcast batched matmul with contiguous/transposed right
operands, and explicit unary operations. Select cases with `-p operation=axisSum,min`
or another operation name. `SigmoidCoverageBenchmarks` compares dispatch with the
retained Java callback at 16, 128, and 4,096 elements (`-p size=...`).
See [kernel coverage results](KERNEL_COVERAGE_RESULTS.md) for the retained and
rejected candidates, reduction order, tolerances, and matched before/after commands.
