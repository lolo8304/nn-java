# Inference without autograd graphs (OPT-05)

Measured 2026-09-17 on Java 26.0.2. Baseline: `dbca4ad` with the new inference
benchmark; after: the OPT-05 working tree. Raw JMH JSON is retained in
[results/opt05](results/opt05).

`GradMode.noGrad(Supplier<T>)` disables recording on the current thread and restores
the previous mode in a finally block. Scopes nest and do not affect other threads.
Every `Variable` operation checks recording and input gradient requirements before
allocating a parent list or backward closure. Unrecorded outputs are ordinary
constant variables without references to the preceding graph. Operations on constant
inputs also skip recording outside a no-grad scope. Tensor views still share their
backing data according to the existing tensor API.

`Sequential.predict` and `evaluate` enter no-grad scopes automatically, including
custom layers and the compiled evaluation loss. `predictClasses` inherits the
prediction behavior. The `training` argument remains independent: an explicit
`forward(input, false)` can still be differentiated, and `forward(input, true)`
inside no-grad still applies dropout. No-grad does not change parameter flags,
existing gradients, weights or optimizer state, and does not prohibit backward
through graphs that were already recorded.

## Reproduction

Run sequentially on the baseline and updated implementation:

```sh
./gradlew :benchmarks:inference --args='-rff benchmarks/results/opt05/after-vector.json'
./gradlew :benchmarks:inference -PtensorBackend=java --args='-rff benchmarks/results/opt05/after-java.json'
```

Use `before-*.json` paths on the baseline. Retain `InferenceBenchmarks` and its
runner/task wiring in both versions. Each operation predicts one batch through
Dense(features,64), ReLU, Dropout(.2), Dense(64,10). Dropout is inactive during
prediction. Input features are 128 or 784; batch size is 1 or 32. Model and inputs
are seeded and constructed outside timing once per trial. JMH consumes the returned
tensor. No loading, metrics, backward, updates, augmentation or checkpoint IO is timed.

Settings: two JVM forks, three one-second warmup iterations and five one-second
measurement iterations per fork, one thread, 256–512 MiB heap, GC profiler.
Latency is microseconds per prediction batch; allocation is bytes per batch.
Confidence intervals are JMH's reported 99.9% intervals. Allocation measures bytes
allocated, not peak retained memory. Results apply to this small synthetic network.

## Measurements

| Backend | Batch | Features | Before µs/batch | After µs/batch | Bytes/batch before → after | Allocation reduction |
|---|---:|---:|---:|---:|---:|---:|
| VECTOR | 1 | 128 | 1.519 ± 0.031 | 1.596 ± 0.196 | 3,272 → 2,544 | 22.2% |
| VECTOR | 1 | 784 | 9.222 ± 0.548 | 9.179 ± 0.316 | 3,272 → 2,544 | 22.2% |
| VECTOR | 32 | 128 | 45.346 ± 4.875 | 50.093 ± 26.261 | 55,896 → 55,163 | 1.3% |
| VECTOR | 32 | 784 | 318.439 ± 72.700 | 245.179 ± 12.054 | 55,898 → 55,154 | 1.3% |
| JAVA | 1 | 128 | 5.431 ± 7.970 | 2.200 ± 0.029 | 3,320 → 2,592 | 21.9% |
| JAVA | 1 | 784 | 10.719 ± 1.466 | 9.628 ± 0.116 | 3,416 → 2,688 | 21.3% |
| JAVA | 32 | 128 | 66.362 ± 1.415 | 65.049 ± 1.591 | 56,064 → 55,311 | 1.3% |
| JAVA | 32 | 784 | 319.973 ± 50.717 | 300.346 ± 1.763 | 56,042 → 55,266 | 1.4% |

Allocation falls consistently. The relative benefit is largest for single-sample
prediction; at batch size 32 the forward tensor buffers dominate allocation.
Variable wrappers and tensor results remain allocated. The measurements do not
establish a general latency speedup: several intervals overlap and some runs have
large timing outliers (especially the baseline Java 128-feature single-sample case
and updated Vector 128-feature batch-32 case). All runs are retained as measured;
no samples were excluded. Use the allocation results as the demonstrated benefit,
and repeat timing on an otherwise idle machine before making latency claims.

## Verification

`./gradlew :tensor:check :nn:check :benchmarks:check` passes 65 tests on each
backend (130 executions), including existing finite-difference gradients, repeated
backward, training convergence and checkpoint/resume checks. Seven new inference
tests cover every autograd operation with contiguous and strided inputs, exact
prediction parity, absence of parent references/backward closures, unchanged
parameter gradients and weights, evaluation optimizer state, exception restoration,
nested scopes, thread isolation, independent dropout mode and RNG consumption,
reusing detached outputs as constants in later training, and backward through a
previously recorded graph inside no-grad. Existing training and explicit evaluation-mode
forward calls remain differentiable.

The repository-wide `./gradlew check` still fails the previously documented
`MnistResumeTest.optionsAndCompatibilityChecks` mismatch: expected 50 default
epochs, actual 80. The unrelated example/test mismatch was left unchanged.
