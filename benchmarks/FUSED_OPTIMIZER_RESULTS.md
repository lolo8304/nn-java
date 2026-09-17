# Fused optimizer updates (OPT-03)

Measured 2026-09-17 on Java 26.0.2, baseline `4937fc0` versus the OPT-03
implementation in this commit. Raw JMH JSON and complete console/profile logs are
retained in [results/opt03](results/opt03).

Adam now updates its owned first and second moments and the parameter in one
traversal. SGD does the same for velocity and parameter, including zero momentum.
Contiguous tensors use either Java loops or explicit Vector API kernels with scalar
tails. Strided tensors use one reusable index cursor. No tensor-sized temporary is
allocated on the ordinary optimizer path after state initialization. Arithmetic
keeps the previous operation order, bias correction and epsilon placement; no FMA
or approximation is introduced. Checkpoint layout is unchanged.

The tensor primitives require exact shapes and disjoint writable address ranges;
validation happens before writes. Read-only gradient inputs may alias an output:
exact layouts work in place, differently mapped overlapping views are snapshotted.
The address-range check conservatively rejects interleaved writable views whose
bounding ranges overlap. `Parameter.applyDelta` now uses the reusable add primitive,
which also fixes empty parameters and overlapping delta views.

## Reproduction

Run each command on both the baseline and updated tree, sequentially, with no
concurrent training/checks:

```sh
./gradlew :benchmarks:training --args='-p features=128 -p samples=1024 -p batch=32 -rff benchmarks/build/results/opt03-vector.json' :benchmarks:profile
./gradlew :benchmarks:training -PtensorBackend=java --args='-p features=128 -p samples=1024 -p batch=32 -rff benchmarks/build/results/opt03-java.json' :benchmarks:profile
```

JMH defaults: two forks, three 1-second warmups and five 1-second measurements per
fork, one thread, 256–512 MiB heap, GC profiler. Each operation is one epoch over
1,024 synthetic samples, 128 features, batch 32, Dense(128,64)/ReLU/Dense(64,10),
MSE and Adam. Setup is outside timing. No augmentation, validation or checkpoint IO.
The phase profile uses three runs, each with five warmup and five measured epochs;
its timings include instrumentation and any GC in that phase. Allocation is not
peak memory. Results describe this workload, not all models or SGD performance.

## Measurements

JMH average time and reported 99.9% confidence interval; decimal MB:

| Backend | Before ms/epoch | After ms/epoch | Speedup | MB/epoch before → after |
|---|---:|---:|---:|---:|
| VECTOR | 13.475 ± 0.330 | 6.436 ± 0.586 | 2.09× | 52.598 → 20.479 |
| JAVA | 15.004 ± 0.117 | 7.558 ± 0.083 | 1.99× | 52.602 → 20.482 |

Optimizer phase, median of the three instrumented runs:

| Backend | ms/epoch before → after | bytes/epoch before → after |
|---|---:|---:|
| VECTOR | 7.801 → 0.321 | 32,110,592 → 5,120 |
| JAVA | 7.991 → 0.288 | 32,110,592 → 5,120 |

Full-epoch allocation falls by about 61%. Optimizer allocation falls by over 99.98%;
the remaining small allocation includes per-step iteration/lambda overhead. The
final profile loss is 0.01980394 before and after on both backends. Backward and
batch loading now dominate the Vector profile, supporting OPT-04 as the next task.

## Verification

`./gradlew :tensor:check :nn:check :benchmarks:check` passes 53 tests on each
backend (106 executions). Coverage includes repeated Adam/SGD reference updates,
zero and nonzero momentum, absent/empty/scalar gradients, strided parameters,
offsets and vector tails, exact and transposed gradient aliases, output rejection
before mutation, gradient preservation, serialized moments/velocity, and restoring
optimizer state between steps. Existing tests cover byte-identical model checkpoint
round trips/resume, gradients and training convergence.

The repository-wide `./gradlew check` also found an unrelated failure in
`MnistResumeTest.optionsAndCompatibilityChecks`: the concurrently edited example
now defaults to 80/30 epochs while its test expects 50/20. That edit was preserved.
