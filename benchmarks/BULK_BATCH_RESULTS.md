# Bulk batch assembly (OPT-04)

Measured 2026-09-17 on Java 26.0.2, baseline `7de3fda` versus the
OPT-04 working tree. Raw JMH JSON is retained in [results/opt04](results/opt04).

`DataLoader` now allocates independent batch tensors and uses checked `copyInto`
operations on their row views. Contiguous samples use `System.arraycopy`; strided
samples use the existing reusable index cursor. Fetching and shuffling are unchanged:
all samples in a batch are fetched once before any copying. No dataset-specific
shortcut bypasses transforms or `Dataset.get`.

Exact sample shapes are checked, including empty tensors. This tightens the old
loader's incomplete validation: previously a larger sample could be silently
truncated, and mismatched empty shapes could escape validation entirely. A rejected
batch is never returned. Scalars, nonzero offsets, transposed samples, empty samples,
and final partial batches are supported. Returned batches own their storage.

## Reproduction

Run the same benchmark source on the baseline loader and updated loader, sequentially:

```sh
./gradlew :benchmarks:training --args='-rff benchmarks/results/opt04/after-vector.json'
./gradlew :benchmarks:training -PtensorBackend=java --args='-rff benchmarks/results/opt04/after-java.json'
./gradlew :benchmarks:profile
./gradlew :benchmarks:profile -PtensorBackend=java
```

For the baseline, retain the new `loadingEpoch` benchmark and restore only
`DataLoader.java` from `7de3fda`; use `before-*.json` output paths. Each JMH operation
loads or trains over 1,024 samples, batch size 32, with 128 or 784 input features.
Loading includes shuffle and consumes every batch through a Blackhole. Training
uses Dense(features,64), ReLU, Dense(64,10), MSE, and Adam. Dataset/model setup is
outside timing; the GC profiler can observe iteration setup, so normalized allocation
is approximate. These measurements exclude augmentation, validation and checkpoint IO.

JMH settings: two forks, three one-second warmup iterations and five one-second
measurement iterations per fork, one thread, 256–512 MiB heap, GC profiler. Tables
report average time and JMH's 99.9% confidence interval, with decimal MB.
The initial Vector baseline had unstable timing (128-feature training was
20.449 ± 21.366 ms/epoch); it is retained as `before-vector-initial.json` but excluded
from the comparison. The Vector baseline was repeated with identical settings.

## Measurements

| Backend | Features | Operation | Before ms/epoch | After ms/epoch | Speedup | MB/epoch before → after |
|---|---:|---|---:|---:|---:|---:|
| VECTOR | 128 | loadingEpoch | 1.287 ± 0.032 | 0.099 ± 0.001 | 12.97× | 8.228 → 1.636 |
| VECTOR | 784 | loadingEpoch | 7.111 ± 0.072 | 0.266 ± 0.004 | 26.71× | 46.093 → 7.022 |
| VECTOR | 128 | trainingEpoch | 6.006 ± 0.154 | 4.288 ± 0.096 | 1.40× | 20.478 → 13.884 |
| VECTOR | 784 | trainingEpoch | 25.980 ± 0.397 | 18.880 ± 0.087 | 1.38× | 79.791 → 40.899 |
| JAVA | 128 | loadingEpoch | 1.270 ± 0.029 | 0.101 ± 0.001 | 12.55× | 8.228 → 1.636 |
| JAVA | 784 | loadingEpoch | 7.192 ± 0.227 | 0.280 ± 0.015 | 25.69× | 45.894 → 7.012 |
| JAVA | 128 | trainingEpoch | 7.250 ± 0.125 | 5.630 ± 0.136 | 1.29× | 20.482 → 13.889 |
| JAVA | 784 | trainingEpoch | 29.827 ± 0.727 | 22.125 ± 0.089 | 1.35× | 79.819 → 40.926 |
At 128 features, loader allocation falls by 80% and full-epoch allocation by 32%.
At 784 features, the reductions are about 85% and 49%, respectively. The output
batch buffers and per-sample view metadata remain allocated; this change does not
reuse batch storage across iterations.

The new phase profiles ([Vector](results/opt04/profile-vector.txt),
[Java](results/opt04/profile-java.txt)) use three same-JVM runs, each with five
warmup and five measured epochs at 128 features. Median loading is 0.308 ms and
1.701 MB/epoch on Vector, and 0.315 ms and 1.701 MB/epoch on Java. Timers and
allocation counters perturb execution, so these times should not be substituted
for the JMH measurements. Vector backward now accounts for 52–60% of instrumented
epoch time and about 10.3 MB/epoch after warmup. All profile runs finish at loss
0.01980394 on both backends, matching the previously recorded OPT-03 profile.
Allocation counters are not peak live memory measurements.

## Verification

`./gradlew :tensor:check :nn:check :benchmarks:check` passes 58 tests on each
backend (116 executions). Loader coverage includes seeded shuffle order, input/target
pairing, fetch-once and fetch-all-before-copy behavior, exact shape rejection,
scalar and empty samples, offset/strided inputs and targets, independent batch storage,
empty datasets, iterator exhaustion, oversized batches and final partial batches.
Existing training tests cover deterministic full epochs and partial batches.

The repository-wide `./gradlew check` still fails the previously documented
`MnistResumeTest.optionsAndCompatibilityChecks`: expected default epochs 50, actual
80. That unrelated example/test mismatch was left unchanged.
