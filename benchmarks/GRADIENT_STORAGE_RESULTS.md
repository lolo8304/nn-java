# Reusable gradient storage (OPT-10)

Measured 2026-09-17 on Apple M4 Pro (arm64), OpenJDK 26.0.2. Baseline:
`69bd410a13f8934dfd1b83d66920ee03ea4341e7`; updated: this working tree.
Both revisions use the new `GradientStorageBenchmarks` harness unchanged.

Each variable retains an owned gradient tensor after its first contribution.
Further contributions use `addInto`; the first contribution after a reset uses
`copyInto`, or a fused activation kernel in overwrite mode. Initial allocation
still copies borrowed contributions to establish ownership. No incoming seed,
saved activation, or another variable's gradient is adopted as storage.
Intermediate gradients reset for each traversal; leaves accumulate until reset.

`zeroGrad()` still makes `grad()` return null, so optimizers skip absent gradients.
It retains the allocation without clearing its contents. First writes overwrite
all elements, including stale NaNs/infinities, rather than multiplying by zero.
A present zero gradient remains different from an absent gradient for momentum.
`grad()` returns live mutable storage: use `grad().copy()` for a snapshot that must
survive later backward calls or resets. Retaining buffers extends their lifetime
until the owning variable is collected; allocation savings are not a peak-memory
claim. Fresh graphs still allocate intermediate buffers; parameters reuse theirs
across batches, and retained graphs also reuse intermediate buffers.

## Measurement scope

Two independent JVM forks, three one-second warmups and five one-second measured
iterations per fork, one thread, GC profiler, 256–512 MiB heap. Before and after
runs use identical workloads and settings; all runs execute sequentially, without
concurrent test or training processes. Error bars below are JMH's 99.9% confidence
interval half-widths. Allocation is normalized bytes per operation; epoch allocation
can include amortized iteration setup.

The backward benchmark constructs a 4,096-element leaf and shared addition graph
outside timing: `shared = leaf + 1`, `output = shared + shared + leaf`. Each measured
operation resets the leaf, traverses backward with a fixed seed, and returns the
leaf gradient. It isolates gradient storage and traversal without arithmetic
contribution temporaries or graph construction. It is a favorable reuse workload,
not representative of every backward operation.

Full epochs train 1,024 synthetic samples, batch 32, Dense(features,64) → ReLU →
Dense(64,10), MSE and Adam. They include loading/shuffling, resets, forward/loss,
backward and updates; exclude augmentation, validation and checkpoint IO.

## Results

| Backend | Workload | Time before → after | Allocation before → after | Allocation reduction |
|---|---|---:|---:|---:|
| VECTOR | Repeated backward | 7.192 ± 0.032 → 4.247 ± 0.402 µs/op | 198,328 → 808 B/op | 99.6% |
| VECTOR | 128-feature epoch | 3.503 ± 0.089 → 3.564 ± 0.020 ms/epoch | 11.381 → 9.086 MB/epoch | 20.2% |
| VECTOR | 784-feature epoch | 15.987 ± 0.136 → 16.298 ± 0.089 ms/epoch | 38.372 → 25.338 MB/epoch | 34.0% |
| JAVA | Repeated backward | 7.186 ± 0.069 → 5.033 ± 0.295 µs/op | 198,328 → 808 B/op | 99.6% |
| JAVA | 128-feature epoch | 5.145 ± 0.034 → 5.109 ± 0.026 ms/epoch | 11.230 → 8.935 MB/epoch | 20.4% |
| JAVA | 784-feature epoch | 20.578 ± 0.256 → 20.364 ± 0.121 ms/epoch | 38.254 → 25.216 MB/epoch | 34.1% |

Repeated backward removes almost all tensor-sized gradient allocation; remaining
bytes include traversal and shape bookkeeping. Full epochs save less because their
intermediate graphs are constructed afresh each batch. These measurements establish
allocation savings, not a general full-epoch speedup. Repeated backward is 41.0%
faster with Vector and 30.0% faster with Java. Java epoch timing intervals overlap,
as do the 128-feature Vector intervals. The 784-feature Vector epoch is 1.9% slower
with non-overlapping reported intervals in this run; this small timing regression
is a tradeoff alongside the 34.0% allocation reduction, not an epoch speedup.

## Reproduction

Run the following at the baseline (with the new benchmark source copied in) and
updated revision, replacing `PHASE` with `before` or `after`:

```sh
./gradlew :benchmarks:kernel --args='.*(GradientStorageBenchmarks.backward|TrainingBenchmarks.trainingEpoch)$ -rff benchmarks/results/gradient-storage/PHASE-vector.json'
./gradlew :benchmarks:kernel -PtensorBackend=java --args='.*(GradientStorageBenchmarks.backward|TrainingBenchmarks.trainingEpoch)$ -rff benchmarks/results/gradient-storage/PHASE-java.json'
```

Raw results: [before Vector](results/gradient-storage/before-vector.json),
[after Vector](results/gradient-storage/after-vector.json),
[before Java](results/gradient-storage/before-java.json),
[after Java](results/gradient-storage/after-java.json).

## Verification

`./gradlew check` passes 98 tests per backend (196 executions, including unchanged
cached tensor checks). Tests cover shared roots, repeated traversals, leaf
accumulation, generic and fused activation buffer reuse after reset, nonuniform
and transposed seeds, overlapping gradient views, seed/value independence,
wrong-shape rejection before mutation, signed zero, stale nonfinite values,
broadcast/transpose contributions, empty and disabled gradients, and optimizer
behavior for absent versus zero gradients with existing momentum. Existing gradient
checks, convergence, inference, optimizer state and training resume tests pass.
