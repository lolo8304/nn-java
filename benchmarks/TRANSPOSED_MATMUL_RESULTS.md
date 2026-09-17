# Transposed-right Vector matmul (OPT-06)

Measured 2026-09-17 on Apple M4 Pro, Java 26.0.2, preferred DoubleVector species.
Baseline dispatch is `b3e97fe`; both measured trees include the new benchmark
harness and the shared output-stride helper refactor. Raw JMH JSON is retained in
[results/opt06](results/opt06).

## Implementation and selection

The VECTOR backend packs the current right operand into a temporary panel of at
most 64 output columns, then calls the existing SIMD-across-columns multiply with
the appropriate output offset/row stride. One panel array is reused within a call;
no packed data is retained between calls. Mutations through a tensor, a transpose
view, or an optimizer are therefore visible on the next multiplication.

The packed path is selected for 2D right operands with nonunit column stride when
rows >= 4, reduction >= 8 and output columns >= 16. Smaller products retain the
scalar fallback; contiguous right rows retain the existing Vector path. Both input
operands can have nonzero offsets, row gaps and nonunit inner strides. Batched
matmul dispatch is unchanged. Each output sums reduction terms in the original
order with separate multiplication and addition; no FMA or horizontal reduction
is introduced.

The cutoff is conservative, based on short exploratory measurements, not a
universal hardware optimum. The candidate sweep used one fork, two 300 ms warmups
and three 300 ms measurements per case. The public-path measurements below use
longer settings. The raw candidates omit Tensor wrapper/shape-check costs.

- A direct gather kernel preserves summation order but performed poorly on this
  JVM/hardware combination: 32x64x128 took about 1,637 µs and allocated 11.6 MB/op,
  versus about 118 µs for the scalar fallback. It remains only in the benchmark
  module as a reproducible rejected candidate.
- Fresh whole-matrix packing plus SIMD took about 33 µs for 32x64x128, but lost
  on a single row (about 4.95 versus 3.70 µs). The crossover sweep included rows
  2 and 4, and small 8/16/32-wide matrices; tiny products are left unchanged.
- For 32x64x784, panel packing took about 20 µs and used 32,784 bytes of scratch,
  versus about 33 µs and 401,424 bytes for whole-matrix packing. Including output
  allocation and multiplication, panels took about 234 µs and 233.5 KB/op versus
  240 µs and 602.2 KB/op for whole-matrix packing.
- Panels are a memory/performance tradeoff: 32x127x65 took about 55 µs with panels
  versus 41 µs with whole-matrix packing. The final one-column panel uses the
  scalar tail. Both were faster than the roughly 129 µs scalar baseline.

Scratch is `8 * reduction * min(outputColumns, 64)` bytes plus array overhead.
Unlike the previous scalar path, this optimization adds temporary allocation.

## Reproduction

Run sequentially, with no concurrent training or tests:

```sh
./gradlew :benchmarks:kernel --args='.*TransposedMatmulBenchmarks.matmul$ -rff benchmarks/results/opt06/after-kernel-vector.json'
./gradlew :benchmarks:training --args='.*TrainingBenchmarks.trainingEpoch$ -rff benchmarks/results/opt06/after-training-vector.json'
./gradlew :benchmarks:profile
```

For baseline measurements, retain the benchmark code and Vector helper, restore
only `Tensor.java` from `b3e97fe` (the scalar transposed-right dispatch), and use
`before-*` output paths. The original contiguous multiply was generalized with an
output offset/stride before either measured run, so it is common to both versions.

Public-path JMH settings: two forks, three one-second warmups and five one-second
measurements per fork, one thread, 256–512 MiB heap, GC profiler. Matrix shapes are
`rows x reduction x outputColumns`; right operands are transpose views. Input/model
construction is outside timing; packing and output allocation are included. Training
uses 1,024 samples, batch 32, Dense(features,64)/ReLU/Dense(64,10), MSE and Adam.
The input data itself does not require gradients, so complete epochs exercise the
32x10x64 transpose-right path for the hidden layer, not the standalone 32x64x784
input-gradient benchmark. No augmentation, validation or checkpoint IO is timed.
Allocation is bytes allocated, not peak live memory. JMH training iteration setup
can contribute to allocation counters. Tables use JMH's 99.9% confidence intervals.

To repeat exploratory comparisons, use the same benchmark class with the `direct`,
`packed`, `packOnly`, `panelPacked`, and `panelPackOnly` filters. The exact shape
parameters and timing settings are retained in each JSON file. Raw candidates
require the Vector module; the public `matmul` benchmark supports either backend.

## Public-path measurements

| Shape (m×k×n) | Before µs/op | After µs/op | Speedup | Bytes/op before → after |
|---|---:|---:|---:|---:|
| 1x64x128 | 3.681 ± 0.019 | 3.586 ± 0.036 | 1.03× | 1,168 → 1,168 |
| 32x64x128 | 117.939 ± 0.372 | 32.611 ± 0.156 | 3.62× | 32,913 → 65,648 |
| 32x64x784 | 718.304 ± 5.163 | 200.899 ± 2.614 | 3.58× | 200,853 → 233,585 |
| 32x10x64 | 10.446 ± 0.127 | 2.977 ± 0.026 | 3.51× | 16,528 → 21,616 |
| 32x127x65 | 128.387 ± 0.387 | 51.288 ± 5.644 | 2.50× | 16,785 → 81,776 |
| 1x256x256 | 35.877 ± 0.245 | 34.905 ± 0.290 | 1.03× | 2,192 → 2,192 |
| 128x256x256 | 4618.620 ± 22.807 | 1156.226 ± 51.211 | 3.99× | 262,320 → 393,400 |

Complete Vector training epochs:

| Features | Before ms/epoch | After ms/epoch | Speedup | MB/epoch before → after |
|---|---:|---:|---:|---:|
| 128 | 4.204 ± 0.023 | 3.931 ± 0.107 | 1.07× | 13.861 → 14.016 |
| 784 | 18.396 ± 0.919 | 18.020 ± 0.175 | 1.02× | 40.868 → 41.022 |
The 128-feature epoch uses about 6.5% less time (1.07× throughput), at roughly
0.16 MB more allocation per epoch from temporary panels. The 784-feature timing
intervals overlap, so its 1.02× point estimate is not evidence of a clear full-epoch
speedup. Single-row products retain the scalar algorithm; their small timing
variation should not be attributed to the packed kernel.

The [updated phase profile](results/opt06/profile-vector.txt) has median epoch time
4.501 ms across three same-JVM runs, each with five warmup and five measured epochs.
These instrumented times are not interchangeable with JMH timing. All three runs
finish at loss 0.01980394, matching previous profiles. Backward still dominates the
profile, so activation/gradient allocation remains a separate optimization target.

## Verification

`./gradlew check` passes 72 tests on each backend, 144 executions including the
examples. New tests compare bits against ordered scalar accumulation across empty
and singleton dimensions, multiple SIMD widths/tails, multiple panels and final
partial panels, left/right offsets, row gaps and nonunit inner strides. They also
cover cancellation, NaN, infinities and signed zeros; input preservation; retained
transpose views after direct/copy updates; and finite-difference input gradients
before and after an SGD step with strided weights. Existing training, gradient,
optimizer and checkpoint/resume checks pass. The JAVA backend remains the scalar
reference and runs without the Vector module.
