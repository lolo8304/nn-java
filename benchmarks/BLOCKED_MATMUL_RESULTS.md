# Cache-blocked matrix multiplication (OPT-07)

Measured 2026-09-17 on Apple M4 Pro, OpenJDK 26.0.2. Baseline is
`30f7389`, with the new benchmark harness present in both versions. Raw JMH
results, including exploratory candidates, are in [results/opt07](results/opt07).

## Implementation

Rank-2 products select cache blocking when rows >= 16, reduction >= 128,
columns >= 32, and reduction × columns >= 32,768. The multiplication for this
cutoff uses `long` arithmetic. Smaller products retain their existing dispatch,
including OPT-06's Vector packing path. Batched and vector products are unchanged.
The threshold and tile sizes are conservative hardware-specific tuning choices,
not universal cache-size detection.

Contiguous right operands use tiles of up to 32 rows, 64 reduction terms and
512 columns, with no scratch allocation. The inner loop traverses contiguous
right/output rows. Left operands can be transposed, offset or otherwise strided.
Strided right operands use fresh 64 × 64 panels, reused across all row blocks
before packing the next reduction panel. Both backends use this packing on
eligible products. Scratch is bounded to 32,768 bytes of payload (32,784 bytes
including the array header on this JVM), instead of the Vector backend's previous
`8 * reduction * min(columns, 64)` payload. Nothing is cached between calls.

Each output continues accumulating terms in increasing reduction-index order,
starting at positive zero. Tiles resume from the previous partial result, use
separate multiplication and addition, and introduce no FMA or horizontal
reductions. This preserves ordered scalar arithmetic, including cancellation and
signed zeros; NaNs are compared with canonicalized NaN bits in tests.

## Method and reproduction

The full public-path sweep uses one fork, two 300 ms warmups and three 300 ms
measurements, one thread, 256–512 MiB heap and the GC profiler. These are short
exploratory measurements; confidence intervals can be wide. Shapes are
`rows x reduction x outputColumns`. Layout `left` transposes the left operand,
`right` transposes the right operand, and `contiguous` transposes neither. Setup
uses fixed seeds outside timing; output allocation and any packing are included.

```sh
./gradlew check
./gradlew :benchmarks:kernel --args='.*BlockedMatmulBenchmarks.matmul$ -f 1 -wi 2 -w 300ms -i 3 -r 300ms -rff benchmarks/results/opt07/after-vector.json'
./gradlew :benchmarks:kernel -PtensorBackend=java --args='.*BlockedMatmulBenchmarks.matmul$ -f 1 -wi 2 -w 300ms -i 3 -r 300ms -rff benchmarks/results/opt07/after-java.json'
./gradlew :benchmarks:training --args='.*TrainingBenchmarks.trainingEpoch$ -f 2 -wi 3 -i 4 -rff benchmarks/results/opt07/after-training-vector.json'
./gradlew :benchmarks:training -PtensorBackend=java --args='.*TrainingBenchmarks.trainingEpoch$ -f 2 -wi 3 -i 4 -rff benchmarks/results/opt07/after-training-java.json'
```

Run sequentially without concurrent tests/training. Create the result directory
first. For the baseline, use `Tensor.java` from `30f7389` with the same harness;
use `before-*` output names. The blocked kernel is unreachable in that version.
Training measurements use two forks, three one-second warmups and four one-second
measurements, with the same heap and GC profiler. Each epoch trains 1,024 samples,
batch 32, Dense(features,64)/ReLU/Dense(64,10), MSE and Adam. Input data does not
require gradients. No augmentation, validation or checkpoint IO is timed.
Allocation is allocated bytes, not peak live memory; iteration setup can
contribute to training allocation counters. Reported ± values are JMH's 99.9%
confidence intervals.

## Tuning observations

The initial 64-column contiguous tiles slowed the 256x512x512 case by roughly
35–50%, despite improving the 32x784x64 layer. Widening to 256 columns helped
moderate shapes but still regressed the largest shape. Four register-held Vector
accumulators through each reduction tile were also rejected: with the large
right-row stride, the largest contiguous case was about twice as slow as baseline.
The final implementation retains the original inner loop and uses 512-column
contiguous tiles. Packed panels remain 64 columns wide.

`candidate-64-*` records the original inner loop with 64-column tiles;
`candidate-256-java` and `candidate-256-vector` record 256-column contiguous tiles
with the original loop. `candidate-register-vector` records the rejected four-vector
accumulator experiment with 256-column contiguous tiles. `candidate-512-*` checks
the selected wider tiles before the final sweep. These timing artifacts describe
experiments, not alternative runtime configuration options.

## Public-path sweep

### VECTOR

| Layout | Shape | Before µs/op | After µs/op | Ratio before/after | Bytes/op before → after |
|---|---|---:|---:|---:|---:|
| contiguous | 1x256x256 | 8.35 ± 1.84 | 8.56 ± 2.92 | 0.98× | 2,144 → 2,144 |
| contiguous | 32x10x64 | 2.99 ± 0.14 | 3.09 ± 0.04 | 0.97× | 16,480 → 16,480 |
| contiguous | 32x128x64 | 31.58 ± 3.45 | 32.96 ± 9.25 | 0.96× | 16,481 → 16,481 |
| contiguous | 32x784x64 | 246.87 ± 57.17 | 195.58 ± 77.65 | 1.26× | 16,558 → 16,540 |
| contiguous | 64x32x128 | 30.25 ± 2.58 | 30.40 ± 5.83 | 1.00× | 65,633 → 65,633 |
| contiguous | 128x256x256 | 1073.03 ± 93.10 | 968.81 ± 192.80 | 1.11× | 262,337 → 262,334 |
| contiguous | 65x257x129 | 336.92 ± 50.35 | 268.73 ± 37.85 | 1.25× | 67,256 → 67,254 |
| contiguous | 256x512x512 | 8432.28 ± 329.09 | 8550.15 ± 581.49 | 0.99× | 1,048,937 → 1,048,939 |
| left | 1x256x256 | 8.51 ± 0.94 | 8.53 ± 1.20 | 1.00× | 2,144 → 2,144 |
| left | 32x10x64 | 3.04 ± 0.31 | 3.08 ± 0.16 | 0.99× | 16,480 → 16,480 |
| left | 32x128x64 | 31.75 ± 5.72 | 34.66 ± 1.64 | 0.92× | 16,481 → 16,481 |
| left | 32x784x64 | 270.53 ± 25.11 | 193.35 ± 78.15 | 1.40× | 16,558 → 16,539 |
| left | 64x32x128 | 30.26 ± 3.79 | 30.62 ± 9.12 | 0.99× | 65,633 → 65,633 |
| left | 128x256x256 | 1084.63 ± 87.32 | 986.30 ± 218.23 | 1.10× | 262,337 → 262,335 |
| left | 65x257x129 | 343.04 ± 34.53 | 269.46 ± 46.97 | 1.27× | 67,256 → 67,254 |
| left | 256x512x512 | 8752.87 ± 312.75 | 8589.95 ± 775.40 | 1.02× | 1,048,944 → 1,048,939 |
| right | 1x256x256 | 35.49 ± 2.17 | 36.14 ± 3.28 | 0.98× | 2,193 → 2,193 |
| right | 32x10x64 | 3.10 ± 0.15 | 3.14 ± 0.27 | 0.99× | 21,616 → 21,616 |
| right | 32x128x64 | 34.44 ± 2.60 | 34.69 ± 3.17 | 0.99× | 82,033 → 82,033 |
| right | 32x784x64 | 268.72 ± 20.80 | 212.71 ± 43.33 | 1.26× | 417,982 → 49,332 |
| right | 64x32x128 | 33.55 ± 1.15 | 33.85 ± 1.62 | 0.99× | 82,033 → 82,033 |
| right | 128x256x256 | 1219.91 ± 184.61 | 1043.41 ± 154.42 | 1.17× | 393,428 → 295,120 |
| right | 65x257x129 | 450.96 ± 133.68 | 341.80 ± 25.82 | 1.32× | 198,858 → 100,040 |
| right | 256x512x512 | 10044.98 ± 652.80 | 8468.92 ± 595.06 | 1.19× | 1,311,135 → 1,081,721 |

### JAVA

| Layout | Shape | Before µs/op | After µs/op | Ratio before/after | Bytes/op before → after |
|---|---|---:|---:|---:|---:|
| contiguous | 1x256x256 | 8.63 ± 0.68 | 8.62 ± 1.96 | 1.00× | 2,192 → 2,192 |
| contiguous | 32x10x64 | 4.02 ± 0.41 | 4.00 ± 0.22 | 1.01× | 16,528 → 16,528 |
| contiguous | 32x128x64 | 45.84 ± 10.95 | 46.03 ± 11.60 | 1.00× | 16,529 → 16,529 |
| contiguous | 32x784x64 | 281.82 ± 10.35 | 236.36 ± 17.76 | 1.19× | 16,535 → 16,557 |
| contiguous | 64x32x128 | 37.85 ± 2.49 | 38.05 ± 5.63 | 0.99× | 65,681 → 65,681 |
| contiguous | 128x256x256 | 1106.50 ± 106.60 | 1107.98 ± 279.74 | 1.00× | 262,313 → 262,338 |
| contiguous | 65x257x129 | 314.20 ± 38.60 | 280.84 ± 36.01 | 1.12× | 67,231 → 67,254 |
| contiguous | 256x512x512 | 8483.89 ± 307.88 | 8519.42 ± 389.98 | 1.00× | 1,048,911 → 1,048,939 |
| left | 1x256x256 | 8.60 ± 1.50 | 8.69 ± 1.92 | 0.99× | 2,192 → 2,192 |
| left | 32x10x64 | 4.03 ± 0.35 | 3.98 ± 0.37 | 1.01× | 16,528 → 16,528 |
| left | 32x128x64 | 45.58 ± 9.88 | 45.67 ± 10.15 | 1.00× | 16,529 → 16,529 |
| left | 32x784x64 | 286.99 ± 70.64 | 233.09 ± 27.10 | 1.23× | 16,535 → 16,557 |
| left | 64x32x128 | 37.65 ± 2.63 | 37.63 ± 4.21 | 1.00× | 65,681 → 65,681 |
| left | 128x256x256 | 1131.02 ± 167.51 | 1103.37 ± 398.31 | 1.03× | 262,314 → 262,337 |
| left | 65x257x129 | 320.64 ± 18.93 | 282.18 ± 20.52 | 1.14× | 67,231 → 67,255 |
| left | 256x512x512 | 8897.35 ± 326.88 | 8582.84 ± 489.09 | 1.04× | 1,048,944 → 1,048,941 |
| right | 1x256x256 | 35.81 ± 2.26 | 36.03 ± 2.46 | 0.99× | 2,193 → 2,193 |
| right | 32x10x64 | 10.33 ± 1.04 | 10.51 ± 0.58 | 0.98× | 16,528 → 16,528 |
| right | 32x128x64 | 128.30 ± 36.69 | 128.55 ± 4.40 | 1.00× | 16,531 → 16,531 |
| right | 32x784x64 | 1016.78 ± 165.54 | 248.84 ± 21.38 | 4.09× | 16,551 → 49,342 |
| right | 64x32x128 | 119.27 ± 15.87 | 117.88 ± 4.53 | 1.01× | 65,683 → 65,683 |
| right | 128x256x256 | 4553.56 ± 226.57 | 1245.38 ± 199.00 | 3.66× | 262,392 → 295,125 |
| right | 65x257x129 | 1190.47 ± 118.32 | 349.37 ± 38.54 | 3.41× | 67,252 → 100,040 |
| right | 256x512x512 | 42219.46 ± 2389.22 | 10032.02 ± 907.90 | 4.21× | 1,049,615 → 1,081,754 |

The short sweep shows about 1.17–1.32× point-estimate gains on the four
eligible transposed-right Vector shapes and 3.41–4.21× on Java. The larger
contiguous 256x512x512 case is effectively unchanged. Several other kernel
intervals overlap; do not read every point-estimate ratio as a statistically
established speedup. Small-path timing changes are noise/code-layout effects,
not benefits from blocking.

For example, Vector 32x784x64 with a transposed right operand reduces scratch
from 401,408 to 32,768 payload bytes (about 368.6 KB less allocation per call).
Java adds a 32 KiB panel on eligible strided-right products in exchange for the
measured speedup. Contiguous products allocate no new scratch.

## Complete training epochs

| Backend | Features | Before ms/epoch | After ms/epoch | Ratio before/after | MB/epoch before → after |
|---|---:|---:|---:|---:|---:|
| VECTOR | 128 | 4.098 ± 0.084 | 4.089 ± 0.035 | 1.00× | 14.017 → 14.016 |
| VECTOR | 784 | 18.785 ± 0.293 | 16.976 ± 0.133 | 1.11× | 41.029 → 41.014 |
| JAVA | 128 | 5.695 ± 0.072 | 5.731 ± 0.058 | 0.99× | 13.865 → 13.865 |
| JAVA | 784 | 22.367 ± 0.097 | 21.084 ± 0.111 | 1.06× | 40.904 → 40.892 |

The 784-feature epoch takes 9.6% less time with Vector and 5.7% less time with
Java. The 128-feature intervals overlap, so no full-epoch improvement is
established for that smaller workload. Allocation is essentially unchanged;
small differences include iteration-setup amortization. These synthetic epochs
exercise blocked forward products, while the new gradient test separately covers
large blocked input-gradient and weight-gradient products.

## Verification

`./gradlew check` passes 76 tests per backend, 152 executions total. The Java
suite runs without the Vector module. `BlockedMatmulTest` compares every output
against ordered scalar accumulation using exact double bits (canonical NaNs),
covering the dispatch boundary, row/reduction/column tile tails, both transpose
directions together and separately, offsets, row gaps, nonunit strides, input
preservation, cancellation, NaN, infinities, signed zero and mutations through
retained views. Existing empty/singleton/small-matrix coverage remains in place.
The added large finite-difference test covers forward, input-gradient and
weight-gradient products that all cross the blocking threshold, with 1e-8 absolute
tolerance. Existing optimizer, convergence and checkpoint/resume tests also pass.
