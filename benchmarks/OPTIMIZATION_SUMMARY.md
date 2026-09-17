# Performance and allocation summary through OPT-10

Latest implementation: OPT-10 working tree, 2026-09-17. This summary includes the work
before the numbered roadmap and every completed item, OPT-01 through OPT-10.

**Latest: reusable gradient storage (OPT-10)**

Controlled before/after runs against `69bd410`, using two JVM forks, three
one-second warmups and five one-second measurements per fork, show **99.6% less
allocation** in repeated shared-graph backward (198,328 → 808 B/op). Backward time
falls 41.0% with Vector and 30.0% with Java. Fresh-graph epochs save **20–34% allocation**.

| Backend | Features | Before → after ms/epoch | Before → after MB/epoch |
|---|---:|---:|---:|
| VECTOR | 128 | 3.503 ± 0.089 → 3.564 ± 0.020 | 11.381 → 9.086 |
| VECTOR | 784 | 15.987 ± 0.136 → 16.298 ± 0.089 | 38.372 → 25.338 |
| JAVA | 128 | 5.145 ± 0.034 → 5.109 ± 0.026 | 11.230 → 8.935 |
| JAVA | 784 | 20.578 ± 0.256 → 20.364 ± 0.121 | 38.254 → 25.216 |

There is no general epoch speedup: timing intervals overlap except for the
784-feature Vector workload, which is 1.9% slower in this run. Buffers remain owned
by each variable across resets; `grad()` is null after reset and returns live mutable
storage after backward. Retention may increase memory lifetime despite reducing
allocation. See [full measurements, raw data and API semantics](GRADIENT_STORAGE_RESULTS.md).

The remaining tables preserve the historical measurements through OPT-09 at `4a4edcb`.
JAVA means the Java backend without the incubator Vector module; VECTOR means
explicit Vector API kernels where supported, with Java fallbacks elsewhere.

All memory figures below are **allocated bytes per operation or epoch**, not peak
heap, retained memory, or process RSS. A reduction in allocation means less allocation
and GC pressure; it does not establish an equal reduction in RAM requirements.
Time reduction is `(before - after) / before`; speedup is `before / after`.
For example, 50% less time means 2× throughput for the same amount of work.
Percentages are calculated from recorded means and rounded for display.

**Historical full-epoch results after OPT-09**

Fresh runs at `4a4edcb` on Apple M4 Pro (arm64), OpenJDK 26.0.2. Each operation
trains 1,024 synthetic samples, batch 32, Dense(features,64) → ReLU → Dense(64,10),
MSE and Adam. Includes shuffle/loading, forward, backward and updates; excludes
model construction, augmentation, validation and checkpoint IO. This exercises the
training changes through OPT-09, including fused ReLU backward. OPT-05's inference
benefit and OPT-08's classification benefit are reported separately below.

Two JVM forks, three one-second warmups and five one-second measurements per fork,
one thread, 256–512 MiB heap, GC profiler. Backends ran sequentially without
concurrent tests or training. ± is JMH's reported 99.9% confidence interval
half-width. GC allocation may include amortized iteration setup.

| Features | JAVA ms/epoch | VECTOR ms/epoch | VECTOR time reduction vs JAVA | VECTOR speedup | JAVA → VECTOR MB allocated/epoch |
|---:|---:|---:|---:|---:|---:|
| 128 | 5.101 ± 0.132 | 3.441 ± 0.078 | **32.5%** | **1.48×** | 11.230 → 11.381 |
| 784 | 20.191 ± 0.173 | 15.681 ± 0.139 | **22.3%** | **1.29×** | 38.251 → 38.370 |

Vector is faster here but allocates slightly more: **1.35%** at 128 features and
**0.31%** at 784 features, consistent with its extra packed-panel storage on
smaller transposed-right products. Backend selection is not itself a memory-saving
switch. Fresh raw results: [JAVA](results/summary-opt09/java.json),
[VECTOR](results/summary-opt09/vector.json); console output:
[JAVA](results/summary-opt09/java.txt), [VECTOR](results/summary-opt09/vector.txt).

Relative to the recorded OPT-01/02 baseline, before fused optimizers, at 128 features:

| Backend | ms/epoch baseline → current | Time reduction | Speedup | MB/epoch baseline → current | Allocation reduction |
|---|---:|---:|---:|---:|---:|
| JAVA | 15.082 → 5.101 | **66.2%** | **2.96×** | 52.602 → 11.230 | **78.7%** |
| VECTOR | 13.390 → 3.441 | **74.3%** | **3.89×** | 52.598 → 11.381 | **78.4%** |

This is a **historical-to-current comparison**, using the same documented workload,
JDK and JMH settings; the baseline revision was not freshly rerun. The percentages
summarize the recorded trajectory, not a newly controlled paired experiment.
The original pre-roadmap implementation has no comparable full-epoch measurement,
so a total speedup from the initial implementation cannot be established.

Reproduce the historical OPT-09 runs at that revision from the repository root:

```sh
mkdir -p benchmarks/results/summary-opt09
./gradlew :benchmarks:training --args='.*TrainingBenchmarks.trainingEpoch$ -rff benchmarks/results/summary-opt09/vector.json'
./gradlew :benchmarks:training -PtensorBackend=java --args='.*TrainingBenchmarks.trainingEpoch$ -rff benchmarks/results/summary-opt09/java.json'
```


**Optimizations before OPT-01**

These changes are already present in the numbered roadmap's baseline. Their gains
must not be added to, or multiplied into, the later JMH results.

| Commit | Implemented change | Evidence or limit |
|---|---|---|
| `7149449` | Direct contiguous loops for map, arithmetic and total sum; bulk copy and array export | Reduced indexing and per-element allocations; historical measurements below |
| `7149449` | Reusable broadcast index cursor and direct stride arithmetic for matrix access, axis sum and softmax | Avoided allocating index arrays in inner loops |
| `7149449` | Softmax backward changed from a quadratic Jacobian calculation to `y * (g - sum(g*y))` | O(classes²) → O(classes) per row; historical softmax forward/backward was 14.09× faster |
| `7149449` | Skip gradient calculations for constant operands | Included in the historical backward results; no isolated percentage available |
| `7149449` | Lazily create Adam moments / SGD velocity; compute Adam bias corrections once per step; collect parameters once per epoch | Removed eager zero-buffer creation and repeated bookkeeping; no isolated percentage available |
| `94a382c` | Move the shared build and benchmark runtime to Java 26 | Allocation counts stayed unchanged in the recorded smoke run; timing was not a controlled JDK comparison |
| `996c8ae` | Selectable Vector backend, enabled by default: SIMD arithmetic, ReLU and supported matrix products, with Java fallbacks | Later controlled JMH comparisons provide stronger evidence than the initial smoke tests |

Broadcast-gradient and repeated-backward correctness fixes accompanied the first
change; they are not counted as independent performance gains.

Historical Java 25 comparison, baseline `1a88c1f` versus `7149449`:

| Workload | Before → after ms/op | Time reduction | Speedup | Before → after B/op | Allocation reduction |
|---|---:|---:|---:|---:|---:|
| Elementwise | 0.475 → 0.161 | 66.1% | 2.95× | 787,072 → 98,808 | 87.4% |
| Copy | 0.051 → 0.007 | 86.3% | 7.29× | 131,264 → 32,912 | 74.9% |
| Axis sum | 0.143 → 0.021 | 85.3% | 6.81× | 197,848 → 4,344 | 97.8% |
| Softmax forward/backward | 4.508 → 0.320 | 92.9% | 14.09× | 13,306,976 → 134,992 | 99.0% |
| Dense forward/loss/backward | 0.837 → 0.359 | 57.1% | 2.33× | 1,003,888 → 282,280 | 71.9% |

These were single-process smoke tests: 100 warmup and 100 measured operations,
32×128 inputs and 128×64 dense weights, on Apple M4 Pro / Java 25.0.2.
They lack confidence intervals. Allocation and algorithmic-complexity changes
are firmer evidence than timing ratios. [Historical source](README.md).

For the initial Vector implementation, later OPT-01/02 JMH measured the allocating
elementwise chain at **24.337 µs JAVA versus 3.697 µs VECTOR: 6.58×, or 84.8%
less time**. Allocation was essentially equal (~98.8 KB). With reusable output
buffers the same chain was **21.842 versus 1.306 µs: 16.72×, or 94.0% less time**.
These are specific multiply → broadcast-add → ReLU workloads, not whole-model
speedups. Copy uses shared bulk-copy logic, so its backend timing differences do
not demonstrate SIMD acceleration. [JMH source](OUTPUT_BUFFER_RESULTS.md).

**Completed roadmap: measured gains per change**

Each row uses its own recorded before/after comparison. The rows cover different
operations and baselines; their percentages are not additive. Epoch figures below
use 1,024 samples, batch 32, a features → 64 → 10 network and Adam. Unless stated
otherwise, the loss is MSE. Allocation values use decimal MB/KB.

| Item | JAVA result | VECTOR result | Allocation effect |
|---|---|---|---|
| [OPT-01 — reliable measurements](OUTPUT_BUFFER_RESULTS.md) | Baseline 15.082 ms/epoch, 128 features | Baseline 13.390 ms/epoch; 11.2% less time than JAVA | ~52.6 MB/epoch; measurement infrastructure, no independent speedup claimed |
| [OPT-02 — reusable output buffers](OUTPUT_BUFFER_RESULTS.md) | Elementwise: 24.337 → 21.842 µs; **10.3% less time**, 1.11×; timing intervals overlap | Elementwise: 3.697 → 1.306 µs; **64.7% less time**, 2.83× | JAVA 98,856 → 72 B (**99.93% less**); VECTOR 98,832 → 48 B (**99.95% less**). Copy-into ~0 B on both |
| [OPT-03 — fused Adam/SGD](FUSED_OPTIMIZER_RESULTS.md) | 128-feature Adam epoch: 15.004 → 7.558 ms; **49.6% less time**, 1.99× | Same epoch: 13.475 → 6.436 ms; **52.2% less time**, 2.09× | ~52.6 → 20.5 MB/epoch (**61.1% less**); optimizer phase 32,110,592 → 5,120 B (**99.984% less**). SGD implemented but not separately timed |
| [OPT-04 — bulk batches](BULK_BATCH_RESULTS.md) | 128-feature epoch: 7.25 → 5.63 ms; **22.3% less time**, 1.29× | Same epoch: 6.006 → 4.288 ms; **28.6% less time**, 1.40× | ~20.48 → 13.89 MB/epoch (**32.2% less**). Loader 8.228 → 1.636 MB (**80.1% less**) |
| [OPT-05 — no-grad inference](NO_GRAD_RESULTS.md) | Single-sample allocation **21.3–21.9% lower** | Single-sample allocation **22.2% lower** | ~728 B saved per single-sample prediction; batch-32 allocation **1.3–1.4% lower**. No general latency gain established |
| [OPT-06 — transposed-right SIMD matmul](TRANSPOSED_MATMUL_RESULTS.md) | Backend unchanged by this item | Eligible kernels **2.50–3.99× faster** (~60–75% less time); 128-feature epoch: 4.204 → 3.931 ms; **6.5% less time**, 1.07× | Tradeoff: 13.861 → 14.016 MB/epoch (**1.1% more**) for fresh packed panels |
| [OPT-07 — cache blocking](BLOCKED_MATMUL_RESULTS.md) | 784-feature epoch: 22.367 → 21.084 ms; **5.7% less time**, 1.06× | Same epoch: 18.785 → 16.976 ms; **9.6% less time**, 1.11× | Epoch allocation essentially unchanged; eligible strided-right scratch bounded to **32 KiB**. No established 128-feature epoch gain |
| [OPT-08 — fused cross entropy](FUSED_CROSS_ENTROPY_RESULTS.md) | Loss forward/backward: 8.465 → 3.621 µs; **57.2% less time**, 2.34× | Loss forward/backward: 8.439 → 3.622 µs; **57.1% less time**, 2.33× | Loss allocation ~52.2 → 16.8 KB (**67.8% less**). Classification epochs save ~1.14 MB (**7.7–7.8%** at 128 features; **2.7%** at 784) |
| [OPT-09 — fused activation backward](FUSED_ACTIVATION_RESULTS.md) | First contribution **5.06–16.40×** faster in short kernel measurements | First contribution **4.96–16.32×** faster in short kernel measurements | First contribution **80.0–83.3% less allocation**; adding to an existing gradient ~0 B. Saved activations remain available to other branches |
| [OPT-10 — reusable gradient storage](GRADIENT_STORAGE_RESULTS.md) | Repeated backward **41.0% faster**; no general epoch speedup | Repeated backward **30.0% faster**; epoch timing intervals overlap | Repeated backward **99.6% less allocation**; full epochs **20–34% less allocation** |

Additional scope behind these results:

- OPT-02 copy reuse reduced time from **1.005 → 0.646 µs JAVA (35.7%)** and
  **1.008 → 0.605 µs VECTOR (40.0%)**, while removing ~32.9 KB allocation per copy.
- OPT-03's instrumented optimizer phase dropped **7.991 → 0.288 ms JAVA (96.4%)**
  and **7.801 → 0.321 ms VECTOR (95.9%)**. These phase timings include instrumentation
  and must not replace the JMH full-epoch timings.
- OPT-04 loader time fell **1.270 → 0.101 ms JAVA (92.0%)** and
  **1.287 → 0.099 ms VECTOR (92.3%)** at 128 features. At 784 features,
  full-epoch time fell **29.827 → 22.125 ms JAVA (25.8%)** and
  **25.980 → 18.880 ms VECTOR (27.3%)**; epoch allocation fell about **49%**.
- OPT-07's larger transposed-right kernel sweep showed **3.41–4.21× JAVA** and
  **1.17–1.32× VECTOR** point-estimate speedups, with wide intervals on some cases.
  Vector scratch for the 32×784×64 example fell **401,408 → 32,768 payload bytes
  (91.8% less)**. Java adds that 32 KiB scratch where it previously used none.
  Large contiguous products did not uniformly improve.
- OPT-08 improved the 128-feature Vector classification epoch **4.243 → 4.090 ms
  (3.6%)**. Java and 784-feature epoch timing intervals overlap. It changes the
  old smoothed loss to standard unsmoothed cross entropy, so identical saturated
  training trajectories are not promised. Its results are separate from MSE.

**OPT-09: all four activation kernels**

4,096 contiguous elements; first contribution includes allocating the destination.
Times below exclude forward passes and graph traversal. These short runs use one
fork, two 200 ms warmups and three 200 ms measurements; speedups are indicative.
The new kernels use scalar loops on both backends, so similar final times are
expected and small differences are not evidence of a Vector advantage.

| Activation | JAVA µs, old → new | JAVA time reduction / speedup | VECTOR µs, old → new | VECTOR time reduction / speedup | Allocation, either backend |
|---|---:|---:|---:|---:|---|
| relu | 27.008 → 1.660 | 93.9% / 16.27× | 26.597 → 1.630 | 93.9% / 16.32× | 197.2 → 32.9 KB; **83.3% less** |
| leakyRelu | 26.687 → 1.627 | 93.9% / 16.40× | 26.620 → 1.632 | 93.9% / 16.31× | 197.2 → 32.9 KB; **83.3% less** |
| sigmoid | 6.085 → 1.202 | 80.3% / 5.06× | 6.037 → 1.217 | 79.8% / 4.96× | 164.7 → 32.9 KB; **80.0% less** |
| tanh | 15.245 → 1.214 | 92.0% / 12.55× | 14.849 → 1.195 | 92.0% / 12.43× | 164.7 → 32.9 KB; **80.0% less** |

Accumulating into an existing destination removes approximately **164.7–197.2 KB
per contribution** (effectively 100% of these kernel allocations). Measured time
reductions are **88.6–95.0% JAVA** and **91.7–95.0% VECTOR**. This does not remove
the surrounding graph's allocations or implement gradient reuse across resets.
[Full activation results and raw samples](FUSED_ACTIVATION_RESULTS.md).

**How to interpret the comparison**

Most earlier JMH epoch comparisons used two forks, three one-second warmups and
five one-second measurements per fork, with a 256–512 MiB heap and GC profiler.
OPT-07 used four measured epoch iterations; its kernel sweep and OPT-09 used shorter
settings. See the linked result files for confidence intervals, exact workloads and
raw samples. Historical smoke tests are intentionally kept separate.

The largest demonstrated full-epoch gains came from removing optimizer temporaries
and bulk-loading batches. Vector acceleration helps selected arithmetic and matrix
kernels, while the same allocation reductions generally benefit both backends.
The next planned work is OPT-11: profile-guided kernel coverage. Peak-memory and
real MNIST augmentation/validation/checkpoint measurements remain unmeasured here.

The implementation through OPT-10 passes **98 tests per backend (196 executions)**,
including numerical gradients, alias/stride behavior, graph branches, repeated
backward, training and checkpoint tests. OPT-10 also verifies storage reuse, aliased seeds, stale nonfinite gradients and
absent versus zero gradients with optimizer momentum.
