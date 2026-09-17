# OPT-11: optimized kernel coverage (OPT-16–OPT-20)

Baseline: `30b7905`, with `CoverageBenchmarks` added before changing the kernels.
Implementation: accompanying OPT-11 working-tree changes. All timings include public
API dispatch and result allocation. Raw JMH data is in [results/opt11](results/opt11).

## Implementation and numerical behavior

- **OPT-16 — reductions:** contiguous extrema use direct Java loops or Vector
  min/max lanes. Axis reductions use one cursor per output traversal. The Vector
  backend reduces contiguous non-last axes across independent output columns.
  Whole-tensor sums and each axis sum preserve their original addition order;
  no horizontal SIMD sum or FMA is used. Contiguous softmax rows vectorize their
  maximum and final division, while retaining scalar `Math.exp` and ordered sums.
- **OPT-17 — broadcasting:** both backends process innermost contiguous
  runs at arbitrary rank, with either input varying or broadcasting a singleton.
  The Vector backend uses SIMD within each run; Java uses a direct scalar loop.
  Subtraction/division preserve operand order. No expanded broadcast inputs are
  materialized. Unsupported strides use Java cursor traversal; overlapping
  inputs retain the existing snapshot policy.
- **OPT-18 — traversal:** internal stride cursors replace per-element index arrays
  in noncontiguous map, copy, toArray, sums, extrema, axis reductions, and batched
  matmul traversal. Generic binary fallback also uses cursors. Public generator
  callbacks still receive a fresh, independently retainable index array. Existing
  buffer/optimizer loops already reuse their index arrays; indexed sorting remains
  outside this change.
- **OPT-19 — batched matmul:** resolve source offsets once per batch and reuse the
  optimized rank-2 kernels, including tiled and freshly packed kernels, writing directly
  into the batch's output region. Supports batch broadcasting, vectors, offsets,
  and transposed/sliced inputs. Each dot product retains ascending inner-index
  accumulation with separate multiply/add. Packing remains local to each call.
- **OPT-20 — unary operations:** explicit `square`, `tanh`, and `log` methods join
  `negate` and `sigmoid`; square retains `pow(2)` semantics. Autograd log/tanh use
  the explicit methods. Arbitrary `map` callbacks are never recognized or rewritten.
  Strided inputs and the JAVA backend retain Java evaluation. Only sigmoid uses
  a new SIMD unary kernel, for contiguous inputs of at least 128 elements.
  Negate, square, tanh, and log retain Java evaluation after candidate measurements.

Extrema may compare values in a different order, but preserve Math.min/Math.max
NaN propagation and signed-zero selection. NaN payload bits are not guaranteed.
Empty whole-tensor min/max still throw; empty axis extrema retain their infinity
identities, sums return zero, and softmax returns empty storage. Nonfinite softmax
rows retain the prior NaN behavior.

## Measurement method

JDK 26.0.2 on the same host as the baseline; one benchmark thread, 256–512 MiB
heap, GC profiler, two independent forks, two 300 ms warmups and three 300 ms
measurements per fork. JAVA runs without resolving the Vector module. This is a
short comparative kernel sweep; raw files retain JMH uncertainty and all samples.
The final epoch comparison uses longer settings: two forks, three one-second
warmups and four one-second measurements. The baseline was rebuilt from `30b7905`
in an isolated checkout and rerun with these same settings (`confirmation-before-training-*.json`).
Small differences or overlapping intervals are not evidence of a speedup.

Each kernel uses 4,096 elements (32×128); broadcasts use 2×32×128 inputs, and
batched products multiply two 32×128 matrices by a broadcast 128×64 matrix.
The transposed variant stores the right operand as 1×64×128. Strided traversal
uses a transposed 32×128 view. Seeds and shape setup are outside timing.

Commands (substitute `before`, `candidate`, or `after` and backend):

```sh
./gradlew :benchmarks:kernel -PtensorBackend=vector --args='.*CoverageBenchmarks.run -f 2 -wi 2 -i 3 -w 300ms -r 300ms -rff benchmarks/results/opt11/after-vector.json'
./gradlew :benchmarks:training -PtensorBackend=vector --args='.*trainingEpoch -p features=128 -f 2 -wi 3 -i 4 -w 1s -r 1s -rff benchmarks/results/opt11/after-training-vector.json'
./gradlew check
```

The baseline benchmark invokes `map(Math::tanh)` and `map(Math::log)`; the new
benchmark invokes the explicit tensor methods. All other calls are unchanged.

## Rejected candidates and sigmoid accuracy

The first candidate sweep (`candidate-*.json`) included Vector negate, square,
sigmoid, tanh, and log. Vector tanh took **181.247 µs** and **229,524 B/op** versus
**20.332 µs** and roughly **32,912 B/op** before the change. It was removed.
Negate (0.871 → 0.888 µs), square (0.892 → 0.887 µs), and log
(14.585 → 13.980 µs) showed no convincing benefit, so their SIMD candidates were
also removed. The new named methods do not imply SIMD acceleration.

The retained sigmoid uses Vector EXP with separate add/divide, and Java Math.exp
for tails. Its finite results are tested against Java with an **8 ULP tolerance**;
NaNs, infinities, zero/saturation, subnormals, and exponent overflow boundaries
are checked separately. This is a tested tolerance, not a guarantee of bitwise
identity for transcendental functions on every CPU/JDK. Java evaluation is retained
for arbitrary callbacks, log, tanh, and softmax exp.

The separate `sigmoid-vector.json` crossover sweep uses inputs uniformly sampled
from [-5,5], so its exp distribution differs from the [0,1] main sweep. At 128
elements, dispatch takes **0.291 µs** versus **0.531 µs** for the Java reference;
at 4,096 elements it takes **8.994 µs** versus **16.068 µs**. At 16 elements the
retained Java path is equivalent (both about **0.078 µs**). These measurements
justify the conservative 128-element threshold; they do not establish an optimal
threshold for every machine.

## Epoch regression found and corrected

The initial short epoch sweep found a Java regression from 5.152 to 6.687 ms,
despite the microbenchmark gains (`initial-after-training-java.json`). Adding
contiguous Java broadcast runs alone did not resolve it: a longer run still took
6.529 ms (`java-broadcast-training.json`). Restoring the original rank-2 method,
which keeps output allocation and scalar arithmetic together, reduced the same
longer workload to approximately 4.73 ms (`java-rank2-training.json`). This suggests
a HotSpot optimization/alias-analysis sensitivity to passing the new output array
through a separate large dispatcher; compiler assembly was not inspected.

The final code preserves that existing rank-2 path and introduces batch output
offsets through the shared Vector and tiled kernels. The Java batched scalar path
also writes directly to each output region. This deliberately retains some scalar
dispatch duplication to avoid the measured regression. The final matched epoch
comparison below checks the retained implementation against a fresh baseline.

A longer intermediate Vector run also regressed (3.994 versus 3.614 ms).
The final dispatcher retains the existing 2D bias fast path, keeps general
broadcast traversal in a separate helper, and calls the offset-aware packed/tiled
kernels directly from rank-2 dispatch. The intermediate common-dispatch run varied
between forks (about 4.01 and 3.52 ms); raw samples are retained in
`vector-before-common-dispatch-training.json` and `vector-common-dispatch-training.json`.
These observations reinforce that kernel speedups alone do not establish epoch
speedups; the final matched epoch measurements below are the result to use.

## Final kernel measurements

Times are µs/op; allocation is B/op, rounded to whole bytes. Broadcast rows use
the final dispatcher confirmation files (`after-broadcast-*.json`); other rows
use `after-*.json`.

| Operation | JAVA before → after | VECTOR before → after | JAVA B/op before → after | VECTOR B/op before → after |
|---|---:|---:|---:|---:|
| sum | 1.994 → 2.047 | 2.099 → 2.083 | 136 → 136 | 136 → 136 |
| axisSum | 1.788 → 1.280 | 1.756 → 0.513 | 4,344 → 1,240 | 4,344 → 1,192 |
| min | 16.511 → 0.939 | 16.538 → 0.932 | 98,416 → 136 | 98,416 → 88 |
| axisMax | 38.773 → 1.220 | 39.096 → 0.532 | 197,857 → 1,240 | 197,857 → 1,192 |
| softmax | 16.752 → 16.553 | 16.774 → 14.808 | 33,752 → 33,032 | 33,752 → 33,032 |
| broadcastRows | 39.985 → 5.623 | 40.962 → 2.406 | 65,809 → 65,936 | 65,809 → 65,936 |
| broadcastColumns | 42.016 → 3.553 | 40.420 → 2.066 | 65,809 → 65,936 | 65,809 → 65,936 |
| stridedMap | 26.423 → 5.302 | 26.052 → 6.092 | 131,265 → 32,936 | 131,265 → 32,936 |
| stridedCopy | 24.632 → 5.460 | 24.411 → 5.794 | 131,265 → 32,936 | 131,265 → 32,936 |
| batchedMatmul | 1351.845 → 85.147 | 1538.325 → 69.261 | 262,527 → 33,194 | 262,531 → 33,194 |
| batchedTranspose | 1437.629 → 259.834 | 1435.629 → 75.596 | 262,529 → 33,237 | 262,529 → 164,298 |
| negate | 0.907 → 0.900 | 0.871 → 0.895 | 32,912 → 32,912 | 32,912 → 32,912 |
| square | 0.922 → 0.934 | 0.892 → 0.994 | 32,912 → 32,912 | 32,912 → 32,912 |
| sigmoid | 12.043 → 12.289 | 11.966 → 9.057 | 32,912 → 32,912 | 32,912 → 32,912 |
| tanh | 20.785 → 20.698 | 20.332 → 20.725 | 32,912 → 32,912 | 32,912 → 32,912 |
| log | 14.495 → 14.548 | 14.585 → 14.597 | 32,912 → 32,912 | 32,912 → 32,912 |

Batched transpose allocates fresh Vector packing scratch, but removes the much
larger generic indexing allocation. Vector versus Java is not a universal speed
claim: both benefit from direct traversal, and unsupported layouts keep Java
fallbacks. Whole-tensor ordered sums deliberately retain scalar addition.

## Complete workload check

The same 1,024-sample, 128-feature synthetic MSE/Adam training workload runs before
and after. It includes data loading, forward, backward, and updates. It does not
exercise every added kernel (notably batched matmul or sigmoid); kernel improvements
must not be presented as equivalent end-to-end improvements. Intervals below are
JMH's reported confidence intervals, not standard deviations.

| Backend | Before ms/epoch ± error | After ms/epoch ± error | Before → after B/epoch |
|---|---:|---:|---:|
| JAVA | 5.258 ± 0.389 | 4.789 ± 0.045 | 8,934,385 → 8,880,954 |
| VECTOR | 3.614 ± 0.068 | 3.655 ± 0.039 | 9,085,890 → 9,028,467 |

The matched longer runs show about 9% lower Java mean epoch time and no
statistically established Vector change (overlapping intervals). These are
workload-specific results, not an all-model training speedup.

## Verification

`./gradlew check` passes **106 tests per backend, 212 executions**. This includes
existing convergence, checkpoint, autograd and ownership checks, plus new ordered
reduction references, cancellation/overflow, NaN/infinity/signed-zero/empty cases,
arbitrary-rank traversal and retained generator indices, both broadcast operand
orders and overlapping views, batched tiling/broadcasting/offsets/vectors, fresh
mutable inputs, sigmoid ULP checks, and contiguous unary finite differences.
No sum reduction order changes were introduced. Java runs without the Vector
module, verifying that optimized dispatch retains the portable backend.
