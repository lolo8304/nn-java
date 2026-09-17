# Tensor and NN optimization roadmap

Status: proposed work, based on the current Java 26 implementation. Vector is the
default backend, with an explicit Java override. Priorities are engineering estimates,
not measured speedup guarantees. Preserve both backends and compare complete training
and inference workloads before accepting a performance change.

## First: reliable measurements

- [x] **Add JMH benchmarks with multiple forks and sufficient warmup.** Measure small,
  medium, and MNIST-sized tensors, vector tails, contiguous/strided inputs, both
  matmul transposes, allocation rate, and GC time. The current 100-operation warmup
  produces variable dense Vector allocations.
- [x] **Add repeatable end-to-end benchmarks.** Measure samples/second, epoch time,
  inference latency, and peak memory on fixed datasets/seeds. Separate data loading,
  augmentation, forward, backward, optimizer, and checkpoint time. Profile before
  choosing the next kernel to optimize. Do not use test accuracy to tune performance.

Implemented measurement foundation: `benchmarks` now supplies forked JMH kernel and
complete synthetic-epoch workloads, allocation/GC profiling, repeated phase profiles,
and optional JFR recording. Real MNIST augmentation/checkpoint profiles and exact
peak-live-memory measurement remain follow-up work; see [the guide](benchmarks/GUIDE.md).

## Tensor priorities

| Priority | Optimization | Current opportunity | Validation / tradeoff |
|---|---|---|---|
| High | Vector matmul with a transposed right operand | `g.matmul(weights.transpose())` uses the scalar fallback in input-gradient computation. Evaluate packed panels or a dedicated transpose-aware kernel. | Benchmark packing overhead and small shapes; verify offset/stride handling. Avoid stale packed weights after optimizer updates. |
| High | Cache-blocked matmul | The current kernel repeatedly streams right-hand rows and output data. Tile larger products to improve cache reuse. | Tune against representative shapes and CPU caches; retain a small-matrix path. Preserve reduction order unless explicitly accepting rounding changes. |
| Foundation implemented | Destination-buffer operations | Public arithmetic/scalar `*Into`, `copyInto`, and `reluInto` now reuse destinations. Fused `axpy`, optimizer integration, and gradient reuse remain future work. | Snapshot semantics protect overlapping views; exact-layout updates and independent destinations avoid full-sized scratch buffers. |
| Medium | Faster reductions | `sum`, axis reductions, min/max, and softmax row reductions still use Java loops or indexed iteration. Add specialized contiguous kernels and evaluate SIMD. | Vector sums change addition order. Test cancellation, extreme magnitudes, NaNs, infinities, signed zero, and empty tensors; use documented tolerances. |
| Medium | Extend broadcast SIMD coverage | The explicit Vector broadcast path handles a contiguous 2D matrix and 1D right operand. Add common singleton-axis and higher-rank cases. | Avoid copying whole broadcast operands; compare setup overhead on small tensors. Preserve operand order for subtraction/division. |
| Medium | Faster noncontiguous traversal | Generic map/copy/extrema and some fallbacks still allocate index arrays. Use internal stride cursors and direct offsets. | Keep public generator callback semantics unchanged; validate arbitrary rank, singleton axes, offsets, and empty shapes. |
| Medium | Batched matmul kernel reuse | Batched matmul still computes each output through the generic indexing path. Resolve batch base addresses once and reuse optimized 2D kernels. | Test broadcasted batch dimensions, vector operands, and strided views. |
| Medium | Specialized unary kernels | Negation, square, sigmoid, tanh, log, and general callbacks retain Java paths. Add dedicated kernels where profiles justify them. | Do not try to vectorize arbitrary callbacks. Check special-value behavior and numerical error for transcendental functions. |
| Later | Parallel large matmul | Independent output tiles can run on a bounded worker pool. | Add size thresholds; avoid scheduling overhead, nested parallelism, and contention with data loading. Measure single-request latency and throughput separately. |
| Later | Float32 tensors | An optional float representation halves raw element storage versus double and may improve memory bandwidth utilization. | Significant API, kernel, optimizer, and persistence work; validate convergence and define accumulator precision. |
| Later | Native BLAS backend | Evaluate a native backend for large dense products while retaining Java/Vector portability. | Include native-call, layout-conversion, copying, packaging, and memory-lifetime costs in the benchmark. |

## Neural-network priorities

| Priority | Optimization | Current opportunity | Validation / tradeoff |
|---|---|---|---|
| High | Fuse Adam and SGD updates | Optimizers build chains of temporary tensors; `Parameter.applyDelta` then iterates with index arrays. Update owned moments, velocities, and parameter storage directly. | Requires safe destination-buffer kernels. Compare multiple steps and save/load/resume behavior; handle absent and empty gradients. |
| High | Inference without autograd graphs | `predict` calls the ordinary forward path; trainable parameters cause graph nodes and backward closures to be retained. Add an exception-safe inference/no-grad context or a dedicated inference path. | Do not interpret `training=false` as globally disabling gradients: dropout mode and gradient recording are separate concepts. Test output parity and that inference leaves parameter gradients unchanged. |
| High | Fuse stable cross entropy from logits | Current loss constructs softmax, smoothing, log, multiply, and sum nodes. A log-sum-exp loss with a direct backward kernel reduces intermediates and handles extreme logits better. | This changes the current epsilon-smoothed objective. Introduce it explicitly or document migration; verify normalization, target semantics, extreme logits, finite differences, and model persistence. |
| High | Bulk batch assembly | `DataLoader` uses `Tensor.generate` plus `Arrays.copyOfRange` for every input/target element. Add checked stack/gather/copy operations to copy each sample in bulk. | Preserve sample order, fetch-once behavior, shape validation, and final partial batches. Keep strided sample support. |
| Medium | Reuse gradient storage | `addGrad` copies the first contribution and allocates another tensor for each accumulation; `zeroGrad` discards storage. Reuse owned buffers where safe. | Preserve shared-graph accumulation, repeated backward calls, leaf-gradient semantics, and non-aliasing with seeds/inputs. |
| Medium | Fuse activation backward kernels | ReLU/LeakyReLU generate masks; sigmoid/tanh create several temporary tensors. Compute each derivative directly into its gradient destination. | Check values around zero, saturation, strided views, and finite-difference gradients. Avoid overwriting activations still needed elsewhere in the graph. |
| Medium | Reduce graph bookkeeping | Each node stores parent collections and closures; each backward traversal creates a topology list and identity set using recursive traversal. Evaluate leaner node representations and iterative traversal. | Profile first. Deep graphs, diamonds, shared nodes, and repeated backward must remain correct; do not cache a topology across changing graphs. |
| Medium | Profile and optimize image augmentation | Determine whether resampling dominates epoch time before optimizing loops or adding bounded prefetch. | Keep training-only augmentation and label integrity. Parallel RNG use must have a defined reproducibility policy and must not leak validation/test data. |
| Medium | Direct argmax/metric kernels | Prediction and metrics use slices and indexed access; `predictClasses` repeatedly clones row shapes. Add a row-wise argmax primitive and reuse it. | Preserve tie behavior, shape validation, and the existing classification rules. |
| Later | Fuse dense + bias + activation | Dense output, bias addition, and activation currently require separate passes and intermediates. Evaluate a fused inference kernel, then training support. | Training must retain the values required by backward; benchmark full layers and maintain separate unfused reference paths. |
| Later | Training workspaces | Reuse batch, activation, and temporary buffers after ownership and lifetime rules exist. | Build on destination-buffer operations; never overwrite values retained by a live autograd graph or user-visible tensors. |
| Later | Cheaper checkpoint serialization | Optimizer serialization creates default zero tensors even for existing state through eager `getOrDefault` arguments, and converts tensors to arrays. Avoid unnecessary defaults and copies. | Keep existing checkpoint compatibility and deterministic optimizer resume tests. Profile checkpoint frequency before prioritizing. |

## Suggested implementation sequence

1. Establish JMH and end-to-end baselines.
2. Add safe internal destination-buffer kernels, then fuse optimizer updates.
3. Add graph-free inference and bulk batch assembly as independent improvements.
4. Optimize transpose-aware and cache-blocked matmul where the profile supports it.
5. Introduce a separately validated stable fused loss and activation backward kernels.
6. Evaluate reductions, wider SIMD coverage, parallelism, and float/native backends
   using the updated profiles.

For each step, retain a reference implementation, run both backend suites, check
relevant gradients, and report runtime plus allocation changes. Numerical changes
need explicit error tolerances and convergence checks. Mixed vector/matrix autograd
support is also unfinished, but it is a correctness/coverage feature rather than
a performance optimization.
