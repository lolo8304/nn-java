# Tensor and NN optimization tracker

Last updated: 2026-09-17. Baseline: commit `4937fc0`, Java 26.0.2, Vector backend
by default, with an explicit Java override. This file is the working backlog for
future optimization tasks. Use the IDs below when selecting work.

Status values: **Done**, **Next**, **Planned**, **In progress**, **Blocked**,
**Deferred**. Only mark an item Done after implementation, relevant tests on both
backends, and recorded measurements. Nothing is currently In progress.

## Evidence behind the order

Source: [measured results](benchmarks/OUTPUT_BUFFER_RESULTS.md) and
[benchmark/profiling guide](benchmarks/GUIDE.md).

- Reusable Vector elementwise operations: **3.697 → 1.306 µs/op**, with allocation
  reduced from **98,832 → 48 B/op**. Buffer primitives and optimizer integration are implemented; gradient integration
  remains outstanding.
- Complete synthetic training: **13.390 ms/epoch** with Vector and **15.082 ms/epoch**
  with Java, approximately **52.6 MB allocated per epoch**.
- In the representative instrumented Vector run, **Adam/parameter updates consume
  55.3% of epoch time and 32.1 MB/epoch**, about 61% of allocations.
- **Loading/shuffling consumes 14.3% of time and 8.2 MB/epoch**. Backward consumes
  19.3% and 10.3 MB; forward/loss consumes 10.8% and 2.0 MB.

These results use 1,024 synthetic samples, 128 features, batch size 32, a
128 → 64 → 10 network, MSE and Adam. They exclude augmentation, validation,
checkpoint IO and inference-only workloads. Phase timing includes instrumentation
and GC effects. Other proposed benefits below are hypotheses, not measured speedups.

The figures above are the pre-OPT-03 baseline. Fused optimizer updates reduce
full-epoch allocation to about 20.5 MB; see the [optimizer measurements](benchmarks/FUSED_OPTIMIZER_RESULTS.md).
OPT-04 then reduces it to about 13.9 MB; see the [bulk batch measurements](benchmarks/BULK_BATCH_RESULTS.md).

## Ordered implementation list

| Order / ID | Status | Area | Optimization | Description and reason | Completion criteria |
|---|---|---|---|---|---|
| 1 · OPT-01 | Done | Benchmarks | Establish reliable measurements | Added forked JMH kernels and complete synthetic epochs, allocation/GC reports, repeated training phase profiles, and optional JFR. This establishes the current baseline. | Implemented in `4937fc0`; guide and measured results linked above. Broader real-workload profiling remains OPT-15. |
| 2 · OPT-02 | Done | Tensor | Reusable output buffers | Added tensor/scalar arithmetic `*Into`, `copyInto`, and `reluInto`. Supports exact in-place updates and snapshots differently mapped overlapping views. | Implemented in `4937fc0`; overlap/stride/shape tests and measured allocation reduction. Fused optimizer operations remain OPT-03. |
| 3 · OPT-03 | Done | NN + Tensor | Fuse Adam and SGD updates | Update owned moments, velocities and parameters directly instead of constructing temporary tensor chains and indexed deltas. Add fused primitives such as `axpy` if needed. This targets the largest measured time/allocation source. | Implemented in this commit; reference steps, absent/empty gradients, resume and alias tests pass on both backends. [Measurements and raw results](benchmarks/FUSED_OPTIMIZER_RESULTS.md). |
| 4 · OPT-04 | Done | NN data | Bulk batch assembly | Replace per-element generator/index-array copying with checked stack/gather/copy operations into batches. Targets the measured 8.2 MB/epoch loader allocation. | Implemented with checked row copies in [DataLoader](nn/src/main/java/ch/lolo/nn/data/DataLoader.java). Shuffle/fetch-once, shape, stride and partial-batch tests pass on both backends. [Loading and epoch measurements](benchmarks/BULK_BATCH_RESULTS.md). |
| 5 · OPT-05 | Done | NN inference | Skip autograd graphs during inference | Added exception-safe, thread-local no-grad scopes and early recording checks to avoid parent lists, backward closures and retained graph intermediates during prediction/evaluation. | Implemented in [GradMode](nn/src/main/java/ch/lolo/nn/autograd/GradMode.java) and prediction/evaluation. Parity, gradient, scope and dropout tests pass on both backends. Training mode remains independent. [Inference measurements](benchmarks/NO_GRAD_RESULTS.md). |
| 6 · OPT-06 | Done | Tensor | Vector matmul with transposed right operand | Use fresh 64-column panels for larger transposed/strided-right products, preserving summation order. Small products retain the scalar path; no packed weights are cached. | Implemented in [VectorKernels](tensor/src/main/java/ch/lolo/tensor/VectorKernels.java). Offset/stride/tail, exact arithmetic, updated-weight and input-gradient tests pass on both backends. [Packing, kernel and epoch measurements](benchmarks/TRANSPOSED_MATMUL_RESULTS.md). |
| 7 · OPT-07 | Done | Tensor | Cache-blocked matrix multiplication | Tile larger rank-2 products on both backends; retain small-product kernels. Bound fresh strided-right scratch to 32 KiB. | Implemented in [MatmulKernels](tensor/src/main/java/ch/lolo/tensor/MatmulKernels.java). Ordered arithmetic, tile boundaries, both transpose directions, views, mutations and large gradients verified on both backends. [Kernel and epoch measurements](benchmarks/BLOCKED_MATMUL_RESULTS.md). |
| 8 · OPT-08 | Done | NN loss | Stable fused cross entropy | Use log-sum-exp directly from logits and a dedicated backward calculation to avoid the softmax/smoothing/log/multiply graph. | Implemented in [Variable.crossEntropy](nn/src/main/java/ch/lolo/nn/autograd/Variable.java). Unsmoothed objective and checkpoint migration documented; weighted targets, extreme logits, finite differences, views, graph lifetime, convergence and persistence tested on both backends. [Separate classification measurements](benchmarks/FUSED_CROSS_ENTROPY_RESULTS.md). |
| 9 · OPT-09 | Planned | NN gradients | Fuse activation backward kernels | Compute ReLU/LeakyReLU, sigmoid and tanh derivatives directly into gradient destinations instead of allocating masks and intermediate tensors. | Finite differences, zero/saturation behavior, strided views and allocation measurements. Retain activations still needed by other graph branches. |
| 10 · OPT-10 | Planned | NN gradients | Reuse gradient storage | Accumulate into owned gradient buffers instead of copying the first contribution and reallocating on subsequent contributions. Reuse storage across resets where safe. | Shared graphs, repeated backward, leaf accumulation, seed non-aliasing and zero-gradient semantics remain correct. Measure backward allocation and full epochs. |
| 11 · OPT-11 | Planned | Tensor | Extend optimized kernel coverage | Add specialized reductions, common broadcast layouts, batched matmul dispatch and unary operations where profiling justifies them. Split work into the subitems below. | Benchmark each addition on both backends; retain Java fallbacks. Test floating-point edge cases and document changes in reduction order. |
| 12 · OPT-12 | Planned | NN data | Optimize augmentation and prefetch | Profile real image transformations, improve hot resampling loops, and add bounded prefetch only if preparation limits training throughput. | Requires real-data profiling in OPT-15. Preserve training-only transforms, labels, deterministic RNG policy and validation/test isolation. |
| 13 · OPT-13 | Deferred | Tensor | Parallel large matrix multiplication | Distribute independent output tiles across a bounded CPU worker pool once serial kernels are tuned. | Size thresholds, no nested oversubscription, controlled interaction with data loading, and latency/throughput measurements on larger models. |
| 14 · OPT-14 | Deferred | Tensor architecture | Float32 and native BLAS evaluation | Evaluate these as separate backend/data-type projects after simpler optimizations. Float32 reduces element storage; native BLAS may help sufficiently large products. | Measure conversion/copy/native-call overhead; define precision, convergence, packaging, memory lifetime and checkpoint compatibility. Preserve portable Java/Vector paths. |

**Start next with OPT-09.** OPT-08 is complete; see [fused cross-entropy results](benchmarks/FUSED_CROSS_ENTROPY_RESULTS.md).
Loss forward/backward is about 2.3× faster with 68% less allocation on both backends.
Classification epochs save about 1.14 MB; the 128-feature Vector epoch improves by
3.6%, while other epoch timing intervals overlap. The MSE baseline is unchanged.
All 168 test executions pass. Cross entropy now uses the standard unsmoothed objective;
existing checkpoints remain loadable but resumed training adopts this new objective.

OPT-07 is complete; see [cache-blocking results](benchmarks/BLOCKED_MATMUL_RESULTS.md).
The 784-feature epoch improves from 18.785 to 16.976 ms with Vector and from
22.367 to 21.084 ms with Java; the 128-feature workload is effectively unchanged.
Larger transposed-right products improve on both backends, while large contiguous
products do not uniformly benefit. All 152 test executions pass.

OPT-06 is complete; see [transposed matmul results](benchmarks/TRANSPOSED_MATMUL_RESULTS.md).
Eligible measured Vector products are 2.5–4.0× faster with packing included.
At 128 features, full epochs improve from 4.204 to 3.931 ms, with allocation rising
from 13.861 to 14.016 MB due to fresh panels. The 784-feature full-epoch timing
intervals overlap; no clear epoch-level gain is established for that shape.

OPT-05 is complete; see [inference results](benchmarks/NO_GRAD_RESULTS.md).
The Vector inference sweep saves roughly 730 bytes per prediction batch (about 22%
at batch size 1 and 1.3% at batch size 32). Timing uncertainty prevents a general
latency speedup claim.

OPT-04 is complete; see [bulk batch results](benchmarks/BULK_BATCH_RESULTS.md).
For the 128-feature workload, loader allocation fell from 8.228 to 1.636 MB/epoch;
Vector full epochs improved from 6.006 to 4.288 ms and Java from 7.250 to 5.630 ms.
After OPT-04, full-epoch allocation was about 13.9 MB. Its instrumented Vector profile
puts loading at 5–8% of epoch time and backward at 52–60%, with about 10.3 MB
allocated in backward after warmup. OPT-09/10 remain promising training follow-ups;
OPT-08 onward is provisional until broader real-data profiles support the order.

## Detailed follow-up backlog

These items preserve narrower opportunities from the original analysis. Their
statuses are independent of their parent items; completing one does not complete
the whole parent optimization.

| ID | Status | Related item | Optimization | Description / acceptance criteria |
|---|---|---|---|---|
| OPT-15 | Planned | OPT-01, OPT-12 | Broader workload profiling | Add real MNIST augmentation/validation/checkpoint breakdowns, standalone inference benchmarks and an explicit peak-memory methodology. Current allocation counters do not measure peak live memory. |
| OPT-16 | Planned | OPT-11 | Reduction kernels | Specialize sum, axis reductions, min/max and softmax row reductions. SIMD sums may reorder additions; test cancellation, large magnitudes, NaNs, infinities, signed zero and empty shapes. |
| OPT-17 | Planned | OPT-11 | Broadcast SIMD coverage | Extend beyond same-shape/scalar operations and 2D matrix + 1D right operand to singleton axes and higher ranks. Avoid materializing full broadcast operands; preserve subtraction/division operand order. |
| OPT-18 | Planned | OPT-11 | Noncontiguous traversal | Replace remaining per-element index arrays in generic map/copy/extrema and fallback paths with stride cursors. Preserve public generator callback semantics and test arbitrary ranks and empty shapes. |
| OPT-19 | Planned | OPT-11 | Batched matmul reuse | Resolve each batch's source offsets once and reuse optimized 2D kernels rather than the generic per-output path. Cover batch broadcasting, vectors and strided views. |
| OPT-20 | Planned | OPT-11 | Unary kernels | Specialize negate, square, sigmoid, tanh and log where measured. Do not attempt to infer arbitrary callback behavior; validate approximation error and special values. |
| OPT-21 | Planned | OPT-10 | Autograd bookkeeping | Profile parent collections, closures, topology allocation and recursive traversal. Evaluate iterative traversal and leaner node storage; preserve deep/shared graphs and repeated backward. |
| OPT-22 | Planned | OPT-05 | Argmax and metric kernels | Replace repeated slicing/indexing/shape cloning with row-wise argmax operations. Preserve ties, shape checks and classification semantics. |
| OPT-23 | Deferred | OPT-05, OPT-07 | Dense + bias + activation fusion | Start with inference to reduce passes and intermediates. Training support must retain the values needed by backward and preserve an unfused reference path. |
| OPT-24 | Deferred | OPT-02, OPT-10 | Training workspaces | Reuse batch, activation and temporary storage only after ownership/lifetime rules are established. Do not overwrite live graph values or user-visible tensors. |
| OPT-25 | Planned | OPT-15 | Checkpoint allocation | Avoid eager default zero tensors and unnecessary array copies during serialization. Preserve the existing format and optimizer resume behavior; prioritize using actual checkpoint cost. |

Mixed vector/matrix autograd support is also unfinished, but is a correctness/API
coverage task rather than a performance optimization; track it separately.

## How to maintain this tracker

1. Select an ID and change its status to In progress. Record a blocker if work stops.
2. Keep a reference path and run relevant checks on both JAVA and VECTOR backends.
3. For numerical changes, include gradient checks, explicit tolerances and convergence
   checks; for mutable buffers, include overlapping-view and graph-lifetime cases.
4. Compare the same shapes, seeds, JDK, heap and benchmark settings before and after.
   Report runtime and allocations; validate kernel gains against complete workloads.
5. Link the implementation commit and result document in the row before marking Done.
   Update the evidence and next-work recommendation when the profile changes.
