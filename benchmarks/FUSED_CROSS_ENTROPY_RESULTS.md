# Fused cross entropy (OPT-08)

Measured 2026-09-17 on Apple M4 Pro (arm64), OpenJDK 26.0.2. The benchmark
compares the fused loss with an exact copy of the pre-OPT-08 smoothed graph in
the same revision (base `cf08676`). Raw results are in [results/opt08](results/opt08).

## Objective and implementation

`CrossEntropyLoss` delegates to a single `Variable.crossEntropy` node. Forward
uses max-shifted log-sum-exp directly on logits. The local adjoint is computed as
`(softmax * targetSum - target) / batch`, retained only when recording gradients,
and scaled by the upstream scalar during backward. Logical-order input snapshots
support strided and offset views; the retained adjoint is independent of subsequent
logit/target mutation. This removes the softmax/smoothing/log/multiply graph, but
still allocates input snapshots, an adjoint and the normal gradient accumulation
buffers. It does not implement reusable gradient storage (OPT-10).

This is deliberately a numerical/API change, not an exact rewrite of the previous
`log((1 - 2e-12) * softmax + 1e-12)` objective. A confidently incorrect unit target
can now have loss above 27.6 and a nonvanishing gradient. Targets must match the
logit shape and be finite and nonnegative. They are not silently normalized;
unnormalized weights use their actual row mass, including zero. Nonfinite logits,
empty tensors and scalars are rejected. Last-axis classes and the old division by
the first dimension (one for vectors) are preserved, including higher ranks.
Double precision still limits representable losses at the most extreme ranges.

The checkpoint schema and `CrossEntropy` identifier are unchanged. Existing
checkpoints restore their weights and optimizer state, then use the new objective.
Their resumed trajectories need not match training with the old smoothed loss.
Save/load/resume consistency under the new implementation is tested with both SGD
and Adam; there is no legacy-objective option in the production API.

## Method

JMH 1.37, two forks, three one-second warmups and five one-second measurements,
one thread, 256–512 MiB heap, GC profiler. Backends run sequentially. Setup resets
the model each iteration. Each epoch includes shuffle/loading, dense forward,
loss, backward and Adam updates. Synthetic teacher argmax labels, 1,024 samples,
batch 32, 128 or 784 features, features → 64 → 10 network, seed 42. Loss-only
measurements use fixed 32 × 10 logits and one-hot targets, with a fresh leaf and
forward/backward each operation. Their feature parameter is unused and provides
a repeated measurement, not a different loss shape.

Both variants use identical data, seeds and optimizer settings. Because objectives
differ, the comparison is not a claim of identical trajectories at saturation.
These classification workloads are separate from `TrainingBenchmarks` and its
unchanged MSE targets/baseline; the figures below are not MSE speedups.

```sh
./gradlew check
./gradlew :benchmarks:classification --args='-f 2 -wi 3 -i 5 -w 1s -r 1s'
./gradlew :benchmarks:classification -PtensorBackend=java --args='-f 2 -wi 3 -i 5 -w 1s -r 1s'
```

The default JSON files are `benchmarks/build/results/classification-{vector,java}.json`;
copy them into `benchmarks/results/opt08` to preserve a run. Reported errors are
JMH's 99.9% confidence interval half-widths. Allocation is bytes per operation,
not peak live memory.

## Validation

`./gradlew check` passes: 84 tests per backend, 168 executions total.

The cross-entropy tests cover one-hot, soft and unnormalized targets; zero mass;
moderate-logit agreement with the legacy objective; extreme logits and large common
offsets; finite differences on transposed, offset and rank-3 inputs; first-axis
reduction; invalid inputs; upstream seeds, shared nodes, repeated backward and
mutation snapshots; no-grad; classification convergence; and checkpoint round trips.
The existing resume suite checks Adam and momentum SGD. The classification workload
test also checks actual updates and legacy/fused parity across a partial final batch.

## Measurements

| Backend | Operation | Features | Legacy ms/op | Fused ms/op | Legacy B/op | Fused B/op |
|---|---|---:|---:|---:|---:|---:|
| VECTOR | classificationEpoch | 128 | 4.242660 ± 0.036547 | 4.090475 ± 0.045171 | 14,796,939 | 13,652,317 |
| VECTOR | classificationEpoch | 784 | 16.704386 ± 0.056260 | 16.790075 ± 0.140954 | 41,797,500 | 40,654,190 |
| VECTOR | lossForwardBackward | 128 | 0.008439 ± 0.000112 | 0.003622 ± 0.000014 | 52,232 | 16,810 |
| VECTOR | lossForwardBackward | 784 | 0.008376 ± 0.000103 | 0.003638 ± 0.000065 | 52,483 | 16,921 |
| JAVA | classificationEpoch | 128 | 5.760954 ± 0.019284 | 5.711453 ± 0.044748 | 14,642,006 | 13,500,411 |
| JAVA | classificationEpoch | 784 | 21.031076 ± 0.483436 | 20.865076 ± 0.188928 | 41,679,322 | 40,532,641 |
| JAVA | lossForwardBackward | 128 | 0.008465 ± 0.000088 | 0.003621 ± 0.000094 | 52,200 | 16,831 |
| JAVA | lossForwardBackward | 784 | 0.008396 ± 0.000100 | 0.003598 ± 0.000054 | 52,264 | 16,851 |

Loss forward/backward is about 2.3× faster on both backends, with about 68% lower
allocation (roughly 52 KB → 17 KB per operation). Complete epochs save about
1.14 MB, approximately 7.7–7.8% at 128 features and 2.7% at 784 features.
The Vector 128-feature epoch improves by 3.6%. Its confidence intervals are
separated; those for Vector 784 and both Java epoch comparisons overlap, so these
runs establish no clear full-epoch timing improvement for those cases. The much
larger loss-only improvement should not be generalized to whole-model throughput.
