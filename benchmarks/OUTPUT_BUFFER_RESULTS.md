# Reusable output buffer measurements

Measured on 2026-09-17 on Apple M4 Pro, OpenJDK 26.0.2 (arm64).
Both backends use the same updated code, a 256–512 MiB heap, one thread, two
independent JVM forks, three one-second warmup iterations and five one-second
measurement iterations per fork. Each result has ten measurement samples.
Kernels use batch=32 and features=128. Timing errors below are reported by JMH.

```bash
./gradlew :benchmarks:kernel --args='.*KernelBenchmarks.(elementwise|elementwiseInto|copy|copyInto)$ -p features=128'
./gradlew :benchmarks:kernel -PtensorBackend=java --args='.*KernelBenchmarks.(elementwise|elementwiseInto|copy|copyInto)$ -p features=128'
```

| Backend | Workload | µs/op ± JMH error | B/op |
|---|---|---:|---:|
| JAVA | copy | 1.005 ± 0.013 | 32,912.007 |
| JAVA | copyInto | 0.646 ± 0.037 | 0.004 |
| JAVA | elementwise | 24.337 ± 3.458 | 98,856.171 |
| JAVA | elementwiseInto | 21.842 ± 3.584 | 72.152 |
| VECTOR | copy | 1.008 ± 0.018 | 32,912.007 |
| VECTOR | copyInto | 0.605 ± 0.027 | 0.004 |
| VECTOR | elementwise | 3.697 ± 0.066 | 98,832.026 |
| VECTOR | elementwiseInto | 1.306 ± 0.172 | 48.009 |

The reusable elementwise workload executes multiply → broadcast bias add → ReLU
using one preallocated destination. The allocating version creates three outputs.
Vector buffer reuse reduced time by about 2.8× and allocation from about 98.8 KB
to 48 B/op in this workload. Java buffer reuse primarily reduced allocations.
The tiny nonzero `copyInto` allocation reported by the profiler is effectively zero.

The `*Into` paths are not universally allocation-free: small shape/index objects
can remain, and differently mapped overlapping storage requires a snapshot.
These measurements use separate input/output storage and exact-layout in-place
updates, not the overlapping-view fallback. Compare workload-specific results,
not old 100-operation smoke timings. Training integration of these buffers into
optimizers is still future work.

Raw JMH JSON is written to `benchmarks/build/results/kernel-{java,vector}.json`.
It includes per-fork samples and uncertainty and is ignored by Git; save it before
rerunning the same benchmark kind/backend. See [the guide](GUIDE.md) for profiling.

## Complete synthetic training baseline

Same JVM/fork/warmup/measurement configuration as above. Workload: 1,024 samples,
128 features, batch 32, Dense(128,64) → ReLU → Dense(64,10), MSE, Adam.
The timed operation includes shuffling/loading and all training computations.
It excludes construction, augmentation, validation, and checkpoint IO.
Buffers are not yet integrated into the optimizer; this establishes the baseline
for the next optimization, rather than demonstrating an end-to-end buffer speedup.

| Backend | ms/epoch ± JMH error | Approx. samples/s | B/epoch |
|---|---:|---:|---:|
| JAVA | 15.082 ± 0.182 | 67,897 | 52,601,662 |
| VECTOR | 13.390 ± 0.279 | 76,474 | 52,597,625 |

Raw JMH results: `benchmarks/build/results/training-{java,vector}.json`.
The GC profiler may also observe iteration-setup allocation; it is an approximation
of allocation per timed epoch.

## Training phase profile

Three same-JVM runs per backend, five warmup and five measured epochs per run,
with the same model/data seed and dimensions above. Final loss was 0.01980394 in
all six runs. Vector epoch time ranged from 14.062 to 14.752 ms (median 14.478);
Java ranged from 15.584 to 17.651 ms (median 15.609).

Vector run 2, whose total epoch time was the median:

| Phase | ms/epoch | Share of total time | Bytes/epoch |
|---|---:|---:|---:|
| Load/shuffle | 2.073 | 14.3% | 8,226,136 |
| Zero gradients | 0.004 | 0.0% | 1,024 |
| Forward + loss | 1.561 | 10.8% | 2,010,112 |
| Backward | 2.790 | 19.3% | 10,254,336 |
| Optimizer | 8.011 | 55.3% | 32,110,592 |

Total allocation was about 52.6 MB/epoch. The optimizer accounts for about 61%
of allocation, making fused Adam/parameter updates the first target supported by
this profile. Batch assembly is another significant allocation source. These are
findings for this synthetic workload, not claims for every network shape or MNIST
augmentation pipeline. Instrumentation, loop overhead and GC pauses affect timing;
phase percentages need not sum to 100%.

The JFR option was smoke-tested separately with a short workload; its recording
contains allocation, execution and GC events. It was not used for the clean timing
tables above. Use a longer recording when investigating hot call stacks.
