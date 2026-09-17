# Local kernel comparison

See [reusable output buffer results](OUTPUT_BUFFER_RESULTS.md) for the latest
forked JMH measurements.

For the new forked JMH benchmarks and training phase profiler, see the
[benchmark guide](GUIDE.md). The tables below are historical smoke-harness results
from before reusable output buffers; they are not JMH measurements.

## Current Java 26 backend measurements

Recorded on 2026-09-17. Local environment: Apple M4 Pro (arm64), OpenJDK 26.0.2,
256–512 MiB heap. The harness performs 100 warmup and 100 measured operations per
workload, with a volatile result sink. Inputs are 32×128; dense weights are 128×64.
Softmax and dense workloads include forward and backward execution; dense also
includes squared mean loss with constant inputs. Allocation counts are per operation.

VECTOR is now the default. Reproduce either backend with:

```bash
./gradlew :examples:runKernelBenchmark -PtensorBackend=java
./gradlew :examples:runKernelBenchmark                         # VECTOR
```

### Latest user-reported comparison

These are the two outputs supplied by the user, kept together as one comparison.

| Workload | JAVA ms/op | VECTOR ms/op | JAVA bytes/op | VECTOR bytes/op | JAVA / VECTOR time |
|---|---:|---:|---:|---:|---:|
| Elementwise | 0.090 | 0.006 | 98,808 | 98,784 | 15.00× |
| Copy | 0.003 | 0.002 | 32,912 | 32,912 | 1.50× |
| Axis sum | 0.007 | 0.010 | 4,344 | 4,344 | 0.70× |
| Softmax forward/backward | 0.074 | 0.073 | 134,992 | 134,992 | 1.01× |
| Dense forward/backward | 0.139 | 0.089 | 282,296 | 282,376 | 1.56× |

In this run, dense forward/backward took about 36% less time with VECTOR.
Ratios are calculated from the rounded printed timings, not unrounded samples.

### Latest local verification after making VECTOR the default

A subsequent verification run produced the following pair. These numbers are
reported separately rather than selecting the fastest measurement from each run.

| Workload | JAVA ms/op | VECTOR ms/op | JAVA bytes/op | VECTOR bytes/op |
|---|---:|---:|---:|---:|
| Elementwise | 0.068 | 0.005 | 98,808 | 98,784 |
| Copy | 0.003 | 0.002 | 32,912 | 32,912 |
| Axis sum | 0.010 | 0.009 | 4,344 | 4,344 |
| Softmax forward/backward | 0.073 | 0.076 | 134,992 | 134,992 |
| Dense forward/backward | 0.142 | 0.131 | 282,296 | 346,299 |

Both are short exploratory runs, not statistically controlled benchmarks. The
100-operation warmup may leave JIT compilation incomplete; the local dense Vector
run also allocated more bytes. Copies and axis sums use the same implementation
in both backends, so their timing differences are run variation, not explicit SIMD
acceleration. Softmax itself retains Java kernels. These results do not measure
end-to-end MNIST training. Use longer warmup and multiple JMH forks for reliable
performance comparisons.

## Historical Java 25 optimization comparison

Measured on Apple M4 Pro (arm64), Homebrew OpenJDK 25.0.2, with a 256–512 MiB heap.
Baseline: commit `1a88c1f`, compiled directly with the same Java 25 compiler.
Updated: optimized Java 25 code before the Java 26 upgrade. Each workload has 100 warmup and 100 measured operations,
with a volatile result sink. Allocations use HotSpot's current-thread allocation counter.
Inputs are 32×128, dense weights are 128×64. Softmax timing includes forward and backward;
dense timing includes forward, squared mean loss and backward with constant inputs.

One process per version, measured sequentially on the same machine:

Baseline:

```text
elementwise            0.475 ms/op       787072 bytes/op
copy                   0.051 ms/op       131264 bytes/op
axis sum               0.143 ms/op       197848 bytes/op
softmax backward       4.508 ms/op     13306976 bytes/op
dense backward         0.837 ms/op      1003888 bytes/op
```

Updated:

```text
elementwise            0.161 ms/op        98808 bytes/op
copy                   0.007 ms/op        32912 bytes/op
axis sum               0.021 ms/op         4344 bytes/op
softmax backward       0.320 ms/op       134992 bytes/op
dense backward         0.359 ms/op       282280 bytes/op
```

These are exploratory measurements, with no confidence intervals. Short warmup, JIT
compilation, GC and system load affect timing; the allocation reduction and linear
softmax derivative are the stronger evidence. This is not a full MNIST training benchmark.
Use JMH with multiple forks and representative shapes for release-level comparisons.
Softmax gradients use an algebraically equivalent expression with different floating-point
rounding, so bit-for-bit equality with previous training runs is not promised.

These Java 25 measurements are historical; the current build targets Java 26.
To compare another revision, compile its `tensor/src/main/java` and `nn/src/main/java`
sources into a separate class directory with `javac`, compile this same harness against
that directory, and run it with the same JVM flags. Do not mix baseline and updated classes.

## Java 26 upgrade run

OpenJDK 26.0.2 (arm64), same machine, inputs, warmup/measurement counts, and
256–512 MiB heap. Reproduce the Java backend with `./gradlew :examples:runKernelBenchmark -PtensorBackend=java`; Gradle
selects the Java 26 toolchain for compilation and execution.

```text
elementwise            0.078 ms/op        98808 bytes/op
copy                   0.002 ms/op        32912 bytes/op
axis sum               0.007 ms/op         4344 bytes/op
softmax backward       0.078 ms/op       134992 bytes/op
dense backward         0.143 ms/op       282280 bytes/op
```

This is a single exploratory run after the upgrade, not a controlled JVM comparison.
The earlier Java 25 results used a different JDK distribution and were recorded
at a different time. Do not attribute the entire timing difference to Java 26;
allocation counts are unchanged.
