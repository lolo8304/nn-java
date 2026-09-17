# Local kernel comparison

Measured on Apple M4 Pro (arm64), Homebrew OpenJDK 25.0.2, with a 256–512 MiB heap.
Baseline: commit `1a88c1f`, compiled directly with the same Java 25 compiler.
Updated: this working tree. Each workload has 100 warmup and 100 measured operations,
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

Run the commands in the root README to reproduce the updated measurements.
To compare another revision, compile its `tensor/src/main/java` and `nn/src/main/java`
sources into a separate class directory with `javac`, compile this same harness against
that directory, and run it with the same JVM flags. Do not mix baseline and updated classes.
