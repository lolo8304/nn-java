# Fused activation backward (OPT-09)

Measured 2026-09-17 on Java 26.0.2. ReLU, LeakyReLU, sigmoid and tanh now
compute each backward contribution directly into an owned gradient destination.
The first contribution allocates just that destination; later contributions add
in place. Saved inputs/outputs and incoming adjoints remain intact for other
branches. General gradient reuse across resets remains OPT-10.

The tensor primitives use direct scalar loops on both backends, with one reusable
index cursor for strided views. They accept exact shapes, offset and strided
sources/destinations, and overwrite or accumulation mode. Exact aliases work in
place; differently mapped overlapping inputs use snapshots. Validation precedes
writes. No tensor-sized scratch buffer is allocated for disjoint destinations.
ReLU retains derivative zero at signed zero; LeakyReLU retains derivative one.
Sigmoid and tanh use saved outputs, preserving saturation behavior. Multiplication
by the derivative is retained even when zero, preserving NaN/infinity behavior.

## Measurement scope and reproduction

The benchmark compares the exact previous tensor formulas plus `addGrad`'s
copy/add with the new fused kernels, for 4,096 contiguous elements. Saved values,
upstream gradients and an existing destination are prepared outside timing.
“First” includes allocation of the owned destination; “add” uses an existing one.
This isolates backward contributions, excluding graph traversal, forward passes
and training. These are short allocation-focused measurements: one fork, two
200 ms warmups and three 200 ms measurements, one thread, GC profiler, 256–512 MiB
heap. Timing is indicative, not a full-epoch speedup claim. Raw samples and JMH
confidence intervals are in [Vector JSON](results/opt09/vector.json) and
[Java JSON](results/opt09/java.json), alongside the console logs.

```sh
./gradlew :benchmarks:kernel --args='ActivationBackwardBenchmarks -f 1 -wi 2 -i 3 -w 200ms -r 200ms -rff benchmarks/results/opt09/vector.json'
./gradlew :benchmarks:kernel -PtensorBackend=java --args='ActivationBackwardBenchmarks -f 1 -wi 2 -i 3 -w 200ms -r 200ms -rff benchmarks/results/opt09/java.json'
```

| Backend | Activation | Contribution | µs/op old → fused | B/op old → fused |
|---|---|---|---:|---:|
| VECTOR | relu | first | 26.597 → 1.630 | 197,177 → 32,864 |
| VECTOR | relu | add | 26.774 → 1.358 | 197,225 → 0 |
| VECTOR | leakyRelu | first | 26.620 → 1.632 | 197,177 → 32,864 |
| VECTOR | leakyRelu | add | 26.694 → 1.355 | 197,225 → 0 |
| VECTOR | sigmoid | first | 6.037 → 1.217 | 164,728 → 32,864 |
| VECTOR | sigmoid | add | 9.414 → 0.785 | 164,776 → 0 |
| VECTOR | tanh | first | 14.849 → 1.195 | 164,698 → 32,864 |
| VECTOR | tanh | add | 14.822 → 0.739 | 164,746 → 0 |
| JAVA | relu | first | 27.008 → 1.660 | 197,177 → 32,864 |
| JAVA | relu | add | 26.868 → 1.358 | 197,225 → 0 |
| JAVA | leakyRelu | first | 26.687 → 1.627 | 197,177 → 32,864 |
| JAVA | leakyRelu | add | 27.306 → 1.359 | 197,225 → 0 |
| JAVA | sigmoid | first | 6.085 → 1.202 | 164,728 → 32,864 |
| JAVA | sigmoid | add | 6.310 → 0.721 | 164,776 → 0 |
| JAVA | tanh | first | 15.245 → 1.214 | 164,699 → 32,864 |
| JAVA | tanh | add | 14.422 → 0.718 | 164,746 → 0 |

First-contribution allocation drops by about 80–83%. Accumulation reports
approximately zero B/op (profiler noise below one byte), versus 165–197 KB/op.
The old ReLU and LeakyReLU mask generation also allocated per-element index arrays.

## Verification

`./gradlew check` passes 91 tests per backend (182 executions) across Java and
Vector. New tests cover
nonuniform finite differences, signed zeros, saturation, NaN/infinity, scalars,
empty dimensions, offsets, strided views, exact and transposed aliases, validation
before mutation, branching consumers, repeated backward and leaf accumulation,
owned destination reuse, and seed/value preservation. Existing inference, training,
checkpoint and convergence checks also pass.
