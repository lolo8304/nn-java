package ch.lolo.benchmark;

import ch.lolo.tensor.Tensor;
import org.openjdk.jmh.annotations.*;
import java.util.concurrent.TimeUnit;

/** Compare explicit dispatch with its retained Java reference at crossover sizes. */
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
public class SigmoidCoverageBenchmarks {
    @Param({"16", "128", "4096"}) public int size;
    private Tensor input;
    @Setup public void setup() { input = Tensor.random(42L, -5.0, 5.0, size); }
    @Benchmark public Tensor dispatched() { return input.sigmoid(); }
    @Benchmark public Tensor reference() { return input.map(v -> 1 / (1 + Math.exp(-v))); }
}
