package ch.lolo.benchmark;

import ch.lolo.nn.autograd.Variable;
import ch.lolo.tensor.Tensor;
import org.openjdk.jmh.annotations.*;
import java.util.concurrent.TimeUnit;

/** Repeated backward on a shared graph, excluding graph construction. */
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
public class GradientStorageBenchmarks {
    private Variable leaf, output;
    private Tensor seed;

    @Setup public void setup() {
        leaf = Variable.parameter(Tensor.random(42L, 4096));
        var shared = leaf.add(1);
        output = shared.add(shared).add(leaf);
        seed = Tensor.ones(4096);
    }

    @Benchmark public Tensor backward() {
        leaf.zeroGrad();
        output.backward(seed);
        return leaf.grad();
    }
}
