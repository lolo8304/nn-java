package ch.lolo.benchmark;

import ch.lolo.tensor.Tensor;
import org.openjdk.jmh.annotations.*;
import java.util.concurrent.TimeUnit;

/** Backward contribution only: saved values and upstream gradients are prepared outside timing. */
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
public class ActivationBackwardBenchmarks {
    @Param({"relu", "leakyRelu", "sigmoid", "tanh"}) public String activation;
    @Param({"fused", "legacy"}) public String implementation;
    @Param({"false", "true"}) public boolean accumulate;
    @Param({"4096"}) public int size;
    private Tensor saved, upstream, destination;

    @Setup public void setup() {
        Tensor input = Tensor.random(42L, -4.0, 4.0, size);
        saved = switch (activation) {
            case "sigmoid" -> input.sigmoid();
            case "tanh" -> input.map(Math::tanh);
            default -> input;
        };
        upstream = Tensor.random(43L, -1.0, 1.0, size);
        destination = Tensor.zeros(size);
    }

    @Benchmark public Tensor backwardContribution() {
        if (implementation.equals("legacy")) {
            Tensor contribution = switch (activation) {
                case "relu" -> upstream.multiply(Tensor.generate(i -> saved.get(i) > 0 ? 1 : 0, saved.shape()));
                case "leakyRelu" -> upstream.multiply(Tensor.generate(i -> saved.get(i) >= 0 ? 1 : .13, saved.shape()));
                case "sigmoid" -> upstream.multiply(saved.multiply(Tensor.ones(saved.shape()).subtract(saved)));
                case "tanh" -> upstream.multiply(Tensor.ones(saved.shape()).subtract(saved.pow(2)));
                default -> throw new IllegalArgumentException(activation);
            };
            // Match Variable.addGrad's previous ownership and accumulation behavior.
            return accumulate ? destination.add(contribution) : contribution.copy();
        }
        Tensor out = accumulate ? destination : Tensor.zeros(size);
        return switch (activation) {
            case "relu" -> saved.reluBackwardInto(upstream, out, accumulate);
            case "leakyRelu" -> saved.leakyReluBackwardInto(upstream, out, accumulate, .13);
            case "sigmoid" -> saved.sigmoidBackwardInto(upstream, out, accumulate);
            case "tanh" -> saved.tanhBackwardInto(upstream, out, accumulate);
            default -> throw new IllegalArgumentException(activation);
        };
    }
}
