package ch.lolo.benchmark;

import ch.lolo.nn.autograd.Variable;
import ch.lolo.nn.loss.*;
import ch.lolo.tensor.Tensor;
import org.openjdk.jmh.annotations.*;
import java.util.concurrent.TimeUnit;

/** Separate classification measurements; legacy is the exact pre-OPT-08 objective. */
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
public class ClassificationBenchmarks {
    @Param({"fused", "legacy"}) public String implementation;
    @Param({"128", "784"}) public int features;
    @Param({"1024"}) public int samples;
    @Param({"32"}) public int batch;
    private TrainingWorkload workload;
    private Loss loss;
    private Tensor logits, targets;

    @Setup(Level.Iteration) public void setup() {
        loss = implementation.equals("fused") ? new CrossEntropyLoss() : new LegacyLoss();
        workload = new TrainingWorkload(samples, features, batch, 42, loss, true);
        logits = Tensor.random(44L, -2.0, 2.0, batch, 10);
        targets = Tensor.generate(i -> i[1] == i[0] % 10 ? 1 : 0, batch, 10);
    }

    @Benchmark public double lossForwardBackward() {
        var x = Variable.parameter(logits);
        var value = loss.compute(x, targets);
        value.backward();
        return value.value().scalar() + x.grad().get(0, 0);
    }

    @Benchmark public double classificationEpoch() {
        double value = workload.epoch();
        if (!Double.isFinite(value)) throw new IllegalStateException("Classification diverged");
        return value;
    }

    static final class LegacyLoss implements Loss {
        public Variable compute(Variable logits, Tensor target) {
            double epsilon = 1e-12;
            var p = logits.softmax(-1).multiply(1 - 2 * epsilon).add(epsilon);
            int batch = logits.value().rank() > 1 ? logits.value().shape()[0] : 1;
            return Variable.of(target).multiply(p.log()).sum().multiply(-1).multiply(1.0 / batch);
        }
        public String type() { return "CrossEntropy"; }
    }
}
