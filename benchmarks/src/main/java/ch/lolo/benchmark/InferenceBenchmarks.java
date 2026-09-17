package ch.lolo.benchmark;

import ch.lolo.nn.NN;
import ch.lolo.nn.Sequential;
import ch.lolo.nn.layers.*;
import ch.lolo.tensor.Tensor;
import org.openjdk.jmh.annotations.*;

import java.util.concurrent.TimeUnit;

/** One prediction batch; construction and input generation are outside timing. */
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
public class InferenceBenchmarks {
    @Param({"128", "784"}) public int features;
    @Param({"1", "32"}) public int batch;
    private Sequential model;
    private Tensor input;

    @Setup(Level.Trial) public void setup() {
        NN.seed(42);
        model = Sequential.of(new Dense(features, 64), new ReLU(), new Dropout(.2), new Dense(64, 10));
        input = Tensor.random(43L, -1.0, 1.0, batch, features);
    }

    @Benchmark public Tensor predict() {
        return model.predict(input);
    }
}
