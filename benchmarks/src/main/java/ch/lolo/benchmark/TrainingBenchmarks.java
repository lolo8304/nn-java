package ch.lolo.benchmark;

import ch.lolo.nn.data.DataLoader;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.annotations.*;
import java.util.concurrent.TimeUnit;

@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
public class TrainingBenchmarks {
    @Param({"128", "784"}) public int features;
    @Param({"1024"}) public int samples;
    @Param({"32"}) public int batch;
    private TrainingWorkload workload;

    @Setup(Level.Iteration) public void setup() {
        workload = new TrainingWorkload(samples, features, batch, 42);
    }

    /** Assemble and consume every batch, including the epoch shuffle. */
    @Benchmark public void loadingEpoch(Blackhole sink) {
        for (var batch : new DataLoader(workload.dataset, workload.batchSize, true))
            sink.consume(batch);
    }

    @Benchmark public double trainingEpoch() {
        double value = workload.epoch();
        if (!Double.isFinite(value)) throw new IllegalStateException("Training diverged");
        return value;
    }
}
