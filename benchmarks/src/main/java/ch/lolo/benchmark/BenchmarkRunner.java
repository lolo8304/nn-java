package ch.lolo.benchmark;

import ch.lolo.tensor.Tensor;
import org.openjdk.jmh.profile.GCProfiler;
import org.openjdk.jmh.results.format.ResultFormatType;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.options.*;
import java.nio.file.*;

/** Defaults can be overridden with normal JMH CLI options passed through Gradle --args. */
public final class BenchmarkRunner {
    public static void main(String[] args) throws Exception {
        String kind = System.getProperty("benchmark.kind", "kernel");
        String backend = Tensor.backend().name().toLowerCase(java.util.Locale.ROOT);
        Path directory = Path.of("benchmarks", "build", "results");
        Files.createDirectories(directory);
        var cli = new CommandLineOptions(args);
        var options = new OptionsBuilder().parent(cli);
        if (cli.getIncludes().isEmpty())
            options.include(switch (kind) {
                case "classification" -> ".*ClassificationBenchmarks.*";
                case "training" -> ".*TrainingBenchmarks.*";
                case "inference" -> ".*InferenceBenchmarks.*";
                default -> ".*KernelBenchmarks.*";
            });
        if (!cli.getForkCount().hasValue()) options.forks(2);
        if (!cli.getWarmupIterations().hasValue()) options.warmupIterations(3);
        if (!cli.getWarmupTime().hasValue()) options.warmupTime(TimeValue.seconds(1));
        if (!cli.getMeasurementIterations().hasValue()) options.measurementIterations(5);
        if (!cli.getMeasurementTime().hasValue()) options.measurementTime(TimeValue.seconds(1));
        if (!cli.getThreads().hasValue()) options.threads(1);
        if (!cli.getResult().hasValue()) options.result(directory.resolve(kind + "-" + backend + ".json").toString());
        if (!cli.getResultFormat().hasValue()) options.resultFormat(ResultFormatType.JSON);
        if (cli.getProfilers().isEmpty()) options.addProfiler(GCProfiler.class);
        if (!cli.getJvmArgs().hasValue()) options.jvmArgs(backend.equals("vector")
                ? new String[]{"-Xms256m", "-Xmx512m", "--add-modules=jdk.incubator.vector", "-Dtensor.backend=vector"}
                : new String[]{"-Xms256m", "-Xmx512m", "-Dtensor.backend=java"});
        options.shouldFailOnError(true);
        System.out.println("Backend: " + Tensor.backend() + "; Java: " + System.getProperty("java.version"));
        new Runner(options.build()).run();
    }
}
