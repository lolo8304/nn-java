package ch.lolo.benchmark;

import ch.lolo.tensor.Tensor;
import org.openjdk.jmh.annotations.*;
import java.util.concurrent.TimeUnit;

/** Public operations, including dispatch and allocation; fixed seeds outside timing. */
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
public class CoverageBenchmarks {
    @Param({"sum", "axisSum", "min", "axisMax", "softmax", "broadcastRows", "broadcastColumns",
            "stridedMap", "stridedCopy", "batchedMatmul", "batchedTranspose", "negate", "square", "sigmoid", "tanh", "log"})
    public String operation;
    private Tensor x, strided, rows, columns, batches, right, transposed;
    @Setup public void setup() {
        x = Tensor.random(42L, 32, 128);
        strided = x.transpose();
        rows = Tensor.random(43L, 2, 1, 128);
        columns = Tensor.random(44L, 2, 32, 1);
        batches = Tensor.random(45L, 2, 32, 128);
        right = Tensor.random(46L, 1, 128, 64);
        transposed = Tensor.random(47L, 1, 64, 128).transpose(0, 2, 1);
    }
    @Benchmark public Tensor run() {
        return switch (operation) {
            case "sum" -> x.sum();
            case "axisSum" -> x.sum(0);
            case "min" -> x.min();
            case "axisMax" -> x.max(0);
            case "softmax" -> x.softmax(-1);
            case "broadcastRows" -> batches.subtract(rows);
            case "broadcastColumns" -> columns.divide(batches);
            case "stridedMap" -> strided.map(v -> v + 1);
            case "stridedCopy" -> strided.copy();
            case "batchedMatmul" -> batches.matmul(right);
            case "batchedTranspose" -> batches.matmul(transposed);
            case "negate" -> x.negate();
            case "square" -> x.pow(2);
            case "sigmoid" -> x.sigmoid();
            case "tanh" -> x.tanh();
            case "log" -> x.log();
            default -> throw new IllegalArgumentException(operation);
        };
    }
}
