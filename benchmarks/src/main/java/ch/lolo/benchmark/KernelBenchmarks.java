package ch.lolo.benchmark;

import ch.lolo.tensor.Tensor;
import ch.lolo.nn.autograd.Variable;
import org.openjdk.jmh.annotations.*;
import java.util.concurrent.TimeUnit;

@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
public class KernelBenchmarks {
    @Param({"128", "784"}) public int features;
    @Param({"32"}) public int batch;
    private Tensor x, y, bias, weights, transposedWeights, out, strided;

    @Setup(Level.Trial) public void setup() {
        x = Tensor.random(42L, batch, features);
        y = Tensor.random(43L, batch, features);
        bias = Tensor.random(44L, features);
        weights = Tensor.random(45L, features, 64);
        transposedWeights = Tensor.random(46L, 64, features).transpose();
        out = Tensor.zeros(batch, features);
        strided = x.transpose();
    }

    @Benchmark public Tensor elementwise() { return x.multiply(y).add(bias).relu(); }
    @Benchmark public Tensor elementwiseInto() {
        x.multiplyInto(y, out);
        out.addInto(bias, out);
        return out.reluInto(out);
    }
    @Benchmark public Tensor copy() { return x.copy(); }
    @Benchmark public Tensor copyInto() { return x.copyInto(out); }
    @Benchmark public Tensor axisSum() { return x.sum(0); }
    @Benchmark public Tensor stridedMap() { return strided.multiply(2); }
    @Benchmark public Tensor matmul() { return x.matmul(weights); }
    @Benchmark public Tensor matmulTransposedRight() { return x.matmul(transposedWeights); }
    @Benchmark public Tensor softmaxBackward() {
        var v = Variable.parameter(x);
        v.softmax(-1).backward(y);
        return v.grad();
    }
    @Benchmark public Tensor denseBackward() {
        // Both input and weight gradients: includes the transpose-right backward path.
        var input = Variable.parameter(x);
        var w = Variable.parameter(weights);
        input.matmul(w).pow(2).mean().backward();
        return w.grad();
    }
}
