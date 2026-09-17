package ch.lolo.benchmark;

import ch.lolo.tensor.Tensor;
import ch.lolo.nn.autograd.Variable;
import java.lang.management.ManagementFactory;

/** Small local comparison harness, not a substitute for JMH across hardware/JVMs. */
public class KernelBenchmark {
    private static volatile Object sink;
    public static void main(String[] args) {
        System.out.println("Tensor backend: " + Tensor.backend());
        System.out.println("Short smoke harness; use :benchmarks:kernel for forked JMH measurements.");
        Tensor x = Tensor.random(42L, 32, 128), y = Tensor.random(43L, 32, 128);
        Tensor bias = Tensor.random(44L, 128);
        measure("elementwise", () -> x.multiply(y).add(bias).relu());
        measure("copy", x::copy);
        measure("axis sum", () -> x.sum(0));
        measure("softmax backward", () -> {
            var v = Variable.parameter(x);
            v.softmax(-1).backward(y);
            return v.grad();
        });
        Tensor weights = Tensor.random(45L, 128, 64);
        measure("dense backward", () -> {
            var w = Variable.parameter(weights);
            Variable.of(x).matmul(w).pow(2).mean().backward();
            return w.grad();
        });
    }
    private static void measure(String name, java.util.function.Supplier<Object> operation) {
        var bean = (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
        long thread = Thread.currentThread().threadId();
        for (int i = 0; i < 100; i++) sink = operation.get();
        int count = 100;
        long bytes = bean.getThreadAllocatedBytes(thread), start = System.nanoTime();
        for (int i = 0; i < count; i++) sink = operation.get();
        long elapsed = System.nanoTime() - start;
        bytes = bean.getThreadAllocatedBytes(thread) - bytes;
        System.out.printf("%-18s %9.3f ms/op %12d bytes/op%n", name, elapsed / (count * 1e6), bytes / count);
    }
}
