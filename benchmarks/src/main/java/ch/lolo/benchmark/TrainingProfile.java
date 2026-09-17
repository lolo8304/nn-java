package ch.lolo.benchmark;

import ch.lolo.nn.autograd.Variable;
import ch.lolo.nn.data.*;
import ch.lolo.tensor.Tensor;
import java.lang.management.ManagementFactory;
import java.util.Arrays;

/** Phase instrumentation is diagnostic; use JMH trainingEpoch for uninstrumented timing. */
public final class TrainingProfile {
    private static final String[] PHASES = {"load/shuffle", "zero gradients", "forward + loss", "backward", "optimizer"};
    private static final com.sun.management.ThreadMXBean ALLOC =
            (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
    private static final long THREAD = Thread.currentThread().threadId();
    private static final boolean ALLOCATION_SUPPORTED = ALLOC.isThreadAllocatedMemorySupported();
    static {
        if (ALLOCATION_SUPPORTED && !ALLOC.isThreadAllocatedMemoryEnabled()) ALLOC.setThreadAllocatedMemoryEnabled(true);
    }
    private static long bytes() { return ALLOCATION_SUPPORTED ? ALLOC.getThreadAllocatedBytes(THREAD) : 0; }
    private static long gcCount() {
        return ManagementFactory.getGarbageCollectorMXBeans().stream().mapToLong(b -> Math.max(0, b.getCollectionCount())).sum();
    }
    private static long gcTime() {
        return ManagementFactory.getGarbageCollectorMXBeans().stream().mapToLong(b -> Math.max(0, b.getCollectionTime())).sum();
    }

    public static void main(String[] args) {
        if (args.length > 5) throw new IllegalArgumentException("Expected [warmupEpochs runs measuredEpochs features samples]");
        int warmup = positive(args, 0, 5), runs = positive(args, 1, 3), epochs = positive(args, 2, 5);
        int features = positive(args, 3, 128), samples = positive(args, 4, 1024);
        System.out.printf("Backend=%s Java=%s samples=%d features=%d batch=32 warmup=%d runs=%d epochs/run=%d%n",
                Tensor.backend(), System.getProperty("java.version"), samples, features, warmup, runs, epochs);
        if (!ALLOCATION_SUPPORTED) System.out.println("Thread allocation counters unavailable; byte columns are zero.");
        System.out.println("Synthetic regression, Dense(features,64)/ReLU/Dense(64,10), MSE, Adam. No augmentation, validation, logging or checkpoint IO in timing.");
        double[] epochTimes = new double[runs];
        for (int run = 0; run < runs; run++) {
            TrainingWorkload workload = new TrainingWorkload(samples, features, 32, 42);
            for (int i = 0; i < warmup; i++) workload.epoch();
            long[] nanos = new long[PHASES.length], allocated = new long[PHASES.length];
            long gcBefore = gcCount(), gcMillisBefore = gcTime(), allBytes = bytes(), start = System.nanoTime();
            double lastLoss = 0;
            for (int epoch = 0; epoch < epochs; epoch++) {
                double totalLoss = 0;
                long markBytes = bytes(), mark = System.nanoTime();
                var iterator = new DataLoader(workload.dataset, 32, true).iterator();
                record(0, mark, markBytes, nanos, allocated);
                while (iterator.hasNext()) {
                    markBytes = bytes(); mark = System.nanoTime();
                    Batch batch = iterator.next();
                    record(0, mark, markBytes, nanos, allocated);
                    markBytes = bytes(); mark = System.nanoTime();
                    workload.optimizer.zeroGrad(workload.parameters);
                    record(1, mark, markBytes, nanos, allocated);
                    markBytes = bytes(); mark = System.nanoTime();
                    var prediction = workload.model.forward(Variable.of(batch.inputs()), true);
                    var loss = workload.loss.compute(prediction, batch.targets());
                    record(2, mark, markBytes, nanos, allocated);
                    markBytes = bytes(); mark = System.nanoTime();
                    loss.backward();
                    record(3, mark, markBytes, nanos, allocated);
                    markBytes = bytes(); mark = System.nanoTime();
                    workload.optimizer.step(workload.parameters);
                    record(4, mark, markBytes, nanos, allocated);
                    totalLoss += loss.value().scalar() * batch.inputs().shape()[0];
                }
                lastLoss = totalLoss / samples;
                if (!Double.isFinite(lastLoss)) throw new IllegalStateException("Training diverged");
            }
            long elapsed = System.nanoTime() - start, totalBytes = bytes() - allBytes;
            epochTimes[run] = elapsed / (epochs * 1e6);
            System.out.printf("Run %d: %.3f ms/epoch, %.0f samples/s, %.0f bytes/epoch, GC=%d collections/%d ms, loss=%.8f%n",
                    run + 1, epochTimes[run], samples * (double) epochs * 1e9 / elapsed,
                    totalBytes / (double) epochs, gcCount() - gcBefore, gcTime() - gcMillisBefore, lastLoss);
            for (int phase = 0; phase < PHASES.length; phase++)
                System.out.printf("  %-18s %9.3f ms/epoch %6.1f%% %12.0f bytes/epoch%n", PHASES[phase],
                        nanos[phase] / (epochs * 1e6), 100.0 * nanos[phase] / elapsed, allocated[phase] / (double) epochs);
            System.out.println("  Percentages exclude loop/instrumentation overhead; allocation counters cover this thread only.");
        }
        Arrays.sort(epochTimes);
        double median = (epochTimes[(runs - 1) / 2] + epochTimes[runs / 2]) / 2;
        System.out.printf("Epoch ms across runs: min=%.3f median=%.3f max=%.3f%n", epochTimes[0], median, epochTimes[runs - 1]);
    }

    private static void record(int phase, long start, long beforeBytes, long[] nanos, long[] allocated) {
        nanos[phase] += System.nanoTime() - start;
        allocated[phase] += bytes() - beforeBytes;
    }
    private static int positive(String[] args, int index, int fallback) {
        int value = args.length > index ? Integer.parseInt(args[index]) : fallback;
        if (value < 1) throw new IllegalArgumentException("All profile arguments must be positive");
        return value;
    }
}
