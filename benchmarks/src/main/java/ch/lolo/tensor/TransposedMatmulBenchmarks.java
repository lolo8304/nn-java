package ch.lolo.tensor;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import jdk.incubator.vector.DoubleVector;
import java.util.concurrent.TimeUnit;

/** Compares public matmul with direct and freshly packed kernel candidates. */
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
public class TransposedMatmulBenchmarks {
    @Param({"1x64x128", "32x64x128", "32x64x784", "32x10x64", "32x127x65", "1x256x256", "128x256x256"})
    public String shape;
    private int m, k, n;
    private Tensor left, right;
    private double[] a, b;

    @Setup(Level.Trial) public void setup() {
        String[] dimensions = shape.split("x");
        m = Integer.parseInt(dimensions[0]);
        k = Integer.parseInt(dimensions[1]);
        n = Integer.parseInt(dimensions[2]);
        left = Tensor.random(42L, -1.0, 1.0, m, k);
        Tensor weights = Tensor.random(43L, -1.0, 1.0, n, k);
        right = weights.transpose();
        a = left.toArray();
        b = weights.toArray();
    }

    @Benchmark public Tensor matmul() { return left.matmul(right); }

    // Raw candidates allocate the same output array; no public tensor wrapper overhead.
    @Benchmark public double[] direct() {
        double[] out = new double[m * n];
        GatherCandidate.matmulStridedRight(a, 0, k, 1, b, 0, 1, k, out, m, k, n);
        return out;
    }

    @Benchmark public double[] packed() {
        double[] packed = pack();
        double[] out = new double[m * n];
        VectorKernels.matmul(a, 0, k, 1, packed, 0, n, out, m, k, n);
        return out;
    }

    @Benchmark public double[] panelPacked() {
        double[] out = new double[m * n];
        VectorKernels.matmulPackedRight(a, 0, k, 1, b, 0, 1, k, out, m, k, n);
        return out;
    }

    @Benchmark public void panelPackOnly(Blackhole sink) {
        int width = Math.min(n, 64);
        double[] panel = new double[k * width];
        for (int start = 0; start < n; start += width) {
            int columns = Math.min(width, n - start);
            for (int q = 0; q < k; q++) for (int col = 0; col < columns; col++)
                panel[q * columns + col] = b[(start + col) * k + q];
            sink.consume(panel);
        }
    }

    @Benchmark public double[] packOnly() { return pack(); }

    private double[] pack() {
        double[] packed = new double[k * n];
        for (int q = 0; q < k; q++) for (int col = 0; col < n; col++)
            packed[q * n + col] = b[col * k + q];
        return packed;
    }
    // Retained rejected candidate for reproducibility. Requires the Vector module.
    private static final class GatherCandidate {
        private static final jdk.incubator.vector.VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
        // SIMD across output columns, gathering the right operand's strided columns.
        // Each lane accumulates q in order, with separate multiplication/addition (no FMA).
        static void matmulStridedRight(double[] a, int ao, int rowStride, int innerStride,
                                       double[] b, int bo, int rightRowStride, int rightColumnStride,
                                       double[] out, int m, int k, int n) {
            if (m == 0 || k == 0 || n == 0) return;
            int bound = SPECIES.loopBound(n);
            int[] columns = new int[SPECIES.length()];
            for (int lane = 0; lane < columns.length; lane++) columns[lane] = lane * rightColumnStride;
            for (int row = 0; row < m; row++) {
                int left = ao + row * rowStride, output = row * n;
                int col = 0;
                for (; col < bound; col += SPECIES.length()) {
                    var sum = DoubleVector.zero(SPECIES);
                    int right = bo + col * rightColumnStride;
                    for (int q = 0; q < k; q++) {
                        var av = DoubleVector.broadcast(SPECIES, a[left + q * innerStride]);
                        var bv = DoubleVector.fromArray(SPECIES, b, right + q * rightRowStride, columns, 0);
                        sum = sum.add(av.mul(bv));
                    }
                    sum.intoArray(out, output + col);
                }
                for (; col < n; col++) {
                    double sum = 0;
                    int right = bo + col * rightColumnStride;
                    for (int q = 0; q < k; q++)
                        sum += a[left + q * innerStride] * b[right + q * rightRowStride];
                    out[output + col] = sum;
                }
            }
        }

    }

}
