package ch.lolo.tensor;

import org.openjdk.jmh.annotations.*;
import java.util.concurrent.TimeUnit;

/** Public-path layer shapes, including forward and both backward layouts. */
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
public class BlockedMatmulBenchmarks {
    @Param({"1x256x256", "32x10x64", "32x128x64", "32x784x64",
            "64x32x128", "128x256x256", "65x257x129", "256x512x512"})
    public String shape;
    @Param({"contiguous", "left", "right"}) public String layout;
    private Tensor a, b;

    @Setup(Level.Trial) public void setup() {
        String[] dimensions = shape.split("x");
        int m = Integer.parseInt(dimensions[0]);
        int k = Integer.parseInt(dimensions[1]);
        int n = Integer.parseInt(dimensions[2]);
        a = layout.equals("left") ? Tensor.random(42L, -1.0, 1.0, k, m).transpose()
                : Tensor.random(42L, -1.0, 1.0, m, k);
        b = layout.equals("right") ? Tensor.random(43L, -1.0, 1.0, n, k).transpose()
                : Tensor.random(43L, -1.0, 1.0, k, n);
    }

    @Benchmark public Tensor matmul() { return a.matmul(b); }
}
