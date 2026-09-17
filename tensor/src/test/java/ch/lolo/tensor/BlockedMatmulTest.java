package ch.lolo.tensor;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class BlockedMatmulTest {
    private static void assertOrdered(Tensor a, Tensor b) {
        double[] beforeA = a.toArray(), beforeB = b.toArray();
        Tensor result = a.matmul(b);
        int m = a.shape()[0], k = a.shape()[1], n = b.shape()[1];
        assertArrayEquals(new int[]{m, n}, result.shape());
        for (int row = 0; row < m; row++) for (int col = 0; col < n; col++) {
            double sum = 0;
            for (int q = 0; q < k; q++) sum += a.get(row, q) * b.get(q, col);
            assertEquals(Double.doubleToLongBits(sum), Double.doubleToLongBits(result.get(row, col)),
                    "row=" + row + ", col=" + col);
        }
        assertArrayEquals(beforeA, a.toArray());
        assertArrayEquals(beforeB, b.toArray());
    }

    @Test void tileTailsAndDispatchBoundariesInEveryLayout() {
        for (int[] shape : new int[][]{{15, 256, 128}, {16, 256, 128}, {33, 257, 129},
                {32, 255, 128}, {32, 128, 256}, {17, 1025, 33}, {32, 127, 259}, {17, 129, 513}}) {
            int m = shape[0], k = shape[1], n = shape[2];
            Tensor a = Tensor.random(42L, -1.0, 1.0, m, k);
            Tensor b = Tensor.random(43L, -1.0, 1.0, k, n);
            for (Tensor left : new Tensor[]{a, a.transpose().copy().transpose()})
                for (Tensor right : new Tensor[]{b, b.transpose().copy().transpose()})
                    assertOrdered(left, right);
        }
    }

    @Test void offsetsGapsAndStridesInBothOperands() {
        int m = 33, k = 257, n = 129;
        Tensor a = Tensor.random(42L, m, 2, k).slice(1, 1);
        Tensor at = Tensor.random(42L, k, 2, m).slice(1, 1).transpose();
        Tensor b = Tensor.random(43L, k, 2, n).slice(1, 1);
        Tensor bt = Tensor.random(43L, n, k, 2).slice(2, 1).transpose();
        for (Tensor left : new Tensor[]{a, at}) for (Tensor right : new Tensor[]{b, bt})
            assertOrdered(left, right);
    }

    @Test void preservesReductionAcrossTilesAndSeesMutations() {
        double[] values = {1e16, 1, -1e16, 1, -0.0, 1e-300, -1e-300};
        Tensor a = Tensor.generate(i -> values[i[1] % values.length], 17, 257);
        a.set(Double.NaN, 1, 64);
        a.set(Double.POSITIVE_INFINITY, 2, 128);
        a.set(Double.NEGATIVE_INFINITY, 3, 192);
        Tensor b = Tensor.generate(i -> i[1] % 3 == 0 ? -0.0 : 1.0, 257, 129);
        assertOrdered(a, b);
        Tensor transposed = b.transpose().copy().transpose();
        assertOrdered(a, transposed);
        transposed.set(7, 64, 65);
        assertOrdered(a, transposed);
        Tensor.zeros(257, 129).copyInto(transposed);
        assertOrdered(a, transposed);
    }
}
