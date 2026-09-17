package ch.lolo.tensor;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class TransposedMatmulTest {
    private static void assertMatchesScalar(Tensor a, Tensor b) {
        int m = a.shape()[0], k = a.shape()[1], n = b.shape()[1];
        double[] beforeA = a.toArray(), beforeB = b.toArray();
        Tensor actual = a.matmul(b);
        assertArrayEquals(new int[]{m, n}, actual.shape());
        for (int row = 0; row < m; row++) for (int col = 0; col < n; col++) {
            double sum = 0;
            for (int q = 0; q < k; q++) sum += a.get(row, q) * b.get(q, col);
            assertEquals(Double.doubleToLongBits(sum), Double.doubleToLongBits(actual.get(row, col)),
                    "row=" + row + ", col=" + col);
        }
        assertArrayEquals(beforeA, a.toArray());
        assertArrayEquals(beforeB, b.toArray());
    }

    @Test
    void matchesOrderedScalarForTailsAndEmptyDimensions() {
        for (int m : new int[]{0, 1, 3, 4, 5}) for (int k : new int[]{0, 1, 2, 3, 7, 17, 64, 127})
            for (int n : new int[]{0, 1, 2, 3, 7, 9, 16, 33, 65}) {
                Tensor a = Tensor.random(42L, -1.0, 1.0, m, k);
                Tensor b = Tensor.random(43L, -1.0, 1.0, n, k).transpose();
                assertMatchesScalar(a, b);
            }
        assertMatchesScalar(Tensor.random(42L, -1.0, 1.0, 8, 19),
                Tensor.random(43L, -1.0, 1.0, 137, 19).transpose());
        assertMatchesScalar(Tensor.random(42L, -1.0, 1.0, 32, 10),
                Tensor.random(43L, -1.0, 1.0, 64, 10).transpose());
    }

    @Test
    void supportsOffsetsRowGapsAndNonunitInnerStridesOnBothSides() {
        int m = 5, k = 17, n = 19;
        Tensor contiguousA = Tensor.random(42L, 2, m, k).slice(0, 1);
        Tensor rowGapsA = Tensor.random(42L, m, 2, k).slice(1, 1);
        Tensor stridedA = Tensor.random(42L, k, 2, m).slice(1, 1).transpose();
        Tensor contiguousB = Tensor.random(43L, 2, n, k).slice(0, 1).transpose();
        Tensor rowGapsB = Tensor.random(43L, n, 2, k).slice(1, 1).transpose();
        Tensor stridedB = Tensor.random(43L, n, k, 2).slice(2, 1).transpose();
        for (Tensor a : new Tensor[]{contiguousA, rowGapsA, stridedA})
            for (Tensor b : new Tensor[]{contiguousB, rowGapsB, stridedB})
                assertMatchesScalar(a, b);
    }

    @Test
    void preservesCancellationSpecialValuesAndSignedZeros() {
        Tensor a = Tensor.of(new double[][]{
                {1e16, 1, -1e16, 1, 0, -0.0, 1e-300},
                {Double.POSITIVE_INFINITY, 1, 0, -1, 1, 0, 1},
                {Double.NaN, 1, 2, 3, 4, 5, 6},
                {-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0}});
        Tensor b = Tensor.of(new double[][]{
                {1, 1, 1, 1, 1, 1, 1},
                {0, 1, 0, 1, -1, 0, 1},
                {-1, 1, 1, 0, 1, 1, 1},
                {1, 2, 3, 4, 5, 6, 7},
                {Double.NEGATIVE_INFINITY, 0, 1, 0, 1, 0, 1}}).transpose();
        assertMatchesScalar(a, b);
        Tensor expandedA = Tensor.generate(i -> a.get(i[0], i[1] % 7), 4, 17);
        Tensor expandedB = Tensor.generate(i -> b.get(i[0] % 7, i[1] % 5), 17, 19);
        // Force a transposed physical layout through the same cancellation/special values.
        assertMatchesScalar(expandedA, expandedB.transpose().copy().transpose());
    }

    @Test
    void retainedTransposeViewSeesEveryWeightUpdate() {
        Tensor a = Tensor.ones(5, 17);
        Tensor weights = Tensor.random(42L, 19, 17);
        Tensor right = weights.transpose();
        Tensor first = a.matmul(right);
        weights.set(weights.get(2, 3) + 10, 2, 3);
        assertMatchesScalar(a, right);
        Tensor second = a.matmul(right);
        assertEquals(first.get(0, 2) + 10, second.get(0, 2), 1e-14);
        Tensor.zeros(19, 17).copyInto(weights);
        assertArrayEquals(new double[95], a.matmul(right).toArray());
    }
}
