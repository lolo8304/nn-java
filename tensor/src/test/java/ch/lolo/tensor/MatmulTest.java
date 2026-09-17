package ch.lolo.tensor;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MatmulTest {
    private void matchesReference(Tensor a, Tensor b) {
        int m = a.shape()[0], k = a.shape()[1], n = b.shape()[1];
        var expected = Tensor.generate(i -> {
            double sum = 0;
            for (int q = 0; q < k; q++) sum += a.get(i[0], q) * b.get(q, i[1]);
            return sum;
        }, m, n);
        var beforeA = a.toArray();
        var beforeB = b.toArray();
        assertArrayEquals(expected.toArray(), a.matmul(b).toArray(), 0);
        assertArrayEquals(beforeA, a.toArray());
        assertArrayEquals(beforeB, b.toArray());
    }

    @Test
    void matchesForContiguousTransposedAndOffsetViews() {
        for (boolean transposeA : new boolean[]{false, true}) {
            for (boolean transposeB : new boolean[]{false, true}) {
                var a = transposeA ? Tensor.random(42L, 5, 3).transpose() : Tensor.random(42L, 3, 5);
                var b = transposeB ? Tensor.random(43L, 7, 5).transpose() : Tensor.random(43L, 5, 7);
                matchesReference(a, b);
            }
        }
        matchesReference(Tensor.random(42L, 2, 3, 5).slice(0, 1), Tensor.random(43L, 2, 5, 7).slice(0, 1));
        matchesReference(Tensor.random(42L, 3, 2, 5).slice(1, 1), Tensor.random(43L, 5, 2, 7).slice(1, 1));
        matchesReference(Tensor.zeros(3, 0), Tensor.zeros(0, 7));
        matchesReference(Tensor.zeros(0, 5), Tensor.zeros(5, 7));
        matchesReference(Tensor.zeros(3, 5), Tensor.zeros(5, 0));
        assertThrows(IllegalArgumentException.class, () -> Tensor.zeros(2, 3).matmul(Tensor.zeros(4, 2)));
    }

    @Test
    void retainsVectorAndBroadcastedMatrixSupport() {
        assertArrayEquals(new double[]{6, 6}, Tensor.ones(2, 3).matmul(Tensor.of(1, 2, 3)).toArray());
        assertArrayEquals(new double[]{6, 6}, Tensor.of(1, 2, 3).matmul(Tensor.ones(3, 2)).toArray());
        var result = Tensor.ones(4, 2, 3).matmul(Tensor.ones(3, 5));
        assertArrayEquals(new int[]{4, 2, 5}, result.shape());
        for (double value : result.toArray()) assertEquals(3, value);
    }
}
