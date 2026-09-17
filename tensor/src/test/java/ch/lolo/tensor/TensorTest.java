package ch.lolo.tensor;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class TensorTest {
    @Test
    void factoriesAndGenerate() {
        assertEquals(6, Tensor.ones(2, 3).sum().scalar());
        Tensor t = Tensor.generate(i -> i[0] * 10 + i[1], 2, 3);
        assertEquals(12, t.get(1, 2));
        assertArrayEquals(Tensor.random(42L, 42, 2, 3).toArray(), Tensor.random(42L, 42, 2, 3).toArray());
    }

    @Test
    void viewIsSharedAndCopyDetached() {
        Tensor a = Tensor.of(new double[][]{{1, 2}, {3, 4}});
        Tensor r = a.slice(0, 1);
        r.set(9, 0);
        assertEquals(9, a.get(1, 0));
        Tensor c = r.copy();
        c.set(7, 0);
        assertEquals(9, a.get(1, 0));
    }

    @Test
    void transposeAndReshape() {
        Tensor a = Tensor.of(new double[][]{{1, 2, 3}, {4, 5, 6}});
        assertEquals(4, a.transpose().get(0, 1));
        assertArrayEquals(new int[]{3, 2}, a.reshape(3, 2).shape());
    }

    @Test
    void broadcast() {
        Tensor a = Tensor.ones(2, 3);
        Tensor b = Tensor.of(1, 2, 3);
        assertArrayEquals(new double[]{2, 3, 4, 2, 3, 4}, a.add(b).toArray());
    }

    @Test
    void matmul() {
        Tensor a = Tensor.of(new double[][]{{1, 2, 3}, {4, 5, 6}});
        Tensor b = Tensor.of(new double[][]{{7, 8}, {9, 10}, {11, 12}});
        assertArrayEquals(new double[]{58, 64, 139, 154}, a.matmul(b).toArray(), 1e-12);
        assertEquals(32, Tensor.of(1, 2, 3).matmul(Tensor.of(4, 5, 6)).scalar());
    }

    @Test
    void reductionsAndMlOps() {
        Tensor a = Tensor.of(new double[][]{{1, 2, 3}, {4, 5, 6}});
        assertArrayEquals(new double[]{6, 15}, a.sum(1).toArray());
        assertArrayEquals(new double[]{1, 4, 9, 16, 25, 36}, a.pow(2).toArray());
        Tensor s = Tensor.of(1, 2, 3).softmax(0);
        assertEquals(1, s.sum().scalar(), 1e-12);
    }

    @Test
    void sorting() {
        Tensor a = Tensor.of(new double[][]{{4, 1, 3}, {9, 2, 5}});
        assertArrayEquals(new double[]{1, 3, 4, 2, 5, 9}, a.sort(1).toArray());
        assertArrayEquals(new double[]{1, 2, 3, 4, 5, 9}, a.sort().toArray());
    }
}
