package ch.lolo.tensor;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class KernelRegressionTest {
    @Test void emptyBroadcastPreservesZeroDimension() {
        assertArrayEquals(new int[]{0, 3}, Tensor.zeros(0, 3).add(Tensor.ones(1, 3)).shape());
        assertEquals(0, Tensor.ones(1, 3).multiply(Tensor.zeros(0, 3)).size());
    }

    @Test void kernelsRespectOffsetsStridesAndDetachedStorage() {
        Tensor base = Tensor.generate(i -> i[0] * 12 + i[1] * 4 + i[2], 3, 3, 4);
        for (Tensor view : new Tensor[]{base.slice(0, 1), base.slice(1, 1), base.slice(0, 1).transpose()}) {
            Tensor expected = Tensor.generate(view::get, view.shape());
            assertArrayEquals(expected.toArray(), view.copy().toArray());
            assertArrayEquals(expected.multiply(2).toArray(), view.map(v -> v * 2).toArray());
            assertArrayEquals(expected.add(expected).toArray(), view.add(view).toArray());
            assertEquals(expected.sum().scalar(), view.sum().scalar());
            Tensor bias = Tensor.generate(i -> i[0] + 1, view.shape()[1]);
            Tensor broadcast = view.add(bias);
            for (int i = 0; i < view.shape()[0]; i++)
                for (int j = 0; j < view.shape()[1]; j++)
                    assertEquals(view.get(i, j) + bias.get(j), broadcast.get(i, j));
            for (int axis = 0; axis < 2; axis++) {
                Tensor sum = view.sum(axis), softmax = view.softmax(axis);
                for (int outer = 0; outer < view.shape()[1 - axis]; outer++) {
                    double total = 0, denominator = 0;
                    for (int k = 0; k < view.shape()[axis]; k++) {
                        double v = axis == 0 ? view.get(k, outer) : view.get(outer, k);
                        total += v;
                        denominator += Math.exp(v);
                    }
                    assertEquals(total, sum.get(outer));
                    for (int k = 0; k < view.shape()[axis]; k++) {
                        double v = axis == 0 ? view.get(k, outer) : view.get(outer, k);
                        double actual = axis == 0 ? softmax.get(k, outer) : softmax.get(outer, k);
                        assertEquals(Math.exp(v) / denominator, actual, 1e-12);
                    }
                }
            }
            double original = view.toArray()[0];
            view.copy().set(-99, 0, 0);
            double[] array = view.toArray(); array[0] = -99;
            assertEquals(original, view.get(0, 0));
        }
    }

    @Test void scalarAndEmptyKernels() {
        assertEquals(6, Tensor.scalar(2).multiply(Tensor.scalar(3)).scalar());
        assertEquals(4, Tensor.scalar(2).map(x -> x * x).scalar());
        assertEquals(0, Tensor.zeros(2, 0).softmax(-1).size());
        assertArrayEquals(new double[]{0, 0}, Tensor.zeros(2, 0).sum(1).toArray());
    }
}
