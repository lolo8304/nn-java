package ch.lolo.tensor;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ActivationBackwardTest {
    private Tensor apply(int kind, Tensor saved, Tensor seed, Tensor out, boolean add) {
        return switch (kind) {
            case 0 -> saved.reluBackwardInto(seed, out, add);
            case 1 -> saved.leakyReluBackwardInto(seed, out, add, .13);
            case 2 -> saved.sigmoidBackwardInto(seed, out, add);
            default -> saved.tanhBackwardInto(seed, out, add);
        };
    }

    private Tensor reference(int kind, Tensor saved, Tensor seed) {
        return seed.multiply(Tensor.generate(i -> {
            double x = saved.get(i);
            return switch (kind) {
                case 0 -> x > 0 ? 1 : 0;
                case 1 -> x >= 0 ? 1 : .13;
                case 2 -> x * (1 - x);
                default -> 1 - Math.pow(x, 2);
            };
        }, saved.shape()));
    }

    @Test void contiguousOffsetsStridesAndEmptyShapes() {
        for (int kind = 0; kind < 4; kind++) for (int n : new int[]{0, 1, 3, 17, 65})
            for (boolean strided : new boolean[]{false, true}) {
                Tensor saved = Tensor.random(1L, -1.0, 1.0, 2, n, 3).slice(0, 1);
                Tensor seed = Tensor.random(2L, -2.0, 2.0, 2, n, 3).slice(0, 1);
                Tensor storage = Tensor.ones(2, n, 3).multiply(-99);
                Tensor out = storage.slice(0, 1);
                if (strided) { saved = saved.transpose(); seed = seed.transpose(); out = out.transpose(); }
                double[] before = saved.toArray(), seedBefore = seed.toArray();
                Tensor expected = reference(kind, saved, seed);
                assertSame(out, apply(kind, saved, seed, out, false));
                assertArrayEquals(expected.toArray(), out.toArray());
                apply(kind, saved, seed, out, true);
                assertArrayEquals(expected.add(expected).toArray(), out.toArray());
                assertArrayEquals(before, saved.toArray());
                assertArrayEquals(seedBefore, seed.toArray());
                for (double x : storage.slice(0, 0).toArray()) assertEquals(-99, x);
            }
    }

    @Test void specialValuesAndScalarsMatchPreviousArithmetic() {
        Tensor saved = Tensor.of(Double.NaN, Double.NEGATIVE_INFINITY, -1, -0.0, 0, 1, Double.POSITIVE_INFINITY);
        for (int kind = 0; kind < 4; kind++) for (double seed : new double[]{-2, 0, Double.POSITIVE_INFINITY, Double.NaN}) {
            Tensor upstream = Tensor.ones(7).multiply(seed), out = Tensor.zeros(7);
            apply(kind, saved, upstream, out, false);
            assertArrayEquals(reference(kind, saved, upstream).toArray(), out.toArray());
            Tensor scalar = Tensor.scalar(.3), g = Tensor.scalar(-2), d = Tensor.scalar(42);
            apply(kind, scalar, g, d, false);
            assertEquals(reference(kind, scalar, g).scalar(), d.scalar());
        }
    }

    @Test void aliasesUseOriginalValuesAndShapesAreValidatedBeforeWrites() {
        for (int kind = 0; kind < 4; kind++) for (boolean add : new boolean[]{false, true})
            for (int alias = 0; alias < 4; alias++) {
                Tensor out = Tensor.random(1L, -1.0, 1.0, 3, 3);
                Tensor saved = alias < 2 ? out : out.transpose();
                Tensor seed = alias % 2 == 0 ? out : out.transpose();
                Tensor expected = reference(kind, saved, seed);
                if (add) expected = out.add(expected);
                apply(kind, saved, seed, out, add);
                assertArrayEquals(expected.toArray(), out.toArray());
            }
        for (int kind = 0; kind < 4; kind++) {
            final int k = kind;
            Tensor out = Tensor.ones(2, 3), wrong = Tensor.ones(3, 2);
            assertThrows(IllegalArgumentException.class, () -> apply(k, out, wrong, out, false));
            assertThrows(IllegalArgumentException.class, () -> apply(k, wrong, wrong, out, true));
            assertArrayEquals(Tensor.ones(2, 3).toArray(), out.toArray());
        }
    }
}
