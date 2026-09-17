package ch.lolo.tensor;

import org.junit.jupiter.api.Test;
import java.util.ArrayList;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class CoverageKernelTest {
    private enum UnaryOp {
        NEGATE, SQUARE, SIGMOID, TANH, LOG;
        double applyAsDouble(double value) {
            return switch (this) {
                case NEGATE -> -value; case SQUARE -> Math.pow(value, 2);
                case SIGMOID -> 1 / (1 + Math.exp(-value));
                case TANH -> Math.tanh(value); case LOG -> Math.log(value);
            };
        }
    }

    private static void exact(double expected, double actual) {
        assertEquals(Double.doubleToLongBits(expected), Double.doubleToLongBits(actual),
                () -> expected + " != " + actual);
    }

    @Test void reductionsPreserveOrderedArithmeticAndSpecialValues() {
        double[] values = {1e300, 1, -1e300, -0.0, 0.0, Double.MIN_VALUE,
                Double.MAX_VALUE, Double.MAX_VALUE, -Double.MAX_VALUE,
                Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY};
        for (int shift = 0; shift < values.length; shift++) {
            final int start = shift;
            Tensor base = Tensor.generate(i -> values[(start + i[0] * 17 + i[1]) % values.length], 5, 17);
            for (Tensor x : new Tensor[]{base, base.transpose(), base.slice(0, 1), base.slice(1, 1)}) {
                double sum = 0, min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
                for (double v : x.toArray()) { sum += v; min = Math.min(min, v); max = Math.max(max, v); }
                exact(sum, x.sum().scalar()); exact(min, x.min().scalar()); exact(max, x.max().scalar());
                for (int ax = 0; ax < x.rank(); ax++) checkAxis(x, ax);
            }
        }
        Tensor ordered = Tensor.generate(i -> new double[]{1e16, 1, -1e16, 1}[i[0]], 4, 17);
        checkAxis(ordered, 0);
        for (double v : ordered.sum(0).toArray()) exact(1, v);
        Tensor overflow = Tensor.generate(i -> i[0] < 2 ? Double.MAX_VALUE : -Double.MAX_VALUE, 3, 17);
        checkAxis(overflow, 0);
        Tensor cancellation = Tensor.of(1e16, 1, -1e16, 1);
        exact(1, cancellation.sum().scalar());
        exact(-0.0, Tensor.of(0.0, -0.0, 0.0, -0.0, 0.0).min().scalar());
        exact(0.0, Tensor.of(-0.0, 0.0, -0.0, 0.0, -0.0).max().scalar());
        Tensor finite = Tensor.random(18L, -10.0, 10.0, 2, 3, 17);
        for (Tensor x : new Tensor[]{finite, finite.transpose(2, 0, 1)})
            for (int ax = 0; ax < x.rank(); ax++) checkAxis(x, ax);
    }

    private static void checkAxis(Tensor x, int axis) {
        int[] shape = x.shape(), output = new int[shape.length - 1];
        for (int d = 0, j = 0; d < shape.length; d++) if (d != axis) output[j++] = shape[d];
        for (int op = 0; op < 3; op++) {
            final int operation = op;
            Tensor expected = Tensor.generate(index -> {
                int[] input = new int[shape.length];
                for (int d = 0, j = 0; d < shape.length; d++) if (d != axis) input[d] = index[j++];
                double acc = operation == 0 ? 0 : operation == 1 ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
                for (int k = 0; k < shape[axis]; k++) {
                    input[axis] = k;
                    double v = x.get(input);
                    acc = operation == 0 ? acc + v : operation == 1 ? Math.min(acc, v) : Math.max(acc, v);
                }
                return acc;
            }, output);
            double[] actual = (op == 0 ? x.sum(axis) : op == 1 ? x.min(axis) : x.max(axis)).toArray();
            double[] reference = expected.toArray();
            assertEquals(reference.length, actual.length);
            for (int i = 0; i < actual.length; i++) exact(reference[i], actual[i]);
        }
    }

    @Test void emptyAndScalarShapes() {
        Tensor empty = Tensor.zeros(2, 0, 17);
        exact(0, empty.sum().scalar());
        assertThrows(IllegalStateException.class, empty::min);
        assertThrows(IllegalStateException.class, empty::max);
        for (int ax = 0; ax < 3; ax++) { checkAxis(empty, ax); assertEquals(0, empty.softmax(ax).size()); }
        assertEquals(0, empty.negate().size());
        assertEquals(0, empty.add(Tensor.ones(1, 1, 17)).size());
        assertEquals(0, empty.matmul(Tensor.zeros(17, 3)).size());
        assertArrayEquals(new double[12], Tensor.zeros(2, 3, 0).matmul(Tensor.zeros(1, 0, 2)).toArray());
        exact(-2, Tensor.scalar(2).negate().scalar());
        exact(4, Tensor.scalar(2).square().scalar());
        exact(2, Tensor.scalar(2).sum().scalar());
    }

    @Test void broadcastsBothOperandOrdersAndAliasedDestinations() {
        Tensor large = Tensor.random(9L, 1, 2, 2, 3, 17);
        for (Tensor small : new Tensor[]{Tensor.random(10L, 1, 2, 1, 3, 1),
                Tensor.random(11L, 17), Tensor.random(12L, 2, 1, 17),
                Tensor.random(13L, 1, 2, 3, 1), Tensor.random(14L, 2, 3, 17).transpose(1, 0, 2).slice(0, 1).reshape(2, 1, 17)}) {
            checkBroadcast(large, small);
            checkBroadcast(small, large);
        }
        Tensor cube = Tensor.random(15L, 3, 3, 3);
        Tensor expected = cube.subtract(cube.transpose(2, 0, 1));
        cube.subtractInto(cube.transpose(2, 0, 1), cube);
        assertArrayEquals(expected.toArray(), cube.toArray());
        Tensor x = Tensor.random(16L, 2, 3, 17);
        Tensor row = x.slice(1, 0).reshape(2, 1, 17);
        expected = x.divide(row);
        x.divideInto(row, x);
        assertArrayEquals(expected.toArray(), x.toArray());
        Tensor backing = Tensor.zeros(17, 3, 2);
        Tensor destination = backing.transpose(2, 1, 0);
        x.addInto(Tensor.ones(2, 1, 17), destination);
        assertArrayEquals(x.add(1).toArray(), destination.toArray());
    }

    private static void checkBroadcast(Tensor a, Tensor b) {
        int[] shape = TensorShape.broadcast(a.shape(), b.shape());
        for (BinaryOp op : BinaryOp.values()) {
            Tensor expected = Tensor.generate(index -> op.applyAsDouble(broadcastGet(a, index), broadcastGet(b, index)), shape);
            Tensor actual = switch (op) {
                case ADD -> a.add(b); case SUBTRACT -> a.subtract(b);
                case MULTIPLY -> a.multiply(b); case DIVIDE -> a.divide(b);
            };
            assertArrayEquals(expected.toArray(), actual.toArray());
        }
    }

    private static double broadcastGet(Tensor x, int[] index) {
        int[] shape = x.shape(), at = new int[shape.length];
        for (int d = 0; d < at.length; d++) at[d] = shape[d] == 1 ? 0 : index[index.length - at.length + d];
        return x.get(at);
    }

    @Test void traversalPreservesOrderAndGeneratorIndexOwnership() {
        Tensor x = Tensor.random(31L, 2, 3, 4, 5).transpose(3, 1, 0, 2).slice(1, 1);
        Tensor reference = Tensor.generate(x::get, x.shape());
        assertArrayEquals(reference.toArray(), x.copy().toArray());
        List<Double> calls = new ArrayList<>();
        assertArrayEquals(reference.add(2).toArray(), x.map(v -> { calls.add(v); return v + 2; }).toArray());
        assertArrayEquals(reference.toArray(), calls.stream().mapToDouble(Double::doubleValue).toArray());
        List<int[]> indices = new ArrayList<>();
        Tensor.generate(i -> { indices.add(i); return 0; }, 2, 3);
        assertNotSame(indices.get(0), indices.get(1));
        assertArrayEquals(new int[]{0, 0}, indices.get(0));
        assertArrayEquals(new int[]{1, 2}, indices.get(5));
        Tensor.zeros(0, 2).transpose().map(v -> { fail("empty callback"); return v; });
    }

    @Test void batchedMatmulMatchesOrderedReferenceIncludingViewsVectorsAndTiling() {
        for (int[] dims : new int[][]{{3, 5, 17}, {16, 129, 257}}) {
            int m = dims[0], k = dims[1], n = dims[2];
            Tensor a = Tensor.random(21L, -1.0, 1.0, 2, 1, m, k);
            Tensor b = Tensor.random(22L, -1.0, 1.0, 1, 3, n, k).transpose(0, 1, 3, 2);
            checkMatmul(a, b);
            checkMatmul(a, b.copy());
            b.set(99, 0, 1, 1, 1);
            checkMatmul(a, b);
        }
        checkMatmul(Tensor.random(28L, 2, 3, 5, 7).slice(0, 1),
                Tensor.random(29L, 3, 7, 2, 17).slice(2, 1));
        Tensor a = Tensor.random(23L, 2, 7, 5).transpose(0, 2, 1);
        checkMatmul(a, Tensor.random(24L, 7));
        checkMatmul(Tensor.random(25L, 5), a);
        checkMatmul(Tensor.random(26L, 7), Tensor.random(27L, 7));
        checkMatmul(Tensor.of(1e16, 1, -1e16, 1).reshape(1, 1, 4), Tensor.ones(1, 4, 3));
    }

    private static void checkMatmul(Tensor a, Tensor b) {
        int[] as = a.shape(), bs = b.shape();
        boolean av = as.length == 1, bv = bs.length == 1;
        int k = as[as.length - 1];
        Tensor actual = a.matmul(b);
        Tensor expected = Tensor.generate(index -> {
            int batchRank = index.length - (av ? 0 : 1) - (bv ? 0 : 1);
            int[] ai = new int[as.length], bi = new int[bs.length];
            if (!av) {
                for (int d = 0; d < as.length - 2; d++) ai[d] = as[d] == 1 ? 0 : index[batchRank - (as.length - 2) + d];
                ai[as.length - 2] = index[batchRank];
            }
            if (!bv) {
                for (int d = 0; d < bs.length - 2; d++) bi[d] = bs[d] == 1 ? 0 : index[batchRank - (bs.length - 2) + d];
                bi[bs.length - 1] = index[index.length - 1];
            }
            double sum = 0;
            for (int q = 0; q < k; q++) { ai[as.length - 1] = q; bi[bv ? 0 : bs.length - 2] = q; sum += a.get(ai) * b.get(bi); }
            return sum;
        }, actual.shape());
        assertArrayEquals(expected.toArray(), actual.toArray());
    }

    @Test void unarySpecialValuesTailsOffsetsAndApproximation() {
        double[] values = {Double.NaN, Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, -0.0, 0.0,
                Double.MIN_VALUE, -Double.MIN_VALUE, Double.MAX_VALUE, -Double.MAX_VALUE, -1000.0, 1000.0, -1.0, 1.0, 0.25, 17, -17, 1e-100, -710, -709, 709, 710, -745, 745};
        Tensor base = Tensor.generate(i -> values[i[1] % values.length], 3, values.length * 9);
        for (Tensor x : new Tensor[]{base.slice(0, 1), base.transpose(), Tensor.random(32L, -20.0, 20.0, 4099)}) {
            for (UnaryOp op : UnaryOp.values()) {
                Tensor actual = switch (op) {
                    case NEGATE -> x.negate(); case SQUARE -> x.square(); case SIGMOID -> x.sigmoid();
                    case TANH -> x.tanh(); case LOG -> x.log();
                };
                double[] input = x.toArray(), output = actual.toArray();
                for (int i = 0; i < input.length; i++) {
                    double expected = op.applyAsDouble(input[i]);
                    if (op == UnaryOp.NEGATE || op == UnaryOp.SQUARE || !Double.isFinite(expected) || expected == 0)
                        exact(expected, output[i]);
                    else assertEquals(expected, output[i], 8 * Math.ulp(expected));
                }
            }
            assertArrayEquals(x.map(v -> Math.pow(v, 2)).toArray(), x.pow(2).toArray());
        }
    }

    @Test void softmaxPreservesScalarEvaluationIncludingNonfiniteRows() {
        for (Tensor x : new Tensor[]{Tensor.random(41L, -1000.0, 1000.0, 3, 17),
                Tensor.random(42L, 5, 17).transpose(),
                Tensor.of(Double.NaN, 1, 2, Double.POSITIVE_INFINITY, 0, 1, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY).reshape(3, 3)}) {
            Tensor actual = x.softmax(-1);
            for (int row = 0; row < x.shape()[0]; row++) {
                double max = Double.NEGATIVE_INFINITY, sum = 0;
                for (int j = 0; j < x.shape()[1]; j++) max = Math.max(max, x.get(row, j));
                for (int j = 0; j < x.shape()[1]; j++) sum += Math.exp(x.get(row, j) - max);
                for (int j = 0; j < x.shape()[1]; j++) exact(Math.exp(x.get(row, j) - max) / sum, actual.get(row, j));
            }
        }
    }
}
