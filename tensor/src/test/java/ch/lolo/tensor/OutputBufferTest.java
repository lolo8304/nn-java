package ch.lolo.tensor;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class OutputBufferTest {
    @Test void separateOffsetAndStridedDestinationsPreserveSurroundingStorage() {
        for (int n : new int[]{0, 1, 3, 17, 65}) {
            Tensor a = Tensor.random(1L, 3, n), b = Tensor.random(2L, n);
            Tensor storage = Tensor.ones(2, 3, n).multiply(-99);
            Tensor out = storage.slice(0, 1);
            assertSame(out, a.addInto(b, out));
            assertArrayEquals(a.add(b).toArray(), out.toArray());
            for (double x : storage.slice(0, 0).toArray()) assertEquals(-99, x);
            a.subtractInto(b, out); assertArrayEquals(a.subtract(b).toArray(), out.toArray());
            a.multiplyInto(b, out); assertArrayEquals(a.multiply(b).toArray(), out.toArray());
            a.divideInto(b, out); assertArrayEquals(a.divide(b).toArray(), out.toArray());
            Tensor strided = Tensor.zeros(n, 3).transpose();
            assertSame(strided, a.copyInto(strided));
            assertArrayEquals(a.toArray(), strided.toArray());
            a.addInto(b, strided); assertArrayEquals(a.add(b).toArray(), strided.toArray());
            a.multiplyInto(2, strided); assertArrayEquals(a.multiply(2).toArray(), strided.toArray());
            a.reluInto(strided); assertArrayEquals(a.relu().toArray(), strided.toArray());
        }
    }

    @Test void exactInPlaceAndRightOperandDestinations() {
        Tensor a = Tensor.random(1L, 3, 17), b = Tensor.random(2L, 3, 17);
        Tensor expected = a.subtract(b);
        assertSame(b, a.subtractInto(b, b));
        assertArrayEquals(expected.toArray(), b.toArray());
        expected = a.multiply(a);
        a.multiplyInto(a, a); assertArrayEquals(expected.toArray(), a.toArray());
        expected = a.add(2); a.addInto(2, a); assertArrayEquals(expected.toArray(), a.toArray());
        expected = a.subtract(3); a.subtractInto(3, a); assertArrayEquals(expected.toArray(), a.toArray());
        expected = a.divide(2); a.divideInto(2, a); assertArrayEquals(expected.toArray(), a.toArray());
        expected = a.relu(); a.reluInto(a); assertArrayEquals(expected.toArray(), a.toArray());
    }

    @Test void overlappingTransposeAndBothOperandsUseOriginalValues() {
        Tensor base = Tensor.generate(i -> 1 + i[0] * 3 + i[1], 3, 3);
        Tensor expected = base.transpose().subtract(base);
        base.transpose().subtractInto(base, base);
        assertArrayEquals(expected.toArray(), base.toArray());
        expected = base.transpose().copy();
        base.transpose().copyInto(base);
        assertArrayEquals(expected.toArray(), base.toArray());
        expected = base.transpose().multiply(2);
        base.transpose().multiplyInto(2, base);
        assertArrayEquals(expected.toArray(), base.toArray());
        expected = base.transpose().relu();
        base.transpose().reluInto(base);
        assertArrayEquals(expected.toArray(), base.toArray());
    }

    @Test void overlappingRowColumnAndBroadcastInputsAreSnapshotted() {
        Tensor base = Tensor.generate(i -> 1 + i[0] * 3 + i[1], 3, 3);
        Tensor column = base.slice(1, 0), row = base.slice(0, 1);
        double[] expected = column.toArray();
        column.copyInto(row); assertArrayEquals(expected, row.toArray());
        Tensor bias = base.slice(0, 0);
        Tensor result = base.add(bias);
        base.addInto(bias, base); assertArrayEquals(result.toArray(), base.toArray());
        Tensor scalarView = base.slice(0, 0).slice(0, 0);
        result = scalarView.divide(base);
        scalarView.divideInto(base, base); assertArrayEquals(result.toArray(), base.toArray());
    }

    @Test void specialValuesScalarAndEmptyResults() {
        Tensor a = Tensor.of(Double.NaN, -0.0, 0.0, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
        Tensor out = Tensor.zeros(2, 5).slice(0, 1);
        a.reluInto(out); assertArrayEquals(a.relu().toArray(), out.toArray());
        a.divideInto(0, out); assertArrayEquals(a.divide(0).toArray(), out.toArray());
        Tensor scalar = Tensor.scalar(3);
        scalar.addInto(4, scalar); assertEquals(7, scalar.scalar());
        Tensor empty = Tensor.zeros(0, 3);
        assertSame(empty, empty.addInto(Tensor.ones(1, 3), empty));
        assertEquals(0, empty.size());
    }

    @Test void invalidShapesDoNotPartiallyWrite() {
        Tensor a = Tensor.ones(2, 3), out = Tensor.ones(3, 2).multiply(99);
        double[] before = out.toArray();
        assertThrows(IllegalArgumentException.class, () -> a.addInto(a, out));
        assertThrows(IllegalArgumentException.class, () -> a.multiplyInto(2, out));
        assertThrows(IllegalArgumentException.class, () -> a.copyInto(out));
        assertThrows(IllegalArgumentException.class, () -> a.reluInto(out));
        assertThrows(IllegalArgumentException.class, () -> a.addInto(Tensor.ones(7), out));
        assertArrayEquals(before, out.toArray());
        assertThrows(NullPointerException.class, () -> a.copyInto(null));
    }
}
