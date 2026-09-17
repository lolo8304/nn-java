package ch.lolo.tensor;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

/** Loaded only when the VECTOR backend is selected; no vector types cross this boundary. */
final class VectorKernels {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private VectorKernels() {}

    private static DoubleVector apply(BinaryOp op, DoubleVector a, DoubleVector b) {
        return switch (op) {
            case ADD -> a.add(b);
            case SUBTRACT -> a.sub(b);
            case MULTIPLY -> a.mul(b);
            case DIVIDE -> a.div(b);
        };
    }

    static void binary(double[] a, int ao, double[] b, int bo,
                       double[] out, int oo, int length, BinaryOp op) {
        int i = 0, bound = SPECIES.loopBound(length);
        for (; i < bound; i += SPECIES.length()) {
            var av = DoubleVector.fromArray(SPECIES, a, ao + i);
            var bv = DoubleVector.fromArray(SPECIES, b, bo + i);
            apply(op, av, bv).intoArray(out, oo + i);
        }
        for (; i < length; i++) out[oo + i] = op.applyAsDouble(a[ao + i], b[bo + i]);
    }

    static void scalar(double[] a, int offset, double value, boolean scalarFirst,
                       double[] out, int outputOffset, int length, BinaryOp op) {
        int i = 0, bound = SPECIES.loopBound(length);
        var scalar = DoubleVector.broadcast(SPECIES, value);
        for (; i < bound; i += SPECIES.length()) {
            var av = DoubleVector.fromArray(SPECIES, a, offset + i);
            (scalarFirst ? apply(op, scalar, av) : apply(op, av, scalar)).intoArray(out, outputOffset + i);
        }
        for (; i < length; i++)
            out[outputOffset + i] = scalarFirst ? op.applyAsDouble(value, a[offset + i]) : op.applyAsDouble(a[offset + i], value);
    }

    static void relu(double[] a, int offset, double[] out, int outputOffset, int length) {
        int i = 0, bound = SPECIES.loopBound(length);
        for (; i < bound; i += SPECIES.length())
            DoubleVector.fromArray(SPECIES, a, offset + i).max(0.0).intoArray(out, outputOffset + i);
        for (; i < length; i++) out[outputOffset + i] = Math.max(0.0, a[offset + i]);
    }

    // SIMD across columns, preserving each element's reduction order and avoiding FMA.
    static void matmul(double[] a, int ao, int rowStride, int innerStride,
                       double[] b, int bo, int rightRowStride,
                       double[] out, int m, int k, int n) {
        int bound = SPECIES.loopBound(n);
        for (int row = 0; row < m; row++) {
            int left = ao + row * rowStride, output = row * n;
            for (int q = 0; q < k; q++) {
                double value = a[left + q * innerStride];
                int right = bo + q * rightRowStride;
                var av = DoubleVector.broadcast(SPECIES, value);
                int col = 0;
                for (; col < bound; col += SPECIES.length()) {
                    var bv = DoubleVector.fromArray(SPECIES, b, right + col);
                    var sum = DoubleVector.fromArray(SPECIES, out, output + col);
                    sum.add(av.mul(bv)).intoArray(out, output + col);
                }
                for (; col < n; col++) out[output + col] += value * b[right + col];
            }
        }
    }
}
