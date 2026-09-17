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

    // Return the vectorized prefix length; Tensor handles the scalar tail.
    static int sgd(double[] p, int po, double[] g, int go, double[] v, int vo,
                   int length, double lr, double momentum) {
        int bound = SPECIES.loopBound(length);
        for (int i = 0; i < bound; i += SPECIES.length()) {
            var grad = DoubleVector.fromArray(SPECIES, g, go + i);
            var parameter = DoubleVector.fromArray(SPECIES, p, po + i);
            var next = DoubleVector.fromArray(SPECIES, v, vo + i).mul(momentum).sub(grad.mul(lr));
            next.intoArray(v, vo + i);
            parameter.add(next).intoArray(p, po + i);
        }
        return bound;
    }

    static int adam(double[] p, int po, double[] g, int go, double[] m, int mo, double[] v, int vo,
                    int length, double lr, double b1, double b2, double c1, double c2, double eps) {
        int bound = SPECIES.loopBound(length);
        for (int i = 0; i < bound; i += SPECIES.length()) {
            var grad = DoubleVector.fromArray(SPECIES, g, go + i);
            var parameter = DoubleVector.fromArray(SPECIES, p, po + i);
            var first = DoubleVector.fromArray(SPECIES, m, mo + i).mul(b1).add(grad.mul(1 - b1));
            var second = DoubleVector.fromArray(SPECIES, v, vo + i).mul(b2).add(grad.mul(grad).mul(1 - b2));
            var delta = first.div(c1).mul(-lr).div(second.div(c2).sqrt().add(eps));
            first.intoArray(m, mo + i);
            second.intoArray(v, vo + i);
            parameter.add(delta).intoArray(p, po + i);
        }
        return bound;
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
        matmul(a, ao, rowStride, innerStride, b, bo, rightRowStride, out, 0, n, m, k, n);
    }

    private static void matmul(double[] a, int ao, int rowStride, int innerStride,
                               double[] b, int bo, int rightRowStride,
                               double[] out, int outputOffset, int outputRowStride, int m, int k, int n) {
        int bound = SPECIES.loopBound(n);
        for (int row = 0; row < m; row++) {
            int left = ao + row * rowStride, output = outputOffset + row * outputRowStride;
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

    // Pack at most 64 output columns, reusing only this call's scratch panel.
    // Read the current values on every invocation: tensor views/weights are mutable.
    static void matmulPackedRight(double[] a, int ao, int rowStride, int innerStride,
                                  double[] b, int bo, int rightRowStride, int rightColumnStride,
                                  double[] out, int m, int k, int n) {
        if (m == 0 || k == 0 || n == 0) return;
        int width = Math.min(n, 64);
        double[] panel = new double[k * width];
        for (int start = 0; start < n; start += width) {
            int columns = Math.min(width, n - start);
            for (int q = 0; q < k; q++) {
                int right = bo + q * rightRowStride + start * rightColumnStride;
                for (int col = 0; col < columns; col++)
                    panel[q * columns + col] = b[right + col * rightColumnStride];
            }
            matmul(a, ao, rowStride, innerStride, panel, 0, columns,
                    out, start, n, m, k, columns);
        }
    }

}
