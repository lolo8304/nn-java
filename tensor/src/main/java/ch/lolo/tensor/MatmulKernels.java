package ch.lolo.tensor;

/** Cache tiling shared by the Java and Vector backends. */
final class MatmulKernels {
    private static final int ROWS = 32;
    private static final int INNER = 64;
    private static final int COLUMNS = 64;

    private MatmulKernels() {}

    // Leave small layers and single-example inference on their existing kernels.
    static boolean shouldBlock(int m, int k, int n) {
        return m >= 16 && k >= 128 && n >= 32 && (long) k * n >= 32_768;
    }

    // Every output still accumulates q=0..k-1 with separate multiply and add.
    static void matmul(double[] a, int ao, int rowStride, int innerStride,
                       double[] b, int bo, int rightRowStride, int rightColumnStride,
                       double[] out, int m, int k, int n, boolean vector) {
        matmul(a, ao, rowStride, innerStride, b, bo, rightRowStride, rightColumnStride, out, 0, m, k, n, vector);
    }

    static void matmul(double[] a, int ao, int rowStride, int innerStride,
                       double[] b, int bo, int rightRowStride, int rightColumnStride,
                       double[] out, int outputOffset, int m, int k, int n, boolean vector) {
        if (rightColumnStride != 1) {
            // Bounded scratch, freshly populated for mutable transpose/slice views.
            double[] panel = new double[INNER * COLUMNS];
            for (int col = 0; col < n; col += COLUMNS) {
                int columns = Math.min(COLUMNS, n - col);
                for (int q = 0; q < k; q += INNER) {
                    int depth = Math.min(INNER, k - q);
                    for (int p = 0; p < depth; p++) {
                        int right = bo + (q + p) * rightRowStride + col * rightColumnStride;
                        for (int j = 0; j < columns; j++)
                            panel[p * columns + j] = b[right + j * rightColumnStride];
                    }
                    // Reuse the packed panel across all rows before replacing it.
                    for (int row = 0; row < m; row += ROWS)
                        tile(a, ao + row * rowStride + q * innerStride, rowStride, innerStride,
                                panel, 0, columns, out, outputOffset + row * n + col, n,
                                Math.min(ROWS, m - row), depth, columns, vector);
                }
            }
        } else {
            int width = 512;
            for (int row = 0; row < m; row += ROWS)
                for (int col = 0; col < n; col += width)
                    for (int q = 0; q < k; q += INNER)
                        tile(a, ao + row * rowStride + q * innerStride, rowStride, innerStride,
                                b, bo + q * rightRowStride + col, rightRowStride,
                                out, outputOffset + row * n + col, n, Math.min(ROWS, m - row),
                                Math.min(INNER, k - q), Math.min(width, n - col), vector);
        }
    }

    private static void tile(double[] a, int ao, int rowStride, int innerStride,
                             double[] b, int bo, int rightRowStride,
                             double[] out, int outputOffset, int outputRowStride,
                             int m, int k, int n, boolean vector) {
        if (vector) {
            VectorKernels.matmul(a, ao, rowStride, innerStride, b, bo, rightRowStride,
                    out, outputOffset, outputRowStride, m, k, n);
            return;
        }
        for (int row = 0; row < m; row++) {
            int left = ao + row * rowStride, output = outputOffset + row * outputRowStride;
            for (int q = 0; q < k; q++) {
                double value = a[left + q * innerStride];
                int right = bo + q * rightRowStride;
                for (int col = 0; col < n; col++) out[output + col] += value * b[right + col];
            }
        }
    }
}
