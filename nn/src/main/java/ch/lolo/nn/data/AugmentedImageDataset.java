package ch.lolo.nn.data;

import ch.lolo.tensor.Tensor;

import java.util.Objects;
import java.util.Random;

/** Training-only image view with fresh affine augmentation on each get(). Not thread-safe. */
public final class AugmentedImageDataset implements Dataset {
    private final Dataset source;
    private final int rows, cols;
    private final double maxShift, minScale, maxScale, maxRotationDegrees;
    private final Random random;

    /** Images must be grayscale, normalized to [0,1], flattened or shaped [rows,cols]. */
    public AugmentedImageDataset(Dataset source, int rows, int cols, double maxShift,
                                 double minScale, double maxScale, double maxRotationDegrees, long seed) {
        this.source = Objects.requireNonNull(source);
        if (rows < 1 || cols < 1 || !Double.isFinite(maxShift) || maxShift < 0
                || !Double.isFinite(minScale) || !Double.isFinite(maxScale)
                || minScale <= 0 || maxScale < minScale
                || !Double.isFinite(maxRotationDegrees) || maxRotationDegrees < 0 || maxRotationDegrees > 180) {
            throw new IllegalArgumentException("Invalid image augmentation settings");
        }
        this.rows = rows;
        this.cols = cols;
        this.maxShift = maxShift;
        this.minScale = minScale;
        this.maxScale = maxScale;
        this.maxRotationDegrees = maxRotationDegrees;
        random = new Random(seed);
    }

    public int size() { return source.size(); }

    public Sample get(int index) {
        var sample = source.get(index);
        var shape = sample.input().shape();
        if (!((shape.length == 1 && shape[0] == rows * cols)
                || (shape.length == 2 && shape[0] == rows && shape[1] == cols))) {
            throw new IllegalArgumentException("Image shape does not match augmentation dimensions");
        }
        double[] pixels = sample.input().toArray();
        int minX = cols, minY = rows, maxX = -1, maxY = -1;
        for (int i = 0; i < pixels.length; i++) {
            if (!Double.isFinite(pixels[i]) || pixels[i] < 0 || pixels[i] > 1)
                throw new IllegalArgumentException("Image pixels must be in [0,1]");
            if (pixels[i] > 0) {
                minX = Math.min(minX, i % cols); maxX = Math.max(maxX, i % cols);
                minY = Math.min(minY, i / cols); maxY = Math.max(maxY, i / cols);
            }
        }
        if (maxX < 0) return sample;
        double cx = (cols - 1) / 2.0, cy = (rows - 1) / 2.0;
        // Reject transformations that would cut off the digit; fall back to the original.
        for (int attempt = 0; attempt < 12; attempt++) {
            double scale = minScale + random.nextDouble() * (maxScale - minScale);
            double angle = Math.toRadians((random.nextDouble() * 2 - 1) * maxRotationDegrees);
            double dx = (random.nextDouble() * 2 - 1) * maxShift;
            double dy = (random.nextDouble() * 2 - 1) * maxShift;
            double cos = Math.cos(angle), sin = Math.sin(angle);
            boolean fits = true;
            for (double y : new double[]{minY - .5, maxY + .5}) {
                for (double x : new double[]{minX - .5, maxX + .5}) {
                    double tx = cx + dx + scale * (cos * (x - cx) - sin * (y - cy));
                    double ty = cy + dy + scale * (sin * (x - cx) + cos * (y - cy));
                    if (tx < -.5 || tx > cols - .5 || ty < -.5 || ty > rows - .5) fits = false;
                }
            }
            if (!fits) continue;
            double[] result = new double[pixels.length];
            for (int y = 0; y < rows; y++) for (int x = 0; x < cols; x++) {
                double tx = (x - cx - dx) / scale, ty = (y - cy - dy) / scale;
                result[y * cols + x] = interpolate(pixels, cx + cos * tx + sin * ty, cy - sin * tx + cos * ty);
            }
            return new Sample(Tensor.of(result).reshape(shape), sample.target());
        }
        return sample;
    }

    private double interpolate(double[] pixels, double x, double y) {
        int x0 = (int) Math.floor(x), y0 = (int) Math.floor(y);
        double fx = x - x0, fy = y - y0;
        double value = (1 - fy) * ((1 - fx) * pixel(pixels, x0, y0) + fx * pixel(pixels, x0 + 1, y0))
                + fy * ((1 - fx) * pixel(pixels, x0, y0 + 1) + fx * pixel(pixels, x0 + 1, y0 + 1));
        return Math.max(0, Math.min(1, value));
    }

    private double pixel(double[] pixels, int x, int y) {
        return x < 0 || x >= cols || y < 0 || y >= rows ? 0 : pixels[y * cols + x];
    }
}
