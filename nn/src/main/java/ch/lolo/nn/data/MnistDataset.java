package ch.lolo.nn.data;

import ch.lolo.tensor.Tensor;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public final class MnistDataset implements Dataset {
    private final byte[] images, labels;
    private final int count, rows, cols;

    public MnistDataset(Path imageFile, Path labelFile) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(imageFile))); DataInputStream la = new DataInputStream(new BufferedInputStream(Files.newInputStream(labelFile)))) {
            if (in.readInt() != 2051 || la.readInt() != 2049) throw new IOException("invalid MNIST IDX magic");
            count = in.readInt();
            int lc = la.readInt();
            if (count != lc) throw new IOException("count mismatch");
            rows = in.readInt();
            cols = in.readInt();
            images = in.readNBytes(count * rows * cols);
            labels = la.readNBytes(count);
        }
    }

    public int size() {
        return count;
    }

    public Sample get(int i) {
        int off = i * rows * cols;
        Tensor x = Tensor.generate(p -> (images[off + p[0]] & 255) / 255.0, rows * cols);
        Tensor y = Tensor.zeros(10);
        y.set(1, labels[i] & 255);
        return new Sample(x, y);
    }

    public int rows() {
        return rows;
    }

    public int cols() {
        return cols;
    }
}
