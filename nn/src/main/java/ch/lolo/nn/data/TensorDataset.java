package ch.lolo.nn.data;

import ch.lolo.tensor.Tensor;

public final class TensorDataset implements Dataset {
    private final Tensor x, y;

    public TensorDataset(Tensor x, Tensor y) {
        if (x.rank() < 1 || y.rank() < 1 || x.shape()[0] != y.shape()[0])
            throw new IllegalArgumentException("sample counts differ");
        this.x = x;
        this.y = y;
    }

    public int size() {
        return x.shape()[0];
    }

    public Sample get(int i) {
        return new Sample(x.slice(0, i), y.slice(0, i));
    }
}
