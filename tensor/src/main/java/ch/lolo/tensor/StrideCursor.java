package ch.lolo.tensor;

/** Internal row-major cursor. Never exposes or allocates per-element indices. */
final class StrideCursor {
    private final int[] shape, strides, index;
    int address;

    StrideCursor(int[] shape, int[] strides, int offset) {
        this.shape = shape;
        this.strides = strides;
        this.index = new int[shape.length];
        this.address = offset;
    }

    static StrideCursor broadcast(int[] outputShape, int[] shape, int[] strides, int offset) {
        int[] mapped = new int[outputShape.length];
        int shift = outputShape.length - shape.length;
        for (int a = 0; a < shape.length; a++)
            if (shape[a] != 1) mapped[a + shift] = strides[a];
        return new StrideCursor(outputShape, mapped, offset);
    }

    void advance() {
        for (int a = shape.length - 1; a >= 0; a--) {
            if (++index[a] < shape[a]) {
                address += strides[a];
                return;
            }
            index[a] = 0;
            address -= (shape[a] - 1) * strides[a];
        }
    }
}
