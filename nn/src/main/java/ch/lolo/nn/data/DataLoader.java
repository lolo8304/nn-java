package ch.lolo.nn.data;

import ch.lolo.nn.NN;
import ch.lolo.tensor.Tensor;

import java.util.Iterator;
import java.util.NoSuchElementException;

public final class DataLoader implements Iterable<Batch> {
    private final Dataset ds;
    private final int batch;
    private final boolean shuffle;

    public DataLoader(Dataset d, int b, boolean s) {
        if (b < 1) throw new IllegalArgumentException();
        ds = d;
        batch = b;
        shuffle = s;
    }

    private static int[] prepend(int n, int[] s) {
        int[] r = new int[s.length + 1];
        r[0] = n;
        System.arraycopy(s, 0, r, 1, s.length);
        return r;
    }

    public int batches() {
        return (ds.size() + batch - 1) / batch;
    }

    public Iterator<Batch> iterator() {
        int[] order = new int[ds.size()];
        for (int i = 0; i < order.length; i++) order[i] = i;
        if (shuffle) for (int i = order.length - 1; i > 0; i--) {
            int j = NN.random().nextInt(i + 1), z = order[i];
            order[i] = order[j];
            order[j] = z;
        }
        return new Iterator<>() {
            int pos;

            public boolean hasNext() {
                return pos < order.length;
            }

            public Batch next() {
                if (!hasNext()) throw new NoSuchElementException();
                int n = Math.min(batch, order.length - pos);
                Sample[] samples = new Sample[n];
                for (int i = 0; i < n; i++) samples[i] = ds.get(order[pos + i]);
                Sample first = samples[0];
                int[] xs = prepend(n, first.input().shape()), ys = prepend(n, first.target().shape());
                Tensor x = Tensor.zeros(xs), y = Tensor.zeros(ys);
                // Fetch all samples before copying: datasets may transform samples on get().
                // copyInto checks exact shapes and handles both contiguous and strided views.
                for (int row = 0; row < n; row++) {
                    samples[row].input().copyInto(x.slice(0, row));
                    samples[row].target().copyInto(y.slice(0, row));
                }
                pos += n;
                return new Batch(x, y);
            }
        };
    }
}
