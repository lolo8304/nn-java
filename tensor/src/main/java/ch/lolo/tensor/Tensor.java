package ch.lolo.tensor;

import java.util.Arrays;
import java.util.Random;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

/**
 * Pure-Java arbitrary-rank double tensor. Slice/transpose are mutable shared views.
 */
public final class Tensor {
    private final double[] data;
    private final int[] shape, strides;
    private final int offset;
    private Tensor(double[] d, int[] s, int[] st, int o) {
        data = d;
        shape = s.clone();
        strides = st.clone();
        offset = o;
    }

    private static Tensor alloc(int... s) {
        int[] x = s.clone();
        return new Tensor(new double[TensorShape.sizeOf(x)], x, TensorShape.strides(x), 0);
    }

    public static Tensor scalar(double x) {
        Tensor t = alloc();
        t.data[0] = x;
        return t;
    }

    public static Tensor zeros(int... s) {
        return alloc(s);
    }

    public static Tensor ones(int... s) {
        Tensor t = alloc(s);
        Arrays.fill(t.data, 1);
        return t;
    }

    public static Tensor random(int... s) {
        return random(new Random(), 0, 1, s);
    }

    public static Tensor random(long seed, int... s) {
        return random(new Random(seed), 0, 1, s);
    }

    public static Tensor random(double min, double max, int... s) {
        return random(new Random(), min, max, s);
    }

    public static Tensor random(long seed, double min, double max, int... s) {
        return random(new Random(seed), min, max, s);
    }

    public static Tensor random(Random r, double min, double max, int... s) {
        if (max <= min) throw new IllegalArgumentException("max <= min");
        Tensor t = alloc(s);
        for (int i = 0; i < t.data.length; i++) t.data[i] = min + r.nextDouble() * (max - min);
        return t;
    }

    public static Tensor generate(Generator g, int... s) {
        Tensor t = alloc(s);
        TensorIterator.each(t.shape, i -> t.set(g.apply(i), i));
        return t;
    }

    public static Tensor of(double... a) {
        return new Tensor(a.clone(), new int[]{a.length}, new int[]{1}, 0);
    }

    public static Tensor of(double[][] a) {
        int r = a.length, c = r == 0 ? 0 : a[0].length;
        Tensor t = alloc(r, c);
        for (int i = 0; i < r; i++) {
            if (a[i].length != c) throw new IllegalArgumentException("ragged");
            for (int j = 0; j < c; j++) t.set(a[i][j], i, j);
        }
        return t;
    }

    public static Tensor of(double[][][] a) {
        int x = a.length, y = x == 0 ? 0 : a[0].length, z = y == 0 ? 0 : a[0][0].length;
        Tensor t = alloc(x, y, z);
        for (int i = 0; i < x; i++) {
            if (a[i].length != y) throw new IllegalArgumentException("ragged");
            for (int j = 0; j < y; j++) {
                if (a[i][j].length != z) throw new IllegalArgumentException("ragged");
                for (int k = 0; k < z; k++) t.set(a[i][j][k], i, j, k);
            }
        }
        return t;
    }

    private static int[] remove(int[] x, int a) {
        int[] r = new int[x.length - 1];
        for (int i = 0, j = 0; i < x.length; i++) if (i != a) r[j++] = x[i];
        return r;
    }

    private static int[] insert(int[] x, int a, int v) {
        int[] r = new int[x.length + 1];
        for (int i = 0, j = 0; i < r.length; i++) r[i] = i == a ? v : x[j++];
        return r;
    }

    private static int[] infer(int[] requested, int size) {
        int[] r = requested.clone();
        int neg = -1;
        long known = 1;
        for (int i = 0; i < r.length; i++) {
            if (r[i] == -1) {
                if (neg >= 0) throw new IllegalArgumentException("only one -1");
                neg = i;
            } else {
                if (r[i] < 0) throw new IllegalArgumentException("bad dimension");
                known *= r[i];
            }
        }
        if (neg >= 0) {
            if (known == 0 || size % known != 0) throw new IllegalArgumentException("cannot infer");
            r[neg] = (int) (size / known);
        }
        if (TensorShape.sizeOf(r) != size) throw new IllegalArgumentException("size mismatch");
        return r;
    }

    public int rank() {
        return shape.length;
    }

    public int[] shape() {
        return shape.clone();
    }

    public int[] strides() {
        return strides.clone();
    }

    public long size() {
        return TensorShape.sizeOf(shape);
    }

    public boolean isScalar() {
        return rank() == 0;
    }

    public boolean isContiguous() {
        return TensorShape.contiguous(shape, strides);
    }

    private int addr(int... i) {
        if (i.length != rank()) throw new IllegalArgumentException("expected " + rank() + " indices");
        int p = offset;
        for (int a = 0; a < i.length; a++) {
            if (i[a] < 0 || i[a] >= shape[a]) throw new IndexOutOfBoundsException();
            p += i[a] * strides[a];
        }
        return p;
    }

    public double get(int... i) {
        return data[addr(i)];
    }

    public Tensor set(double v, int... i) {
        data[addr(i)] = v;
        return this;
    }

    public double scalar() {
        if (!isScalar()) throw new IllegalStateException("not scalar");
        return data[offset];
    }

    private int axis(int a) {
        if (a < 0) a += rank();
        if (a < 0 || a >= rank()) throw new IllegalArgumentException("bad axis");
        return a;
    }

    public Tensor slice(int a, int n) {
        a = axis(a);
        if (n < 0 || n >= shape[a]) throw new IndexOutOfBoundsException();
        int[] s = new int[rank() - 1], st = new int[rank() - 1];
        for (int i = 0, j = 0; i < rank(); i++)
            if (i != a) {
                s[j] = shape[i];
                st[j++] = strides[i];
            }
        return new Tensor(data, s, st, offset + n * strides[a]);
    }

    public Tensor transpose() {
        int[] a = new int[rank()];
        for (int i = 0; i < rank(); i++) a[i] = rank() - 1 - i;
        return transpose(a);
    }

    public Tensor transpose(int... a) {
        if (a.length != rank()) throw new IllegalArgumentException();
        boolean[] seen = new boolean[rank()];
        int[] s = new int[rank()], st = new int[rank()];
        for (int i = 0; i < a.length; i++) {
            int x = axis(a[i]);
            if (seen[x]) throw new IllegalArgumentException("duplicate axis");
            seen[x] = true;
            s[i] = shape[x];
            st[i] = strides[x];
        }
        return new Tensor(data, s, st, offset);
    }

    public Tensor reshape(int... ns) {
        ns = infer(ns, (int) size());
        if (isContiguous()) return new Tensor(data, ns, TensorShape.strides(ns), offset);
        return copy().reshape(ns);
    }

    public Tensor flatten() {
        return reshape((int) size());
    }

    public Tensor copy() {
        Tensor t = alloc(shape);
        if (isContiguous()) System.arraycopy(data, offset, t.data, 0, t.data.length);
        else TensorIterator.each(shape, i -> t.set(get(i), i));
        return t;
    }

    /** Returns an elementwise transformation in new, independent storage. */
    public Tensor map(DoubleUnaryOperator f) {
        Tensor t = alloc(shape);
        if (isContiguous()) {
            for (int i = 0; i < t.data.length; i++) t.data[i] = f.applyAsDouble(data[offset + i]);
        } else TensorIterator.each(shape, i -> t.set(f.applyAsDouble(get(i)), i));
        return t;
    }

    private int broadcastAddress(int[] index) {
        int address = offset, shift = index.length - rank();
        for (int a = 0; a < rank(); a++)
            if (shape[a] != 1) address += index[a + shift] * strides[a];
        return address;
    }

    private Tensor bin(Tensor b, DoubleBinaryOperator f) {
        int[] s = TensorShape.broadcast(shape, b.shape);
        if (b.isScalar()) return map(x -> f.applyAsDouble(x, b.scalar()));
        if (isScalar()) return b.map(x -> f.applyAsDouble(scalar(), x));
        Tensor t = alloc(s);
        if (Arrays.equals(shape, b.shape) && isContiguous() && b.isContiguous()) {
            for (int i = 0; i < t.data.length; i++)
                t.data[i] = f.applyAsDouble(data[offset + i], b.data[b.offset + i]);
        } else {
            int[] index = new int[s.length];
            for (int i = 0; i < t.data.length; i++) {
                t.data[i] = f.applyAsDouble(data[broadcastAddress(index)], b.data[b.broadcastAddress(index)]);
                for (int a = s.length - 1; a >= 0; a--) {
                    if (++index[a] < s[a]) break;
                    index[a] = 0;
                }
            }
        }
        return t;
    }

    public Tensor add(Tensor b) {
        return bin(b, (x, y) -> x + y);
    }

    public Tensor subtract(Tensor b) {
        return bin(b, (x, y) -> x - y);
    }

    public Tensor multiply(Tensor b) {
        return bin(b, (x, y) -> x * y);
    }

    public Tensor divide(Tensor b) {
        return bin(b, (x, y) -> x / y);
    }

    public Tensor add(double x) {
        return map(v -> v + x);
    }

    public Tensor subtract(double x) {
        return map(v -> v - x);
    }

    public Tensor multiply(double x) {
        return map(v -> v * x);
    }

    public Tensor divide(double x) {
        return map(v -> v / x);
    }

    public Tensor negate() {
        return map(v -> -v);
    }

    public Tensor pow(double p) {
        return map(v -> Math.pow(v, p));
    }

    public Tensor relu() {
        return map(v -> Math.max(0, v));
    }

    public Tensor sigmoid() {
        return map(v -> 1 / (1 + Math.exp(-v)));
    }

    public Tensor matmul(Tensor b) {
        if (rank() == 0 || b.rank() == 0) throw new IllegalArgumentException("scalar matmul");
        boolean av = rank() == 1, bv = b.rank() == 1;
        int m = av ? 1 : shape[rank() - 2], k = shape[rank() - 1], bk = bv ? b.shape[0] : b.shape[b.rank() - 2], n = bv ? 1 : b.shape[b.rank() - 1];
        if (k != bk) throw new IllegalArgumentException("inner dimensions differ");
        if (rank() == 2 && b.rank() == 2) return matmul2d(b, m, k, n);
        int[] ab = av ? new int[0] : Arrays.copyOf(shape, rank() - 2), bb = bv ? new int[0] : Arrays.copyOf(b.shape, b.rank() - 2), batch = TensorShape.broadcast(ab, bb);
        if (av && bv) {
            double s = 0;
            for (int q = 0; q < k; q++) s += get(q) * b.get(q);
            return scalar(s);
        }
        int[] os = new int[batch.length + (av ? 0 : 1) + (bv ? 0 : 1)];
        int p = 0;
        for (int x : batch) os[p++] = x;
        if (!av) os[p++] = m;
        if (!bv) os[p] = n;
        Tensor out = alloc(os);
        TensorIterator.each(os, oi -> {
            int[] bi = Arrays.copyOf(oi, batch.length);
            int row = av ? 0 : oi[batch.length], col = bv ? 0 : oi[oi.length - 1];
            double s = 0;
            for (int q = 0; q < k; q++) s += matGet(bi, row, q, av) * (bv ? b.get(q) : b.matGet(bi, q, col, false));
            out.set(s, oi);
        });
        return out;
    }

    // Direct stride arithmetic also supports transposed and sliced views used in backpropagation.
    private Tensor matmul2d(Tensor b, int m, int k, int n) {
        Tensor out = alloc(m, n);
        for (int row = 0; row < m; row++) {
            int left = offset + row * strides[0];
            int output = row * n;
            if (b.strides[1] == 1) {
                // Traverse contiguous right-hand rows, keeping each output's summation order.
                for (int q = 0; q < k; q++) {
                    double value = data[left + q * strides[1]];
                    int right = b.offset + q * b.strides[0];
                    for (int col = 0; col < n; col++) {
                        out.data[output + col] += value * b.data[right + col];
                    }
                }
            } else {
                for (int col = 0; col < n; col++) {
                    double sum = 0;
                    int right = b.offset + col * b.strides[1];
                    for (int q = 0; q < k; q++) {
                        sum += data[left + q * strides[1]] * b.data[right + q * b.strides[0]];
                    }
                    out.data[output + col] = sum;
                }
            }
        }
        return out;
    }

    private double matGet(int[] batch, int row, int col, boolean vec) {
        if (vec) return data[offset + col * strides[0]];
        int br = rank() - 2, shift = batch.length - br, address = offset;
        for (int x = 0; x < br; x++)
            if (shape[x] != 1) address += batch[x + shift] * strides[x];
        return data[address + row * strides[br] + col * strides[br + 1]];
    }

    public Tensor sum() {
        double[] s = {0};
        if (isContiguous()) {
            int end = offset + (int) size();
            for (int i = offset; i < end; i++) s[0] += data[i];
        } else TensorIterator.each(shape, i -> s[0] += get(i));
        return scalar(s[0]);
    }

    public Tensor mean() {
        return scalar(size() == 0 ? Double.NaN : sum().scalar() / size());
    }

    public Tensor min() {
        if (size() == 0) throw new IllegalStateException("empty");
        double[] x = {Double.POSITIVE_INFINITY};
        TensorIterator.each(shape, i -> x[0] = Math.min(x[0], get(i)));
        return scalar(x[0]);
    }

    public Tensor max() {
        if (size() == 0) throw new IllegalStateException("empty");
        double[] x = {Double.NEGATIVE_INFINITY};
        TensorIterator.each(shape, i -> x[0] = Math.max(x[0], get(i)));
        return scalar(x[0]);
    }

    public Tensor sum(int a) {
        a = axis(a);
        int[] os = remove(shape, a);
        Tensor t = zeros(os);
        int ax = a;
        int[] position = {0};
        TensorIterator.each(os, o -> {
            int base = offset;
            for (int d = 0, j = 0; d < rank(); d++)
                if (d != ax) base += o[j++] * strides[d];
            double sum = 0;
            for (int k = 0; k < shape[ax]; k++) sum += data[base + k * strides[ax]];
            t.data[position[0]++] = sum;
        });
        return t;
    }

    public Tensor mean(int a) {
        int ax = axis(a);
        return sum(ax).divide(shape[ax]);
    }

    public Tensor min(int a) {
        return extrema(a, true);
    }

    public Tensor max(int a) {
        return extrema(a, false);
    }

    private Tensor extrema(int a, boolean min) {
        a = axis(a);
        int[] os = remove(shape, a);
        Tensor t = alloc(os);
        Arrays.fill(t.data, min ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY);
        int ax = a;
        TensorIterator.each(shape, i -> {
            int[] o = remove(i, ax);
            double v = get(i);
            t.set(min ? Math.min(t.get(o), v) : Math.max(t.get(o), v), o);
        });
        return t;
    }

    public Tensor softmax(int a) {
        a = axis(a);
        Tensor out = alloc(shape);
        int ax = a;
        int[] outer = remove(shape, ax);
        TensorIterator.each(outer, o -> {
            int source = offset, target = 0;
            for (int d = 0, j = 0; d < rank(); d++) {
                if (d == ax) continue;
                source += o[j] * strides[d];
                target += o[j++] * out.strides[d];
            }
            double max = Double.NEGATIVE_INFINITY;
            for (int k = 0; k < shape[ax]; k++) max = Math.max(max, data[source + k * strides[ax]]);
            double sum = 0;
            for (int k = 0; k < shape[ax]; k++) {
                double e = Math.exp(data[source + k * strides[ax]] - max);
                out.data[target + k * out.strides[ax]] = e;
                sum += e;
            }
            for (int k = 0; k < shape[ax]; k++) out.data[target + k * out.strides[ax]] /= sum;
        });
        return out;
    }

    public Tensor sort() {
        double[] x = flatten().toArray();
        Arrays.sort(x);
        return Tensor.of(x);
    }

    public Tensor sort(int a) {
        a = axis(a);
        Tensor out = copy();
        int ax = a;
        int[] outer = remove(shape, ax);
        TensorIterator.each(outer, o -> {
            double[] x = new double[shape[ax]];
            for (int k = 0; k < x.length; k++) x[k] = get(insert(o, ax, k));
            Arrays.sort(x);
            for (int k = 0; k < x.length; k++) out.set(x[k], insert(o, ax, k));
        });
        return out;
    }

    public double[] toArray() {
        double[] x = new double[(int) size()];
        if (isContiguous()) System.arraycopy(data, offset, x, 0, x.length);
        else {
            int[] p = {0};
            TensorIterator.each(shape, i -> x[p[0]++] = get(i));
        }
        return x;
    }

    @Override
    public String toString() {
        return "Tensor(shape=" + Arrays.toString(shape) + ", values=" + Arrays.toString(toArray()) + ")";
    }

    @FunctionalInterface
    public interface Generator {
        double apply(int... index);
    }
}
