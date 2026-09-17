package ch.lolo.tensor;

import java.util.Arrays;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;

/**
 * Pure-Java arbitrary-rank double tensor. Slice/transpose are mutable shared views.
 */
public final class Tensor {
    private static volatile TensorBackend backend = initialBackend();

    private static TensorBackend initialBackend() {
        String name = System.getProperty("tensor.backend", "vector");
        TensorBackend selected;
        try {
            selected = TensorBackend.valueOf(name.toUpperCase(java.util.Locale.ROOT));
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("tensor.backend must be java or vector: " + name, e);
        }
        checkBackend(selected);
        return selected;
    }

    private static void checkBackend(TensorBackend selected) {
        java.util.Objects.requireNonNull(selected, "backend");
        if (selected == TensorBackend.VECTOR && !isVectorAvailable())
            throw new IllegalStateException("VECTOR requires JVM option --add-modules jdk.incubator.vector");
    }

    /** Whether the Vector API module is resolved in this JVM; not a speed guarantee. */
    public static boolean isVectorAvailable() {
        return ModuleLayer.boot().findModule("jdk.incubator.vector").isPresent();
    }

    /** Current process-wide backend; defaults to VECTOR, overridable by -Dtensor.backend=java|vector. */
    public static TensorBackend backend() {
        return backend;
    }

    /**
     * Selects the backend for subsequent operations, including existing tensors and models.
     * Configure before training; do not switch while other threads are computing.
     * VECTOR requires --add-modules jdk.incubator.vector at JVM startup.
     */
    public static void setBackend(TensorBackend selected) {
        checkBackend(selected);
        backend = selected;
    }

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

    /**
     * Copies into an existing tensor of exactly the same shape and returns destination.
     * Shared views are supported: results are as if all inputs were read before any writes.
     * Noncontiguous overlapping layouts may require an input snapshot.
     */
    public Tensor copyInto(Tensor destination) {
        requireShape(destination, shape);
        if (sameLayout(destination)) return destination;
        if (isContiguous() && destination.isContiguous()) {
            System.arraycopy(data, offset, destination.data, destination.offset, (int) size());
        } else {
            Tensor source = safeInput(destination);
            int[] index = new int[rank()];
            for (int i = 0, length = (int) size(); i < length; i++) {
                destination.data[destination.broadcastAddress(index)] = source.data[source.broadcastAddress(index)];
                advance(index, shape);
            }
        }
        return destination;
    }

    private static void requireShape(Tensor destination, int[] expected) {
        java.util.Objects.requireNonNull(destination, "destination");
        if (!Arrays.equals(destination.shape, expected))
            throw new IllegalArgumentException("destination shape must be " + Arrays.toString(expected));
    }

    private boolean sameLayout(Tensor other) {
        return data == other.data && offset == other.offset
                && Arrays.equals(shape, other.shape) && Arrays.equals(strides, other.strides);
    }

    private long lastAddress() {
        long last = offset;
        for (int a = 0; a < rank(); a++) last += (long) (shape[a] - 1) * strides[a];
        return last;
    }

    // All current views have nonnegative strides. Bounding intervals may conservatively
    // overlap for interleaved slices; copying in that case is safe, though unnecessary.
    private Tensor safeInput(Tensor destination) {
        if (data != destination.data || sameLayout(destination) || size() == 0 || destination.size() == 0)
            return this;
        return offset <= destination.lastAddress() && destination.offset <= lastAddress() ? copy() : this;
    }

    private static void advance(int[] index, int[] dimensions) {
        for (int a = dimensions.length - 1; a >= 0; a--) {
            if (++index[a] < dimensions[a]) break;
            index[a] = 0;
        }
    }

    private Tensor bin(Tensor b, BinaryOp op) {
        return binaryInto(b, alloc(TensorShape.broadcast(shape, b.shape)), op);
    }

    private Tensor binaryInto(Tensor b, Tensor destination, BinaryOp op) {
        int[] resultShape = TensorShape.broadcast(shape, b.shape);
        requireShape(destination, resultShape); // Validate before mutating any shared storage.
        Tensor left = safeInput(destination), right = b.safeInput(destination);
        if (right.isScalar()) return left.scalarInto(right.scalar(), destination, op, false);
        if (left.isScalar()) return right.scalarInto(left.scalar(), destination, op, true);
        int length = (int) destination.size();
        boolean vector = backend == TensorBackend.VECTOR;
        if (Arrays.equals(left.shape, right.shape) && left.isContiguous()
                && right.isContiguous() && destination.isContiguous()) {
            if (vector) {
                VectorKernels.binary(left.data, left.offset, right.data, right.offset,
                        destination.data, destination.offset, length, op);
            } else {
                for (int i = 0; i < length; i++) destination.data[destination.offset + i] =
                        op.applyAsDouble(left.data[left.offset + i], right.data[right.offset + i]);
            }
        } else if (vector && left.rank() == 2 && right.rank() == 1 && right.shape[0] == left.shape[1]
                && left.isContiguous() && right.isContiguous() && destination.isContiguous()) {
            for (int row = 0; row < left.shape[0]; row++)
                VectorKernels.binary(left.data, left.offset + row * left.shape[1], right.data, right.offset,
                        destination.data, destination.offset + row * left.shape[1], left.shape[1], op);
        } else {
            int[] index = new int[resultShape.length];
            for (int i = 0; i < length; i++) {
                destination.data[destination.broadcastAddress(index)] = op.applyAsDouble(
                        left.data[left.broadcastAddress(index)], right.data[right.broadcastAddress(index)]);
                advance(index, resultShape);
            }
        }
        return destination;
    }

    private Tensor scalarOp(double scalar, BinaryOp op, boolean scalarFirst) {
        return scalarInto(scalar, alloc(shape), op, scalarFirst);
    }

    private Tensor scalarInto(double scalar, Tensor destination, BinaryOp op, boolean scalarFirst) {
        requireShape(destination, shape);
        Tensor source = safeInput(destination);
        int length = (int) size();
        if (source.isContiguous() && destination.isContiguous()) {
            if (backend == TensorBackend.VECTOR) {
                VectorKernels.scalar(source.data, source.offset, scalar, scalarFirst,
                        destination.data, destination.offset, length, op);
            } else {
                for (int i = 0; i < length; i++) {
                    double value = source.data[source.offset + i];
                    destination.data[destination.offset + i] = scalarFirst
                            ? op.applyAsDouble(scalar, value) : op.applyAsDouble(value, scalar);
                }
            }
        } else {
            int[] index = new int[rank()];
            for (int i = 0; i < length; i++) {
                double value = source.data[source.broadcastAddress(index)];
                destination.data[destination.broadcastAddress(index)] = scalarFirst
                        ? op.applyAsDouble(scalar, value) : op.applyAsDouble(value, scalar);
                advance(index, shape);
            }
        }
        return destination;
    }

    /** Broadcasts this + b into destination; shared views have snapshot semantics. */
    public Tensor addInto(Tensor b, Tensor destination) { return binaryInto(b, destination, BinaryOp.ADD); }
    /** Broadcasts this - b into destination; destination must have the exact result shape. */
    public Tensor subtractInto(Tensor b, Tensor destination) { return binaryInto(b, destination, BinaryOp.SUBTRACT); }
    /** Broadcasts this * b into destination; shared views have snapshot semantics. */
    public Tensor multiplyInto(Tensor b, Tensor destination) { return binaryInto(b, destination, BinaryOp.MULTIPLY); }
    /** Broadcasts this / b into destination; shared views have snapshot semantics. */
    public Tensor divideInto(Tensor b, Tensor destination) { return binaryInto(b, destination, BinaryOp.DIVIDE); }
    /** Writes this + scalar into a same-shaped destination. */
    public Tensor addInto(double scalar, Tensor destination) { return scalarInto(scalar, destination, BinaryOp.ADD, false); }
    /** Writes this - scalar into a same-shaped destination. */
    public Tensor subtractInto(double scalar, Tensor destination) { return scalarInto(scalar, destination, BinaryOp.SUBTRACT, false); }
    /** Writes this * scalar into a same-shaped destination. */
    public Tensor multiplyInto(double scalar, Tensor destination) { return scalarInto(scalar, destination, BinaryOp.MULTIPLY, false); }
    /** Writes this / scalar into a same-shaped destination. */
    public Tensor divideInto(double scalar, Tensor destination) { return scalarInto(scalar, destination, BinaryOp.DIVIDE, false); }

    public Tensor add(Tensor b) {
        return bin(b, BinaryOp.ADD);
    }

    public Tensor subtract(Tensor b) {
        return bin(b, BinaryOp.SUBTRACT);
    }

    public Tensor multiply(Tensor b) {
        return bin(b, BinaryOp.MULTIPLY);
    }

    public Tensor divide(Tensor b) {
        return bin(b, BinaryOp.DIVIDE);
    }

    public Tensor add(double x) {
        return scalarOp(x, BinaryOp.ADD, false);
    }

    public Tensor subtract(double x) {
        return scalarOp(x, BinaryOp.SUBTRACT, false);
    }

    public Tensor multiply(double x) {
        return scalarOp(x, BinaryOp.MULTIPLY, false);
    }

    public Tensor divide(double x) {
        return scalarOp(x, BinaryOp.DIVIDE, false);
    }

    public Tensor negate() {
        return map(v -> -v);
    }

    public Tensor pow(double p) {
        return map(v -> Math.pow(v, p));
    }

    public Tensor relu() {
        return reluInto(alloc(shape));
    }

    /** Writes ReLU into a same-shaped destination, with snapshot semantics for shared views. */
    public Tensor reluInto(Tensor destination) {
        requireShape(destination, shape);
        Tensor source = safeInput(destination);
        int length = (int) size();
        if (source.isContiguous() && destination.isContiguous()) {
            if (backend == TensorBackend.VECTOR)
                VectorKernels.relu(source.data, source.offset, destination.data, destination.offset, length);
            else for (int i = 0; i < length; i++)
                destination.data[destination.offset + i] = Math.max(0, source.data[source.offset + i]);
        } else {
            int[] index = new int[rank()];
            for (int i = 0; i < length; i++) {
                destination.data[destination.broadcastAddress(index)] = Math.max(0, source.data[source.broadcastAddress(index)]);
                advance(index, shape);
            }
        }
        return destination;
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
        if (backend == TensorBackend.VECTOR && b.strides[1] == 1) {
            VectorKernels.matmul(data, offset, strides[0], strides[1], b.data, b.offset,
                    b.strides[0], out.data, m, k, n);
            return out;
        }
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
