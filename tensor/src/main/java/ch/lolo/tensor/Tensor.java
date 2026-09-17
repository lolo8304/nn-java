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
        else {
            StrideCursor cursor = new StrideCursor(shape, strides, offset);
            for (int i = 0; i < t.data.length; i++, cursor.advance()) t.data[i] = data[cursor.address];
        }
        return t;
    }

    /** Returns an elementwise transformation in new, independent storage. */
    public Tensor map(DoubleUnaryOperator f) {
        Tensor t = alloc(shape);
        if (isContiguous()) {
            for (int i = 0; i < t.data.length; i++) t.data[i] = f.applyAsDouble(data[offset + i]);
        } else {
            StrideCursor cursor = new StrideCursor(shape, strides, offset);
            for (int i = 0; i < t.data.length; i++, cursor.advance())
                t.data[i] = f.applyAsDouble(data[cursor.address]);
        }
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
            return broadcastInto(left, right, destination, op, vector, resultShape, length);
        }
        return destination;
    }

    // Keep uncommon layouts out of binaryInto to limit the JIT impact of the
    // larger arbitrary-rank dispatcher on existing common paths.
    private static Tensor broadcastInto(Tensor left, Tensor right, Tensor destination, BinaryOp op,
                                        boolean vector, int[] resultShape, int length) {
        int last = resultShape.length - 1;
        int ls = left.shape[left.rank() - 1] == 1 ? 0 : left.strides[left.rank() - 1];
        int rs = right.shape[right.rank() - 1] == 1 ? 0 : right.strides[right.rank() - 1];
        if (destination.isContiguous() && ls <= 1 && rs <= 1) {
            int width = resultShape[last];
            if (length == 0) return destination;
            int[] outer = resultShape.clone();
            outer[last] = 1;
            StrideCursor lc = StrideCursor.broadcast(outer, left.shape, left.strides, left.offset);
            StrideCursor rc = StrideCursor.broadcast(outer, right.shape, right.strides, right.offset);
            for (int i = 0; i < length; i += width, lc.advance(), rc.advance()) {
                if (!vector) {
                    int leftBase = lc.address, rightBase = rc.address, outputBase = destination.offset + i;
                    for (int j = 0; j < width; j++)
                        destination.data[outputBase + j] = op.applyAsDouble(
                                left.data[leftBase + j * ls], right.data[rightBase + j * rs]);
                } else if (ls == 1 && rs == 1)
                    VectorKernels.binary(left.data, lc.address, right.data, rc.address,
                            destination.data, destination.offset + i, width, op);
                else if (ls == 1)
                    VectorKernels.scalar(left.data, lc.address, right.data[rc.address], false,
                            destination.data, destination.offset + i, width, op);
                else if (rs == 1)
                    VectorKernels.scalar(right.data, rc.address, left.data[lc.address], true,
                            destination.data, destination.offset + i, width, op);
                else Arrays.fill(destination.data, destination.offset + i, destination.offset + i + width,
                            op.applyAsDouble(left.data[lc.address], right.data[rc.address]));
            }
        } else {
            StrideCursor lc = StrideCursor.broadcast(resultShape, left.shape, left.strides, left.offset);
            StrideCursor rc = StrideCursor.broadcast(resultShape, right.shape, right.strides, right.offset);
            StrideCursor dc = new StrideCursor(resultShape, destination.strides, destination.offset);
            for (int i = 0; i < length; i++, lc.advance(), rc.advance(), dc.advance())
                destination.data[dc.address] = op.applyAsDouble(left.data[lc.address], right.data[rc.address]);
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

    /**
     * Updates velocity = momentum * velocity - gradient * learningRate, then adds it
     * to this parameter tensor. All tensors must have exactly the same shape.
     * Writable storage address ranges must be disjoint; gradient aliases have snapshot semantics.
     * Returns this tensor. No element-sized temporary storage is needed for disjoint inputs.
     */
    public Tensor sgdStep(Tensor gradient, Tensor velocity, double learningRate, double momentum) {
        requireShape(gradient, shape);
        requireShape(velocity, shape);
        requireDisjoint(this, velocity);
        Tensor g = gradient.safeInput(this).safeInput(velocity);
        int length = (int) size(), start = 0;
        boolean contiguous = isContiguous() && g.isContiguous() && velocity.isContiguous();
        if (contiguous && backend == TensorBackend.VECTOR)
            start = VectorKernels.sgd(data, offset, g.data, g.offset, velocity.data, velocity.offset,
                    length, learningRate, momentum);
        int[] index = contiguous ? null : new int[rank()];
        for (int i = start; i < length; i++) {
            int pi = contiguous ? offset + i : broadcastAddress(index);
            int gi = contiguous ? g.offset + i : g.broadcastAddress(index);
            int vi = contiguous ? velocity.offset + i : velocity.broadcastAddress(index);
            double next = velocity.data[vi] * momentum - g.data[gi] * learningRate;
            velocity.data[vi] = next;
            data[pi] += next;
            if (!contiguous) advance(index, shape);
        }
        return this;
    }

    /**
     * Updates Adam's first/second moments and this parameter in one traversal.
     * correction1/2 are 1 - beta1/2^step; epsilon is outside the square root.
     * Shapes must match exactly and the three writable storage address ranges must be disjoint.
     * Gradient aliases have snapshot semantics. Returns this tensor.
     */
    public Tensor adamStep(Tensor gradient, Tensor first, Tensor second, double learningRate,
                           double beta1, double beta2, double correction1, double correction2, double epsilon) {
        requireShape(gradient, shape);
        requireShape(first, shape);
        requireShape(second, shape);
        requireDisjoint(this, first);
        requireDisjoint(this, second);
        requireDisjoint(first, second);
        Tensor g = gradient.safeInput(this).safeInput(first).safeInput(second);
        int length = (int) size(), start = 0;
        boolean contiguous = isContiguous() && g.isContiguous() && first.isContiguous() && second.isContiguous();
        if (contiguous && backend == TensorBackend.VECTOR)
            start = VectorKernels.adam(data, offset, g.data, g.offset, first.data, first.offset,
                    second.data, second.offset, length, learningRate, beta1, beta2, correction1, correction2, epsilon);
        int[] index = contiguous ? null : new int[rank()];
        for (int i = start; i < length; i++) {
            int pi = contiguous ? offset + i : broadcastAddress(index);
            int gi = contiguous ? g.offset + i : g.broadcastAddress(index);
            int mi = contiguous ? first.offset + i : first.broadcastAddress(index);
            int vi = contiguous ? second.offset + i : second.broadcastAddress(index);
            double grad = g.data[gi];
            double m = first.data[mi] * beta1 + grad * (1 - beta1);
            double v = second.data[vi] * beta2 + (grad * grad) * (1 - beta2);
            first.data[mi] = m;
            second.data[vi] = v;
            data[pi] += -learningRate * (m / correction1) / (Math.sqrt(v / correction2) + epsilon);
            if (!contiguous) advance(index, shape);
        }
        return this;
    }

    private static void requireDisjoint(Tensor a, Tensor b) {
        if (a.data == b.data && a.size() != 0 && b.size() != 0
                && a.offset <= b.lastAddress() && b.offset <= a.lastAddress())
            throw new IllegalArgumentException("optimizer outputs must have disjoint storage ranges");
    }

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

    /**
     * Writes upstream * ReLU'(this input) into destination, or adds it when accumulate is true.
     * Shapes must match exactly. Shared views have snapshot semantics; differently mapped
     * overlapping inputs may require copies. ReLU's derivative at either signed zero is zero.
     */
    public Tensor reluBackwardInto(Tensor upstream, Tensor destination, boolean accumulate) {
        return activationBackwardInto(upstream, destination, accumulate, ActivationDerivative.RELU, 0);
    }

    /** Like reluBackwardInto, but uses derivative one at zero and alpha for negative inputs. */
    public Tensor leakyReluBackwardInto(Tensor upstream, Tensor destination, boolean accumulate, double alpha) {
        return activationBackwardInto(upstream, destination, accumulate, ActivationDerivative.LEAKY_RELU, alpha);
    }

    /** Like reluBackwardInto, but this tensor is the saved sigmoid OUTPUT, not its input. */
    public Tensor sigmoidBackwardInto(Tensor upstream, Tensor destination, boolean accumulate) {
        return activationBackwardInto(upstream, destination, accumulate, ActivationDerivative.SIGMOID, 0);
    }

    /** Like reluBackwardInto, but this tensor is the saved tanh OUTPUT, not its input. */
    public Tensor tanhBackwardInto(Tensor upstream, Tensor destination, boolean accumulate) {
        return activationBackwardInto(upstream, destination, accumulate, ActivationDerivative.TANH, 0);
    }

    private enum ActivationDerivative {
        RELU, LEAKY_RELU, SIGMOID, TANH;

        double at(double x, double alpha) {
            return switch (this) {
                case RELU -> x > 0 ? 1 : 0;
                case LEAKY_RELU -> x >= 0 ? 1 : alpha;
                case SIGMOID -> x * (1 - x);
                case TANH -> 1 - x * x;
            };
        }
    }

    private Tensor activationBackwardInto(Tensor upstream, Tensor destination, boolean accumulate,
                                          ActivationDerivative derivative, double alpha) {
        requireShape(upstream, shape);
        requireShape(destination, shape);
        Tensor saved = safeInput(destination), g = upstream.safeInput(destination);
        int length = (int) size();
        boolean contiguous = saved.isContiguous() && g.isContiguous() && destination.isContiguous();
        int[] index = contiguous ? null : new int[rank()];
        for (int i = 0; i < length; i++) {
            int si = contiguous ? saved.offset + i : saved.broadcastAddress(index);
            int gi = contiguous ? g.offset + i : g.broadcastAddress(index);
            int di = contiguous ? destination.offset + i : destination.broadcastAddress(index);
            double contribution = g.data[gi] * derivative.at(saved.data[si], alpha);
            destination.data[di] = accumulate ? destination.data[di] + contribution : contribution;
            if (!contiguous) advance(index, shape);
        }
        return destination;
    }

    /**
     * Elementwise sigmoid. Contiguous Vector inputs of at least 128 elements use
     * Vector EXP; finite results may differ by a few ULPs from Java Math.exp.
     */
    public Tensor sigmoid() {
        if (backend == TensorBackend.VECTOR && isContiguous() && size() >= 128) {
            Tensor out = alloc(shape);
            VectorKernels.sigmoid(data, offset, out.data, out.data.length);
            return out;
        }
        return map(v -> 1 / (1 + Math.exp(-v)));
    }

    /** Elementwise square; retains Java pow semantics. */
    public Tensor square() { return pow(2); }

    /** Elementwise hyperbolic tangent, evaluated with Math.tanh. */
    public Tensor tanh() { return map(Math::tanh); }

    /** Elementwise natural logarithm, evaluated with Math.log. */
    public Tensor log() { return map(Math::log); }

    public Tensor matmul(Tensor b) {
        if (rank() == 0 || b.rank() == 0) throw new IllegalArgumentException("scalar matmul");
        boolean av = rank() == 1, bv = b.rank() == 1;
        int m = av ? 1 : shape[rank() - 2], k = shape[rank() - 1], bk = bv ? b.shape[0] : b.shape[b.rank() - 2], n = bv ? 1 : b.shape[b.rank() - 1];
        if (k != bk) throw new IllegalArgumentException("inner dimensions differ");
        if (rank() == 2 && b.rank() == 2) return matmul2d(b, m, k, n);
        int[] ab = av ? new int[0] : Arrays.copyOf(shape, rank() - 2), bb = bv ? new int[0] : Arrays.copyOf(b.shape, b.rank() - 2), batch = TensorShape.broadcast(ab, bb);
        if (av && bv) {
            double s = 0;
            for (int q = 0; q < k; q++) s += data[offset + q * strides[0]] * b.data[b.offset + q * b.strides[0]];
            return scalar(s);
        }
        int[] os = new int[batch.length + (av ? 0 : 1) + (bv ? 0 : 1)];
        int p = 0;
        for (int x : batch) os[p++] = x;
        if (!av) os[p++] = m;
        if (!bv) os[p] = n;
        Tensor out = alloc(os);
        if (out.data.length == 0) return out;
        StrideCursor ac = StrideCursor.broadcast(batch, ab, Arrays.copyOf(strides, ab.length), offset);
        StrideCursor bc = StrideCursor.broadcast(batch, bb, Arrays.copyOf(b.strides, bb.length), b.offset);
        int count = TensorShape.sizeOf(batch);
        for (int i = 0; i < count; i++, ac.advance(), bc.advance())
            matmulInto(b, ac.address, av ? 0 : strides[rank() - 2], strides[rank() - 1],
                    bc.address, bv ? b.strides[0] : b.strides[b.rank() - 2], bv ? 0 : b.strides[b.rank() - 1],
                    out.data, i * m * n, m, k, n);
        return out;
    }

    // Keep allocation and scalar rank-2 arithmetic together: HotSpot can prove
    // output storage does not alias either input and auto-vectorize the Java loop.
    private Tensor matmul2d(Tensor b, int m, int k, int n) {
        Tensor out = alloc(m, n);
        if (MatmulKernels.shouldBlock(m, k, n)) {
            MatmulKernels.matmul(data, offset, strides[0], strides[1], b.data, b.offset,
                    b.strides[0], b.strides[1], out.data, 0, m, k, n, backend == TensorBackend.VECTOR);
            return out;
        }
        if (backend == TensorBackend.VECTOR) {
            if (b.strides[1] == 1) {
                VectorKernels.matmul(data, offset, strides[0], strides[1], b.data, b.offset,
                        b.strides[0], out.data, m, k, n);
                return out;
            }
            // Amortize fresh panel packing across rows; small products keep the scalar path.
            if (m >= 4 && k >= 8 && n >= 16) {
                VectorKernels.matmulPackedRight(data, offset, strides[0], strides[1], b.data, b.offset,
                        b.strides[0], b.strides[1], out.data, 0, m, k, n);
                return out;
            }
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

    // Output starts zeroed; each element accumulates q=0..k-1 without FMA.
    private void matmulInto(Tensor b, int ao, int ars, int aks, int bo, int bks, int bcs,
                            double[] out, int oo, int m, int k, int n) {
        if (m == 0 || n == 0 || k == 0) return;
        if (MatmulKernels.shouldBlock(m, k, n)) {
            MatmulKernels.matmul(data, ao, ars, aks, b.data, bo, bks, bcs,
                    out, oo, m, k, n, backend == TensorBackend.VECTOR);
            return;
        }
        if (backend == TensorBackend.VECTOR) {
            if (bcs == 1) {
                VectorKernels.matmul(data, ao, ars, aks, b.data, bo, bks, out, oo, n, m, k, n);
                return;
            }
            if (m >= 4 && k >= 8 && n >= 16) {
                VectorKernels.matmulPackedRight(data, ao, ars, aks, b.data, bo, bks, bcs,
                        out, oo, m, k, n);
                return;
            }
        }
        for (int row = 0; row < m; row++) {
            int left = ao + row * ars, output = oo + row * n;
            if (bcs == 1) {
                for (int q = 0; q < k; q++) {
                    double value = data[left + q * aks];
                    int right = bo + q * bks;
                    for (int col = 0; col < n; col++) out[output + col] += value * b.data[right + col];
                }
            } else {
                for (int col = 0; col < n; col++) {
                    double sum = 0;
                    int right = bo + col * bcs;
                    for (int q = 0; q < k; q++) sum += data[left + q * aks] * b.data[right + q * bks];
                    out[output + col] = sum;
                }
            }
        }
    }

    /** Sums in logical row-major order, including on the Vector backend. */
    public Tensor sum() {
        double sum = 0;
        if (isContiguous()) {
            for (int i = 0, length = (int) size(); i < length; i++) sum += data[offset + i];
        } else {
            StrideCursor cursor = new StrideCursor(shape, strides, offset);
            for (int i = 0, length = (int) size(); i < length; i++, cursor.advance()) sum += data[cursor.address];
        }
        return scalar(sum);
    }

    public Tensor mean() {
        return scalar(size() == 0 ? Double.NaN : sum().scalar() / size());
    }

    public Tensor min() { return extrema(true); }

    public Tensor max() { return extrema(false); }

    private Tensor extrema(boolean min) {
        int length = (int) size();
        if (length == 0) throw new IllegalStateException("empty");
        if (isContiguous()) {
            if (backend == TensorBackend.VECTOR)
                return scalar(VectorKernels.extrema(data, offset, length, min));
            double value = min ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
            for (int i = 0; i < length; i++) value = min ? Math.min(value, data[offset + i]) : Math.max(value, data[offset + i]);
            return scalar(value);
        }
        double value = min ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
        StrideCursor cursor = new StrideCursor(shape, strides, offset);
        for (int i = 0; i < length; i++, cursor.advance())
            value = min ? Math.min(value, data[cursor.address]) : Math.max(value, data[cursor.address]);
        return scalar(value);
    }

    public Tensor sum(int a) { return reduce(a, 0); }

    public Tensor mean(int a) {
        int ax = axis(a);
        return sum(ax).divide(shape[ax]);
    }

    public Tensor min(int a) { return reduce(a, 1); }

    public Tensor max(int a) { return reduce(a, 2); }

    // Operation: 0 = sum, 1 = min, 2 = max. SIMD spans independent output elements,
    // never the sum's reduction axis.
    private Tensor reduce(int a, int operation) {
        int ax = axis(a);
        int[] os = remove(shape, ax);
        Tensor out = alloc(os);
        double initial = operation == 0 ? 0 : operation == 1 ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
        if (operation != 0) Arrays.fill(out.data, initial);
        if (out.data.length == 0) return out;
        if (backend == TensorBackend.VECTOR && isContiguous() && ax < rank() - 1) {
            int inner = 1;
            for (int d = ax + 1; d < rank(); d++) inner *= shape[d];
            for (int o = 0; o < out.data.length; o += inner)
                VectorKernels.reduceRows(data, offset + (o / inner) * shape[ax] * inner,
                        out.data, o, shape[ax], inner, operation);
            return out;
        }
        StrideCursor cursor = new StrideCursor(os, remove(strides, ax), offset);
        for (int i = 0; i < out.data.length; i++, cursor.advance()) {
            double value = initial;
            for (int k = 0; k < shape[ax]; k++) {
                double next = data[cursor.address + k * strides[ax]];
                value = operation == 0 ? value + next : operation == 1 ? Math.min(value, next) : Math.max(value, next);
            }
            out.data[i] = value;
        }
        return out;
    }

    public Tensor softmax(int a) {
        int ax = axis(a);
        Tensor out = alloc(shape);
        if (out.data.length == 0) return out;
        int[] outer = remove(shape, ax);
        StrideCursor source = new StrideCursor(outer, remove(strides, ax), offset);
        StrideCursor target = new StrideCursor(outer, remove(out.strides, ax), 0);
        for (int row = 0, count = TensorShape.sizeOf(outer); row < count; row++, source.advance(), target.advance()) {
            double max = Double.NEGATIVE_INFINITY;
            boolean vector = backend == TensorBackend.VECTOR && strides[ax] == 1 && out.strides[ax] == 1;
            if (vector) max = VectorKernels.extrema(data, source.address, shape[ax], false);
            else for (int k = 0; k < shape[ax]; k++) max = Math.max(max, data[source.address + k * strides[ax]]);
            double sum = 0;
            for (int k = 0; k < shape[ax]; k++) {
                double e = Math.exp(data[source.address + k * strides[ax]] - max);
                out.data[target.address + k * out.strides[ax]] = e;
                sum += e;
            }
            if (vector) VectorKernels.scalar(out.data, target.address, sum, false, out.data, target.address, shape[ax], BinaryOp.DIVIDE);
            else for (int k = 0; k < shape[ax]; k++) out.data[target.address + k * out.strides[ax]] /= sum;
        }
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
            StrideCursor cursor = new StrideCursor(shape, strides, offset);
            for (int i = 0; i < x.length; i++, cursor.advance()) x[i] = data[cursor.address];
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
