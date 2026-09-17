package ch.lolo.nn.io;

import ch.lolo.nn.Layer;
import ch.lolo.nn.Sequential;
import ch.lolo.nn.layers.*;
import ch.lolo.nn.loss.BinaryCrossEntropyLoss;
import ch.lolo.nn.loss.CrossEntropyLoss;
import ch.lolo.nn.loss.Loss;
import ch.lolo.nn.loss.MSELoss;
import ch.lolo.nn.optim.Adam;
import ch.lolo.nn.optim.Optimizer;
import ch.lolo.nn.optim.SGD;
import ch.lolo.tensor.Tensor;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public final class ModelIO {
    private static final int MAGIC = 0x4A4E4E31;

    private ModelIO() {
    }

    public static void save(Sequential m, Path p) throws IOException {
        try (DataOutputStream o = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(p)))) {
            o.writeInt(MAGIC);
            o.writeInt(1);
            o.writeInt(m.layers().size());
            for (Layer l : m.layers()) {
                o.writeUTF(l.type());
                switch (l) {
                    case Dense d -> {
                        o.writeInt(d.inputSize());
                        o.writeInt(d.outputSize());
                        writeTensor(o, d.weight().value());
                        writeTensor(o, d.bias().value());
                    }
                    case LeakyReLU x -> o.writeDouble(x.alpha());
                    case Softmax x -> o.writeInt(x.axis());
                    case Dropout x -> o.writeDouble(x.probability());
                    default -> {
                    }
                }
            }
            o.writeBoolean(m.optimizer() != null);
            if (m.optimizer() != null) {
                o.writeUTF(m.optimizer().type());
                o.writeUTF(m.loss().type());
                m.optimizer().writeState(o, m.parameters());
            }
        }
    }

    public static Sequential load(Path p) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(p)))) {
            if (in.readInt() != MAGIC) throw new IOException("not JNN1");
            if (in.readInt() != 1) throw new IOException("unsupported version");
            int n = in.readInt();
            List<Layer> ls = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                String t = in.readUTF();
                ls.add(switch (t) {
                    case "Dense" -> {
                        int a = in.readInt(), b = in.readInt();
                        yield new Dense(a, b, readTensor(in), readTensor(in));
                    }
                    case "ReLU" -> new ReLU();
                    case "Sigmoid" -> new Sigmoid();
                    case "Tanh" -> new Tanh();
                    case "LeakyReLU" -> new LeakyReLU(in.readDouble());
                    case "Softmax" -> new Softmax(in.readInt());
                    case "Flatten" -> new Flatten();
                    case "Dropout" -> new Dropout(in.readDouble());
                    default -> throw new IOException("unknown layer " + t);
                });
            }
            Sequential m = new Sequential(ls);
            if (in.readBoolean()) {
                String ot = in.readUTF(), lt = in.readUTF();
                Optimizer o = switch (ot) {
                    case "Adam" -> new Adam(.001);
                    case "SGD" -> new SGD(.01);
                    default -> throw new IOException("optimizer " + ot);
                };
                Loss l = switch (lt) {
                    case "MSE" -> new MSELoss();
                    case "CrossEntropy" -> new CrossEntropyLoss();
                    case "BinaryCrossEntropy" -> new BinaryCrossEntropyLoss();
                    default -> throw new IOException("loss " + lt);
                };
                m.compile(o, l);
                o.readState(in, m.parameters());
            }
            return m;
        }
    }

    private static void writeTensor(DataOutput o, Tensor t) throws IOException {
        o.writeInt(t.rank());
        for (int x : t.shape()) o.writeInt(x);
        for (double x : t.toArray()) o.writeDouble(x);
    }

    private static Tensor readTensor(DataInput in) throws IOException {
        int r = in.readInt(), n = 1;
        int[] s = new int[r];
        for (int i = 0; i < r; i++) {
            s[i] = in.readInt();
            n *= s[i];
        }
        double[] a = new double[n];
        for (int i = 0; i < n; i++) a[i] = in.readDouble();
        int[] q = {0};
        return Tensor.generate(ix -> a[q[0]++], s);
    }
}
