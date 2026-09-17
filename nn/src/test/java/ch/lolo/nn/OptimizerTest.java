package ch.lolo.nn;

import ch.lolo.nn.optim.*;
import ch.lolo.tensor.Tensor;
import org.junit.jupiter.api.Test;
import java.io.*;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class OptimizerTest {
    @Test void adamMatchesReferenceAndSerializedMoments() throws Exception { verify(true, .7); }
    @Test void momentumMatchesReferenceAndSerializedVelocity() throws Exception { verify(false, .7); }
    @Test void plainSgdMatchesReference() throws Exception { verify(false, 0); }

    private void verify(boolean adam, double momentum) throws Exception {
        for (int[] shape : new int[][]{{}, {0, 3}, {19}, {3, 5}}) {
            Tensor value = Tensor.random(42L, -1.0, 1.0, shape);
            if (shape.length == 2) value = value.transpose();
            Parameter p = new Parameter(value), absent = new Parameter(Tensor.ones(2));
            Tensor reference = value.copy(), m = Tensor.zeros(value.shape()), v = Tensor.zeros(value.shape());
            Optimizer optimizer = adam ? new Adam(.03, .8, .95, 1e-5) : new SGD(.03, momentum);
            var parameters = List.of(p, absent);
            optimizer.step(parameters); // Absent gradients still advance Adam's global step.
            for (int step = 2; step <= 9; step++) {
                Tensor gradient = Tensor.random(step, -2.0, 2.0, value.shape());
                p.zeroGrad();
                p.backward(gradient);
                Tensor heldGradient = p.grad();
                double[] beforeGradient = heldGradient.toArray();
                if (adam) {
                    m = m.multiply(.8).add(gradient.multiply(.2));
                    v = v.multiply(.95).add(gradient.pow(2).multiply(1 - .95));
                    Tensor mm = m, vv = v;
                    double c1 = 1 - Math.pow(.8, step), c2 = 1 - Math.pow(.95, step);
                    reference = reference.add(Tensor.generate(i -> -.03 * (mm.get(i) / c1)
                            / (Math.sqrt(vv.get(i) / c2) + 1e-5), value.shape()));
                } else {
                    v = v.multiply(momentum).subtract(gradient.multiply(.03));
                    reference = reference.add(v);
                }
                optimizer.step(parameters);
                assertSame(value, p.value());
                assertSame(heldGradient, p.grad());
                assertArrayEquals(beforeGradient, p.grad().toArray(), 0);
                assertArrayEquals(reference.toArray(), value.toArray(), 1e-14);
                assertArrayEquals(new double[]{1, 1}, absent.value().toArray(), 0);
                var bytes = new ByteArrayOutputStream();
                optimizer.writeState(new DataOutputStream(bytes), parameters);
                var in = new DataInputStream(new ByteArrayInputStream(bytes.toByteArray()));
                assertEquals(.03, in.readDouble());
                if (adam) {
                    assertEquals(.8, in.readDouble()); assertEquals(.95, in.readDouble());
                    assertEquals(1e-5, in.readDouble()); assertEquals(step, in.readInt());
                    assertTensor(in, m); assertTensor(in, v);
                } else {
                    assertEquals(momentum, in.readDouble()); assertTensor(in, v);
                }
                // Restore into a fresh optimizer each step, including state for absent parameters.
                optimizer = adam ? new Adam(1) : new SGD(1);
                optimizer.readState(new DataInputStream(new ByteArrayInputStream(bytes.toByteArray())), parameters);
            }
        }
    }

    private void assertTensor(DataInputStream in, Tensor expected) throws Exception {
        assertEquals(expected.rank(), in.readInt());
        for (int dimension : expected.shape()) assertEquals(dimension, in.readInt());
        for (double element : expected.toArray()) assertEquals(element, in.readDouble(), 1e-14);
    }

    @Test void applyDeltaHandlesEmptyAndOverlappingViews() {
        new Parameter(Tensor.zeros(0, 3)).applyDelta(Tensor.zeros(0, 3));
        Tensor value = Tensor.of(new double[][]{{1, 2}, {3, 4}});
        new Parameter(value).applyDelta(value.transpose());
        assertArrayEquals(new double[]{2, 5, 5, 8}, value.toArray(), 0);
        assertThrows(IllegalArgumentException.class, () -> new Parameter(value).applyDelta(Tensor.ones(2)));
    }
}
