package ch.lolo.nn;

import ch.lolo.nn.autograd.*;
import ch.lolo.nn.data.TensorDataset;
import ch.lolo.nn.layers.Dense;
import ch.lolo.nn.loss.CrossEntropyLoss;
import ch.lolo.nn.optim.SGD;
import ch.lolo.tensor.Tensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.Path;
import static org.junit.jupiter.api.Assertions.*;

class CrossEntropyTest {
    private final CrossEntropyLoss loss = new CrossEntropyLoss();
    private double value(Tensor x, Tensor y) { return loss.compute(Variable.of(x), y).value().scalar(); }

    @Test void normalizedAndWeightedTargets() {
        for (double[] targets : new double[][]{{1, 0, 0}, {.2, .3, .5}, {2, 3, 5}, {0, 0, 0}}) {
            var x = Variable.parameter(Tensor.zeros(3));
            var y = Tensor.of(targets);
            var out = loss.compute(x, y);
            double mass = targets[0] + targets[1] + targets[2];
            assertEquals(mass * Math.log(3), out.value().scalar(), 1e-14);
            out.backward();
            for (int i = 0; i < 3; i++) assertEquals(mass / 3 - targets[i], x.grad().get(i), 1e-14);
        }
    }

    @Test void agreesWithLegacyAwayFromSaturation() {
        Tensor data = Tensor.of(new double[][]{{.2, -.4, .8}, {1, -.2, -.5}});
        Tensor target = Tensor.of(new double[][]{{.1, .2, .7}, {0, 1, 0}});
        var fused = Variable.parameter(data);
        var legacy = Variable.parameter(data);
        var actual = loss.compute(fused, target);
        var p = legacy.softmax(-1).multiply(1 - 2e-12).add(1e-12);
        var expected = Variable.of(target).multiply(p.log()).sum().multiply(-.5);
        assertEquals(expected.value().scalar(), actual.value().scalar(), 1e-11);
        actual.backward();
        expected.backward();
        assertArrayEquals(legacy.grad().toArray(), fused.grad().toArray(), 1e-11);
        assertEquals(0, value(Tensor.of(42), Tensor.of(3)));
    }

    @Test void extremeLogitsAndObjectiveChange() {
        var x = Variable.parameter(Tensor.of(1000, -1000));
        var out = loss.compute(x, Tensor.of(0, 1));
        assertEquals(2000, out.value().scalar());
        out.backward();
        assertArrayEquals(new double[]{1, -1}, x.grad().toArray(), 0);
        assertEquals(0, value(Tensor.of(1000, -1000), Tensor.of(1, 0)));
        assertEquals(Math.log(2), value(Tensor.of(1e300, 1e300), Tensor.of(1, 0)), 1e-15);
        assertEquals(0, value(Tensor.of(Double.MAX_VALUE, -Double.MAX_VALUE), Tensor.of(1, 0)));
        double old = -Math.log(1e-12);
        assertTrue(out.value().scalar() > old * 50);
    }

    @Test void finiteDifferencesStridedAndHigherRank() {
        Tensor x = Tensor.of(new double[][]{{.2, -.3}, {1.2, 2}, {-.7, .8}}).transpose();
        Tensor y = Tensor.of(new double[][]{{.2, 0}, {.3, 2}, {.5, 1}}).transpose();
        checkGradient(x, y);
        checkGradient(x.reshape(1, 2, 3), y.reshape(1, 2, 3));
        assertEquals(value(x, y) * 2, value(x.reshape(1, 2, 3), y.reshape(1, 2, 3)), 1e-14);
        Tensor offset = Tensor.of(new double[][]{{9, 9, 9}, {.2, -.4, .8}}).slice(0, 1);
        checkGradient(offset, Tensor.of(.1, .2, .7));
    }

    private void checkGradient(Tensor x, Tensor y) {
        var v = Variable.parameter(x);
        loss.compute(v, y).backward();
        double[] data = x.toArray(), grad = v.grad().toArray();
        for (int i = 0; i < data.length; i++) {
            double old = data[i], h = 1e-5;
            data[i] = old + h;
            double plus = value(Tensor.of(data).reshape(x.shape()), y);
            data[i] = old - h;
            double minus = value(Tensor.of(data).reshape(x.shape()), y);
            data[i] = old;
            assertEquals((plus - minus) / (2 * h), grad[i], 1e-9);
        }
    }

    @Test void graphLifetimeSeedsAndNoGrad() {
        var x = Variable.parameter(Tensor.of(.2, -.5));
        var y = Tensor.of(.3, .7);
        var out = loss.compute(x, y);
        out.backward(Tensor.scalar(2));
        double[] first = x.grad().toArray();
        x.value().set(99, 0);
        y.set(10, 1);
        out.backward(Tensor.scalar(2));
        assertArrayEquals(new double[]{2 * first[0], 2 * first[1]}, x.grad().toArray(), 1e-14);
        x.zeroGrad();
        out.add(out).backward();
        assertArrayEquals(first, x.grad().toArray(), 1e-14);
        var detached = GradMode.noGrad(() -> loss.compute(x, y));
        assertFalse(detached.requiresGrad());
        assertEquals(value(x.value(), y), detached.value().scalar());
        assertFalse(loss.compute(Variable.of(x.value()), y).requiresGrad());
    }

    @Test void invalidInputs() {
        assertThrows(IllegalArgumentException.class, () -> value(Tensor.scalar(1), Tensor.scalar(1)));
        assertThrows(IllegalArgumentException.class, () -> value(Tensor.zeros(0, 3), Tensor.zeros(0, 3)));
        assertThrows(IllegalArgumentException.class, () -> value(Tensor.zeros(2, 3), Tensor.zeros(3)));
        for (double bad : new double[]{-1, Double.NaN, Double.POSITIVE_INFINITY})
            assertThrows(IllegalArgumentException.class, () -> value(Tensor.zeros(2), Tensor.of(1, bad)));
        for (double bad : new double[]{Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY})
            assertThrows(IllegalArgumentException.class, () -> value(Tensor.of(0, bad), Tensor.of(1, 0)));
    }

    @Test void convergenceAndPersistence(@TempDir Path directory) throws Exception {
        NN.seed(42);
        var model = Sequential.of(new Dense(2, 2)).compile(new SGD(.15), loss);
        var data = new TensorDataset(Tensor.of(new double[][]{{-2, -1}, {-1, -2}, {1, 2}, {2, 1}}),
                Tensor.of(new double[][]{{1, 0}, {1, 0}, {0, 1}, {0, 1}}));
        double before = model.evaluate(data, 4).loss();
        model.fit(data, 40, 4, false);
        assertTrue(model.evaluate(data, 4).loss() < before * .1);
        assertEquals(1, model.evaluate(data, 4).accuracy());
        Path file = directory.resolve("classification.nn");
        model.save(file);
        var restored = Sequential.load(file);
        assertInstanceOf(CrossEntropyLoss.class, restored.loss());
        assertEquals(model.evaluate(data, 4), restored.evaluate(data, 4));
        model.fit(data, 1, 4, false);
        restored.fit(data, 1, 4, false);
        for (int i = 0; i < model.parameters().size(); i++)
            assertArrayEquals(model.parameters().get(i).value().toArray(), restored.parameters().get(i).value().toArray(), 0);
    }
}
