package ch.lolo.nn;

import ch.lolo.nn.autograd.Variable;
import ch.lolo.tensor.Tensor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class GradientStorageTest {
    @Test void sharedGraphsReuseBuffersAcrossTraversalsAndResets() {
        var x = Variable.parameter(Tensor.of(2, 3));
        var shared = x.multiply(2);
        var first = shared.add(shared).add(x);
        first.backward(Tensor.of(1, 2));
        Tensor leafBuffer = x.grad(), sharedBuffer = shared.grad(), rootBuffer = first.grad();
        assertArrayEquals(new double[]{5, 10}, x.grad().toArray());
        first.backward(Tensor.of(3, 4));
        assertArrayEquals(new double[]{20, 30}, x.grad().toArray());
        assertSame(leafBuffer, x.grad());
        assertSame(sharedBuffer, shared.grad());
        assertSame(rootBuffer, first.grad());
        // A different root must clear the shared intermediate, but accumulate the leaf.
        shared.add(x).backward(Tensor.ones(2));
        assertArrayEquals(new double[]{23, 33}, x.grad().toArray());
        x.zeroGrad(); x.zeroGrad();
        assertNull(x.grad());
        first.backward(Tensor.ones(2));
        assertSame(leafBuffer, x.grad());
        assertArrayEquals(new double[]{5, 5}, x.grad().toArray());
    }

    @Test void firstContributionOverwritesNonfiniteStorageAndPreservesSignedZero() {
        var x = Variable.parameter(Tensor.zeros(3));
        x.backward(Tensor.of(Double.NaN, Double.POSITIVE_INFINITY, 99));
        Tensor owned = x.grad();
        x.zeroGrad();
        x.backward(Tensor.of(-0.0, 0.0, 3));
        assertSame(owned, x.grad());
        assertArrayEquals(new double[]{-0.0, 0.0, 3}, x.grad().toArray());
        assertEquals(Double.doubleToRawLongBits(-0.0), Double.doubleToRawLongBits(x.grad().get(0)));
    }

    @Test void seedsAndViewsDoNotBecomeOwnedStorage() {
        Tensor value = Tensor.of(new double[][]{{1, 2}, {3, 4}});
        var x = Variable.parameter(value);
        var y = x.add(0);
        Tensor seed = Tensor.of(new double[][]{{2, 3}, {5, 7}}).transpose();
        y.backward(seed);
        assertNotSame(seed, y.grad());
        assertNotSame(y.grad(), x.grad());
        x.zeroGrad();
        // Alias the root's retained buffer with a differently mapped seed view.
        y.backward(y.grad().transpose());
        assertArrayEquals(new double[]{2, 3, 5, 7}, x.grad().toArray());
        x.grad().set(99, 0, 0);
        assertArrayEquals(new double[]{2, 5, 3, 7}, seed.toArray());
        assertArrayEquals(new double[]{1, 2, 3, 4}, value.toArray());
        assertEquals(2, y.grad().get(0, 0));
    }

    @Test void leafBackwardAcceptsItsOwnGradientAndRejectsWrongSeedBeforeReset() {
        var x = Variable.parameter(Tensor.of(new double[][]{{1, 2}, {3, 4}}));
        x.backward(Tensor.of(new double[][]{{2, 3}, {5, 7}}));
        Tensor owned = x.grad();
        x.backward(owned.transpose());
        assertSame(owned, x.grad());
        assertArrayEquals(new double[]{4, 8, 8, 14}, owned.toArray());
        var root = x.add(1);
        root.backward(Tensor.ones(2, 2));
        Tensor rootBuffer = root.grad();
        assertThrows(IllegalArgumentException.class, () -> root.backward(Tensor.ones(4)));
        assertSame(rootBuffer, root.grad());
        assertArrayEquals(new double[]{1, 1, 1, 1}, root.grad().toArray());
        assertArrayEquals(new double[]{5, 9, 9, 15}, x.grad().toArray());
    }

    @Test void transposeAndBroadcastContributionsReuseCorrectShape() {
        var x = Variable.parameter(Tensor.ones(2, 3));
        var y = x.transpose().add(x.transpose());
        Tensor seed = Tensor.of(new double[][]{{1, 2}, {3, 4}, {5, 6}});
        y.backward(seed);
        Tensor owned = x.grad();
        x.zeroGrad(); y.backward(seed);
        assertSame(owned, x.grad());
        assertArrayEquals(new double[]{2, 6, 10, 4, 8, 12}, x.grad().toArray());
        var bias = Variable.parameter(Tensor.ones(1, 3));
        var broadcast = bias.add(Tensor.ones(2, 3));
        broadcast.backward(Tensor.ones(2, 3));
        Tensor biasBuffer = bias.grad();
        bias.zeroGrad(); broadcast.backward(Tensor.zeros(2, 3));
        assertSame(biasBuffer, bias.grad());
        assertArrayEquals(new double[]{0, 0, 0}, bias.grad().toArray());
    }

    @Test void resetGradientIsSkippedByOptimizersEvenWithExistingMomentum() {
        for (var optimizer : new ch.lolo.nn.optim.Optimizer[]{
                new ch.lolo.nn.optim.Adam(.1), new ch.lolo.nn.optim.SGD(.1, .9)}) {
            var p = new Parameter(Tensor.scalar(2));
            p.backward();
            optimizer.step(java.util.List.of(p));
            double updated = p.value().scalar();
            p.zeroGrad();
            optimizer.step(java.util.List.of(p));
            assertEquals(updated, p.value().scalar());
            assertNull(p.grad());
            // A present zero gradient is different: momentum still updates the value.
            p.backward(Tensor.scalar(0));
            optimizer.step(java.util.List.of(p));
            assertNotEquals(updated, p.value().scalar());
        }
    }

    @Test void absentEmptyAndDisabledGradientsRemainDistinct() {
        var empty = Variable.parameter(Tensor.zeros(0, 3));
        empty.backward(Tensor.zeros(0, 3));
        Tensor owned = empty.grad();
        empty.zeroGrad();
        assertNull(empty.grad());
        empty.backward(Tensor.zeros(0, 3));
        assertSame(owned, empty.grad());
        var constant = Variable.of(Tensor.scalar(1));
        constant.backward();
        assertNull(constant.grad());
        var unused = Variable.parameter(Tensor.scalar(2));
        unused.backward(); unused.zeroGrad();
        constant.add(3).backward();
        assertNull(unused.grad());
    }
}
