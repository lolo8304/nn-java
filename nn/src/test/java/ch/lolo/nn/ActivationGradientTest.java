package ch.lolo.nn;

import ch.lolo.nn.autograd.Variable;
import ch.lolo.tensor.Tensor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ActivationGradientTest {
    private Variable apply(int kind, Variable x) {
        return switch (kind) {
            case 0 -> x.relu();
            case 1 -> x.leakyRelu(.13);
            case 2 -> x.sigmoid();
            default -> x.tanh();
        };
    }

    @Test void finiteDifferencesOnStridedInputsWithNonuniformSeeds() {
        Tensor input = Tensor.of(new double[][]{{-.8, .3, 1.2}, {-1.3, .7, -.4}}).transpose();
        Tensor seed = Tensor.of(new double[][]{{-.2, .5, 1.4}, {.7, -.8, .3}}).transpose();
        for (int kind = 0; kind < 4; kind++) {
            var x = Variable.parameter(input);
            apply(kind, x).backward(seed);
            for (int i = 0; i < 3; i++) for (int j = 0; j < 2; j++) {
                Tensor plus = input.copy(), minus = input.copy();
                plus.set(input.get(i, j) + 1e-6, i, j);
                minus.set(input.get(i, j) - 1e-6, i, j);
                double numeric = (apply(kind, Variable.of(plus)).value().multiply(seed).sum().scalar()
                        - apply(kind, Variable.of(minus)).value().multiply(seed).sum().scalar()) / 2e-6;
                assertEquals(numeric, x.grad().get(i, j), 1e-8);
            }
        }
    }

    @Test void zeroAndSaturationConventions() {
        for (int kind = 0; kind < 4; kind++) {
            var x = Variable.parameter(Tensor.of(-1000, -0.0, 0.0, 1000));
            apply(kind, x).backward(Tensor.ones(4));
            double[] expected = switch (kind) {
                case 0 -> new double[]{0, 0, 0, 1};
                case 1 -> new double[]{.13, 1, 1, 1};
                case 2 -> new double[]{0, .25, .25, 0};
                default -> new double[]{0, 1, 1, 0};
            };
            assertArrayEquals(expected, x.grad().toArray());
            var empty = Variable.parameter(Tensor.zeros(0, 3));
            apply(kind, empty).backward(Tensor.zeros(0, 3));
            assertEquals(0, empty.grad().size());
        }
    }

    @Test void accumulationReusesOwnedDestinationWithoutAliasingSeedOrActivation() {
        for (int kind = 0; kind < 4; kind++) {
            var x = Variable.parameter(Tensor.of(-.7, .4, 1.2));
            var y = apply(kind, x);
            Tensor seed = Tensor.of(.2, -.3, .8);
            y.backward(seed);
            Tensor owned = x.grad(), first = owned.copy();
            y.backward(seed);
            assertSame(owned, x.grad());
            assertArrayEquals(first.multiply(2).toArray(), owned.toArray());
            owned.set(999, 0);
            assertEquals(.2, seed.get(0));
            assertEquals(-.7, x.value().get(0));
            assertNotEquals(999, y.value().get(0));
            assertNotEquals(999, y.grad().get(0));
            x.zeroGrad();
            assertNull(x.grad());
            y.backward(seed);
            assertNotSame(owned, x.grad());
            assertArrayEquals(first.toArray(), x.grad().toArray());
        }
    }

    @Test void branchesSavedValuesAndRepeatedBackward() {
        for (int kind = 0; kind < 4; kind++) {
            Tensor input = Tensor.of(-.7, .4, 1.2), seed = Tensor.of(.2, -.3, .8);
            var reference = Variable.parameter(input.copy());
            apply(kind, reference).backward(seed);
            Tensor derivative = reference.grad().copy();
            var x = Variable.parameter(input);
            var y = apply(kind, x);
            double[] saved = y.value().toArray(), before = input.toArray(), seedBefore = seed.toArray();
            // Both saved activation and input are consumed by other graph branches.
            var z = y.multiply(y).add(y).add(x.multiply(x));
            Tensor expected = derivative.multiply(y.value().multiply(2).add(1)).add(input.multiply(seed).multiply(2));
            z.backward(seed);
            assertArrayEquals(expected.toArray(), x.grad().toArray(), 1e-14);
            z.backward(seed);
            assertArrayEquals(expected.multiply(2).toArray(), x.grad().toArray(), 1e-14);
            x.zeroGrad(); z.backward(seed);
            assertArrayEquals(expected.toArray(), x.grad().toArray(), 1e-14);
            assertArrayEquals(saved, y.value().toArray());
            assertArrayEquals(before, input.toArray());
            assertArrayEquals(seedBefore, seed.toArray());
            // Two fused backward kernels contribute to the same leaf destination.
            var leaf = Variable.parameter(input.copy());
            apply(kind, leaf).add(apply(kind, leaf)).backward(seed);
            assertArrayEquals(derivative.multiply(2).toArray(), leaf.grad().toArray());
        }
    }
}
