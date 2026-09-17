package ch.lolo.nn;

import ch.lolo.nn.autograd.GradMode;
import ch.lolo.nn.autograd.Variable;
import ch.lolo.nn.data.TensorDataset;
import ch.lolo.nn.layers.*;
import ch.lolo.nn.loss.MSELoss;
import ch.lolo.nn.optim.Adam;
import ch.lolo.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.Executors;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.*;

class InferenceTest {
    @Test
    void allOperationsPreserveValuesWithoutRetainingGraph() throws Exception {
        List<Function<Variable, Variable>> operations = List.of(
                x -> x.add(x), x -> x.add(Tensor.ones(2, 2)), x -> x.add(2),
                x -> x.subtract(x), x -> x.multiply(x), x -> x.multiply(2),
                x -> x.divide(x), x -> x.matmul(x), Variable::sum, Variable::mean,
                x -> x.pow(2), Variable::log, Variable::relu, Variable::sigmoid,
                Variable::tanh, x -> x.leakyRelu(.1), x -> x.reshape(4),
                Variable::flatten, Variable::transpose, x -> x.softmax(-1));
        var parents = Variable.class.getDeclaredField("parents");
        var backward = Variable.class.getDeclaredField("backwardFn");
        parents.setAccessible(true);
        backward.setAccessible(true);
        for (boolean strided : new boolean[]{false, true}) {
            Tensor data = Tensor.of(new double[][]{{.2, .4}, {.6, .8}});
            var input = Variable.parameter(strided ? data.transpose() : data);
            input.sum().backward();
            Tensor gradient = input.grad();
            for (var operation : operations) {
                var recorded = operation.apply(input);
                var detached = GradMode.noGrad(() -> operation.apply(input));
                assertTrue(recorded.requiresGrad());
                assertFalse(detached.requiresGrad());
                assertArrayEquals(recorded.value().shape(), detached.value().shape());
                assertArrayEquals(recorded.value().toArray(), detached.value().toArray());
                assertTrue(((List<?>) parents.get(detached)).isEmpty());
                assertNull(backward.get(detached));
                detached.backward(Tensor.ones(detached.value().shape()));
                assertSame(gradient, input.grad());
                assertNull(detached.grad());
                var constant = operation.apply(Variable.of(input.value()));
                assertFalse(constant.requiresGrad());
                assertTrue(((List<?>) parents.get(constant)).isEmpty());
                assertNull(backward.get(constant));
            }
        }
    }

    @Test
    void nestedScopesRestoreAfterExceptionsAndDoNotAffectOtherThreads() throws Exception {
        var parameter = Variable.parameter(Tensor.scalar(3));
        try (var executor = Executors.newSingleThreadExecutor()) {
            GradMode.noGrad(() -> {
                assertFalse(GradMode.isEnabled());
                assertThrows(IllegalStateException.class, () -> GradMode.noGrad(() -> {
                    throw new IllegalStateException("nested failure");
                }));
                assertFalse(GradMode.isEnabled());
                try {
                    assertTrue(executor.submit(() -> parameter.pow(2).requiresGrad()).get());
                } catch (Exception e) {
                    throw new AssertionError(e);
                }
                // Explicitly created parameters retain their identity and trainability.
                assertTrue(Variable.parameter(Tensor.scalar(1)).requiresGrad());
                return null;
            });
        }
        assertThrows(IllegalStateException.class, () -> GradMode.noGrad(() -> {
            throw new IllegalStateException("outer failure");
        }));
        assertTrue(GradMode.isEnabled());
        assertTrue(parameter.requiresGrad());
        parameter.pow(2).backward();
        assertEquals(6, parameter.grad().scalar());
    }

    @Test
    void predictionMatchesRecordedForwardAndPreservesParameterGradients() {
        NN.seed(42);
        var model = Sequential.of(new Flatten(), new Dense(4, 6), new ReLU(), new Dropout(.5),
                new LeakyReLU(), new Sigmoid(), new Tanh(), new Dense(6, 3), new Softmax());
        Tensor input = Tensor.random(43L, 2, 2, 2);
        model.forward(Variable.of(input), true).sum().backward();
        var parameters = model.parameters();
        var gradients = parameters.stream().map(Variable::grad).toList();
        var gradientValues = gradients.stream().map(Tensor::toArray).toList();
        var weights = parameters.stream().map(p -> p.value().toArray()).toList();
        var reference = model.forward(Variable.of(input), false);
        assertTrue(reference.requiresGrad());
        for (int run = 0; run < 3; run++)
            assertArrayEquals(reference.value().toArray(), model.predict(input).toArray());
        for (int i = 0; i < parameters.size(); i++) {
            assertSame(gradients.get(i), parameters.get(i).grad());
            assertArrayEquals(gradientValues.get(i), parameters.get(i).grad().toArray());
            assertArrayEquals(weights.get(i), parameters.get(i).value().toArray());
        }
        assertTrue(GradMode.isEnabled());
        parameters.forEach(Variable::zeroGrad);
        model.predict(input);
        parameters.forEach(p -> assertNull(p.grad()));
        model.forward(Variable.of(input), false).sum().backward();
        parameters.forEach(p -> assertNotNull(p.grad()));
    }

    @Test
    void dropoutTrainingModeIsIndependentOfRecordingAndPredictionConsumesNoRandomness() {
        var dropout = new Dropout(.5);
        var input = Variable.parameter(Tensor.ones(64));
        NN.seed(123);
        var recorded = dropout.forward(input, true);
        NN.seed(123);
        var detached = GradMode.noGrad(() -> dropout.forward(input, true));
        assertArrayEquals(recorded.value().toArray(), detached.value().toArray());
        assertTrue(recorded.requiresGrad());
        assertFalse(detached.requiresGrad());
        assertTrue(java.util.Arrays.stream(detached.value().toArray()).anyMatch(v -> v == 0));
        assertTrue(java.util.Arrays.stream(detached.value().toArray()).anyMatch(v -> v == 2));
        assertSame(input, dropout.forward(input, false));
        var model = Sequential.of(dropout);
        NN.seed(123);
        double expectedRandom = NN.random().nextDouble();
        NN.seed(123);
        assertArrayEquals(input.value().toArray(), model.predict(input.value()).toArray());
        assertEquals(expectedRandom, NN.random().nextDouble());
    }

    @Test
    void predictionAndEvaluationScopeIncludesCustomLayersAndLossAndRestoresOnFailure() {
        Layer inspection = new Layer() {
            public Variable forward(Variable input, boolean training) {
                assertFalse(training);
                assertFalse(GradMode.isEnabled());
                return input.multiply(2);
            }
            public String type() { return "inspection"; }
        };
        var model = Sequential.of(inspection).compile(new Adam(.01), new ch.lolo.nn.loss.Loss() {
            public Variable compute(Variable prediction, Tensor target) {
                assertFalse(GradMode.isEnabled());
                return new MSELoss().compute(prediction, target);
            }
            public String type() { return "inspection"; }
        });
        assertEquals(2, model.predict(Tensor.scalar(1)).scalar());
        var result = model.evaluate(new TensorDataset(Tensor.ones(3, 1), Tensor.zeros(3, 1)), 2);
        assertEquals(4, result.loss());
        assertTrue(GradMode.isEnabled());
        var dense = Sequential.of(new Dense(2, 1));
        assertThrows(IllegalArgumentException.class, () -> dense.predict(Tensor.ones(1, 3)));
        assertTrue(GradMode.isEnabled());
        assertThrows(IllegalArgumentException.class,
                () -> model.evaluate(new TensorDataset(Tensor.zeros(0, 1), Tensor.zeros(0, 1)), 2));
        assertTrue(GradMode.isEnabled());
    }

    @Test
    void detachedResultsCanBeConstantsInLaterTrainingAndExistingGraphsStillBackpropagate() {
        var source = Variable.parameter(Tensor.scalar(3));
        var recorded = source.pow(2);
        var detached = GradMode.noGrad(() -> source.multiply(4));
        GradMode.noGrad(() -> {
            recorded.backward();
            return null;
        });
        assertEquals(6, source.grad().scalar());
        var other = Variable.parameter(Tensor.scalar(2));
        detached.multiply(other).backward();
        assertEquals(12, other.grad().scalar());
        assertEquals(6, source.grad().scalar());
        assertNull(detached.grad());
    }

    @Test
    void evaluationPreservesGradientsWeightsAndOptimizerStep() {
        NN.seed(42);
        var optimizer = new Adam(.01);
        var loss = new MSELoss();
        var model = Sequential.of(new Dense(2, 3), new Dropout(.5), new Dense(3, 1))
                .compile(optimizer, loss);
        Tensor x = Tensor.ones(3, 2), y = Tensor.zeros(3, 1);
        loss.compute(model.forward(Variable.of(x), true), y).backward();
        var parameters = model.parameters();
        var gradients = parameters.stream().map(Variable::grad).toList();
        var values = gradients.stream().map(Tensor::toArray).toList();
        var weights = parameters.stream().map(p -> p.value().toArray()).toList();
        double expected = loss.compute(model.forward(Variable.of(x), false), y).value().scalar();
        var evaluation = model.evaluate(new TensorDataset(x, y), 2);
        assertEquals(expected, evaluation.loss(), 1e-14);
        assertEquals(0, optimizer.stepCount());
        for (int i = 0; i < parameters.size(); i++) {
            assertSame(gradients.get(i), parameters.get(i).grad());
            assertArrayEquals(values.get(i), parameters.get(i).grad().toArray());
            assertArrayEquals(weights.get(i), parameters.get(i).value().toArray());
        }
    }

}
