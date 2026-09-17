package ch.lolo.examples;

import ch.lolo.nn.NN;
import ch.lolo.nn.Sequential;
import ch.lolo.nn.layers.Dense;
import ch.lolo.nn.layers.Sigmoid;
import ch.lolo.nn.layers.Tanh;
import ch.lolo.nn.loss.BinaryCrossEntropyLoss;
import ch.lolo.nn.optim.Adam;
import ch.lolo.tensor.Tensor;

public class XorExample {
    static void main(String[] a) {
        NN.seed(42);
        Tensor x = Tensor.of(new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}});
        Tensor y = Tensor.of(new double[][]{{0}, {1}, {1}, {0}});
        Sequential m = Sequential.of(new Dense(2, 8), new Tanh(), new Dense(8, 1), new Sigmoid()).compile(new Adam(.03), new BinaryCrossEntropyLoss());
        m.fit(x, y, 500, 4);
        System.out.println(m.predict(x));
    }
}
