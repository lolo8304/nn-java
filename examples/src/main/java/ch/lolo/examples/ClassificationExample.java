package ch.lolo.examples;

import ch.lolo.nn.Sequential;
import ch.lolo.nn.layers.Dense;
import ch.lolo.nn.layers.Tanh;
import ch.lolo.nn.loss.CrossEntropyLoss;
import ch.lolo.nn.optim.Adam;
import ch.lolo.tensor.Tensor;

public class ClassificationExample {
    static void main(String[] a) {
        Tensor x = Tensor.of(new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}});
        Tensor y = Tensor.of(new double[][]{{1, 0}, {0, 1}, {0, 1}, {1, 0}});
        Sequential m = Sequential.of(new Dense(2, 8), new Tanh(), new Dense(8, 2)).compile(new Adam(.03), new CrossEntropyLoss());
        m.fit(x, y, 200, 4);
        System.out.println(m.predictClasses(x));
    }
}
