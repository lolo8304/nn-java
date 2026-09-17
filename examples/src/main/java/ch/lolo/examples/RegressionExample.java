package ch.lolo.examples;

import ch.lolo.nn.Sequential;
import ch.lolo.nn.layers.Dense;
import ch.lolo.nn.layers.ReLU;
import ch.lolo.nn.loss.MSELoss;
import ch.lolo.nn.optim.Adam;
import ch.lolo.tensor.Tensor;

public class RegressionExample {
    static void main(String[] a) {
        Tensor x = Tensor.generate(i -> i[0] / 20.0, 100, 1);
        Tensor y = x.multiply(3).add(2);
        Sequential m = Sequential.of(new Dense(1, 16), new ReLU(), new Dense(16, 1)).compile(new Adam(.01), new MSELoss());
        m.fit(x, y, 100, 10);
        System.out.println(m.predict(Tensor.of(new double[][]{{2}})));
    }
}
