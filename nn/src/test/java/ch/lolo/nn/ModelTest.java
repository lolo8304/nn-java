package ch.lolo.nn;

import ch.lolo.nn.layers.Dense;
import ch.lolo.nn.layers.ReLU;
import ch.lolo.nn.layers.Sigmoid;
import ch.lolo.nn.layers.Tanh;
import ch.lolo.nn.loss.BinaryCrossEntropyLoss;
import ch.lolo.nn.optim.Adam;
import ch.lolo.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class ModelTest {
    @Test
    void forwardShape() {
        var m = Sequential.of(new Dense(3, 5), new ReLU(), new Dense(5, 2));
        assertArrayEquals(new int[]{4, 2}, m.predict(Tensor.ones(4, 3)).shape());
    }

    @Test
    void saveLoad() throws Exception {
        var m = Sequential.of(new Dense(2, 3), new Tanh(), new Dense(3, 1), new Sigmoid()).compile(new Adam(.01), new BinaryCrossEntropyLoss());
        var x = Tensor.ones(2, 2);
        var before = m.predict(x).toArray();
        Path p = Files.createTempFile("jnn", ".nn");
        m.save(p);
        var loaded = Sequential.load(p);
        assertArrayEquals(before, loaded.predict(x).toArray(), 1e-12);
        Files.delete(p);
    }
}
