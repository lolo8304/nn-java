package ch.lolo.nn;

import ch.lolo.nn.data.TensorDataset;
import ch.lolo.tensor.Tensor;
import org.junit.jupiter.api.Test;
import java.util.HashSet;
import static org.junit.jupiter.api.Assertions.*;

class DatasetTest {
    @Test
    void splitIsDisjointCompleteAndReproducible() {
        var source = new TensorDataset(Tensor.generate(i -> i[0], 100, 1), Tensor.zeros(100, 1));
        var split = source.splitTraining(.1, 42);
        var repeated = source.splitTraining(.1, 42);
        assertEquals(90, split[0].size());
        assertEquals(10, split[1].size());
        var seen = new HashSet<Double>();
        for (int part = 0; part < 2; part++) {
            for (int i = 0; i < split[part].size(); i++) {
                double id = split[part].get(i).input().get(0);
                assertTrue(seen.add(id), "Sample appears more than once");
                assertEquals(id, repeated[part].get(i).input().get(0));
            }
        }
        assertEquals(100, seen.size());
    }

    @Test
    void rejectsEmptyPartitions() {
        var data = new TensorDataset(Tensor.zeros(1, 2), Tensor.zeros(1, 2));
        assertThrows(IllegalArgumentException.class, () -> data.splitTraining(.1, 42));
        assertThrows(IllegalArgumentException.class, () -> data.splitTraining(0, 42));
    }
}
