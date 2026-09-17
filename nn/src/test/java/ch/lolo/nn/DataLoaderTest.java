package ch.lolo.nn;

import ch.lolo.nn.data.DataLoader;
import ch.lolo.nn.data.Dataset;
import ch.lolo.nn.data.Sample;
import ch.lolo.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.NoSuchElementException;

import static org.junit.jupiter.api.Assertions.*;

class DataLoaderTest {
    @Test
    void fetchesEachSampleOnceAndKeepsInputsAndTargetsPaired() {
        for (boolean shuffle : new boolean[]{false, true}) {
            int[] calls = new int[7];
            Dataset data = new Dataset() {
                public int size() { return calls.length; }
                public Sample get(int i) {
                    calls[i]++;
                    // Noncontiguous inputs and scalar targets exercise arbitrary sample shapes.
                    return new Sample(Tensor.generate(p -> i * 100 + p[0] * 10 + p[1], 2, 3).transpose(), Tensor.scalar(i));
                }
            };
            var iterator = new DataLoader(data, 3, shuffle).iterator();
            var seen = new HashSet<Integer>();
            int batches = 0;
            while (iterator.hasNext()) {
                var batch = iterator.next();
                int count = batch.inputs().shape()[0];
                assertEquals(batches++ == 2 ? 1 : 3, count);
                assertArrayEquals(new int[]{count, 3, 2}, batch.inputs().shape());
                for (int row = 0; row < count; row++) {
                    int id = (int) batch.targets().get(row);
                    assertTrue(seen.add(id));
                    for (int x = 0; x < 3; x++) for (int y = 0; y < 2; y++)
                        assertEquals(id * 100 + y * 10 + x, batch.inputs().get(row, x, y));
                }
            }
            assertEquals(7, seen.size());
            for (int count : calls) assertEquals(1, count);
            assertThrows(NoSuchElementException.class, iterator::next);
        }
    }
}
