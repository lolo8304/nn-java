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

    @Test
    void preservesSeededShuffleOrderAndSequentialOrder() {
        Dataset data = new Dataset() {
            public int size() { return 11; }
            public Sample get(int i) { return new Sample(Tensor.scalar(i), Tensor.scalar(-i)); }
        };
        for (boolean shuffle : new boolean[]{false, true}) {
            int[] expected = java.util.stream.IntStream.range(0, data.size()).toArray();
            var random = new java.util.Random(123);
            if (shuffle) for (int i = expected.length - 1; i > 0; i--) {
                int j = random.nextInt(i + 1), previous = expected[i];
                expected[i] = expected[j];
                expected[j] = previous;
            }
            NN.seed(123);
            int pos = 0;
            for (var batch : new DataLoader(data, 4, shuffle)) {
                for (int row = 0; row < batch.inputs().shape()[0]; row++) {
                    assertEquals(expected[pos], batch.inputs().get(row));
                    assertEquals(-expected[pos++], batch.targets().get(row));
                }
            }
            assertEquals(expected.length, pos);
        }
    }

    @Test
    void rejectsDifferentSampleShapesEvenWhenElementCountsMatchOrAreEmpty() {
        for (boolean target : new boolean[]{false, true}) {
            for (int[] other : new int[][]{{3}, {1, 2}, {1}}) {
                assertMismatchedShapesRejected(Tensor.zeros(2), Tensor.zeros(other), target);
            }
            assertMismatchedShapesRejected(Tensor.zeros(0, 2), Tensor.zeros(0, 3), target);
        }
    }

    private void assertMismatchedShapesRejected(Tensor first, Tensor second, boolean target) {
        int[] calls = new int[2];
        Dataset data = new Dataset() {
            public int size() { return 2; }
            public Sample get(int i) {
                calls[i]++;
                Tensor value = i == 0 ? first : second;
                return target ? new Sample(Tensor.scalar(i), value) : new Sample(value, Tensor.scalar(i));
            }
        };
        assertThrows(IllegalArgumentException.class, () -> new DataLoader(data, 2, false).iterator().next());
        assertArrayEquals(new int[]{1, 1}, calls);
    }

    @Test
    void copiesOffsetStridedTargetsAndKeepsBatchesIndependent() {
        Tensor backing = Tensor.generate(p -> p[0] * 100 + p[1] * 10 + p[2], 3, 2, 4);
        Dataset data = new Dataset() {
            public int size() { return 3; }
            public Sample get(int i) {
                return new Sample(backing.slice(0, i), backing.slice(0, i).transpose());
            }
        };
        var iterator = new DataLoader(data, 2, false).iterator();
        var first = iterator.next();
        var last = iterator.next();
        assertArrayEquals(new int[]{1, 4, 2}, last.targets().shape());
        for (int row = 0; row < 2; row++) for (int a = 0; a < 2; a++) for (int b = 0; b < 4; b++) {
            assertEquals(row * 100 + a * 10 + b, first.inputs().get(row, a, b));
            assertEquals(row * 100 + a * 10 + b, first.targets().get(row, b, a));
        }
        backing.set(-1, 0, 0, 0);
        assertEquals(0, first.inputs().get(0, 0, 0));
        first.inputs().set(-2, 1, 1, 3);
        assertEquals(113, backing.get(1, 1, 3));
        assertEquals(213, last.targets().get(0, 3, 1));
    }

    @Test
    void handlesEmptyDatasetsEmptySamplesAndOversizedBatches() {
        Dataset empty = new Dataset() {
            public int size() { return 0; }
            public Sample get(int i) { throw new AssertionError("empty dataset fetched"); }
        };
        var loader = new DataLoader(empty, 3, true);
        assertEquals(0, loader.batches());
        assertFalse(loader.iterator().hasNext());
        assertThrows(NoSuchElementException.class, loader.iterator()::next);
        Dataset data = new Dataset() {
            public int size() { return 2; }
            public Sample get(int i) { return new Sample(Tensor.zeros(2, 0), Tensor.zeros(0)); }
        };
        loader = new DataLoader(data, 5, false);
        assertEquals(1, loader.batches());
        var batch = loader.iterator().next();
        assertArrayEquals(new int[]{2, 2, 0}, batch.inputs().shape());
        assertArrayEquals(new int[]{2, 0}, batch.targets().shape());
        assertEquals(0, batch.inputs().size());
    }

    @Test
    void fetchesWholeBatchBeforeCopyingReusedSampleStorage() {
        Tensor shared = Tensor.scalar(-1);
        Dataset data = new Dataset() {
            public int size() { return 3; }
            public Sample get(int i) {
                shared.set(i);
                return new Sample(shared, shared);
            }
        };
        var batch = new DataLoader(data, 3, false).iterator().next();
        for (int row = 0; row < 3; row++) {
            assertEquals(2, batch.inputs().get(row));
            assertEquals(2, batch.targets().get(row));
        }
    }
}
