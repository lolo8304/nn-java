package ch.lolo.nn;

import ch.lolo.nn.data.*;
import ch.lolo.tensor.Tensor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class AugmentedImageDatasetTest {
    private Dataset image(Tensor pixels) {
        var sample = new Sample(pixels, Tensor.of(0, 0, 0, 1, 0, 0, 0, 0, 0, 0));
        return new Dataset() {
            public int size() { return 1; }
            public Sample get(int i) { return sample; }
        };
    }

    @Test
    void transformsAreFreshReproducibleAndPreserveSourceAndLabel() {
        var pixels = Tensor.generate(i -> {
            int x = i[0] % 28, y = i[0] / 28;
            return x >= 10 && x <= 13 && y >= 5 && y <= 22 ? 1 : 0;
        }, 784);
        var source = image(pixels);
        var before = pixels.toArray();
        var a = new AugmentedImageDataset(source, 28, 28, 3, .85, 1.15, 10, 43);
        var b = new AugmentedImageDataset(source, 28, 28, 3, .85, 1.15, 10, 43);
        var first = a.get(0);
        assertArrayEquals(first.input().toArray(), b.get(0).input().toArray());
        var second = a.get(0);
        assertArrayEquals(second.input().toArray(), b.get(0).input().toArray());
        assertFalse(java.util.Arrays.equals(first.input().toArray(), second.input().toArray()));
        assertArrayEquals(before, source.get(0).input().toArray());
        assertSame(source.get(0).target(), first.target());
        assertArrayEquals(new int[]{784}, first.input().shape());
        assertTrue(first.input().sum().scalar() > 0);
        for (double value : first.input().toArray()) assertTrue(value >= 0 && value <= 1);
    }

    @Test
    void identityBlankAndImpossibleTransformsAreSafe() {
        var pixels = Tensor.random(42L, 28, 28);
        var identity = new AugmentedImageDataset(image(pixels), 28, 28, 0, 1, 1, 0, 43);
        assertArrayEquals(pixels.toArray(), identity.get(0).input().toArray(), 0);
        assertArrayEquals(new int[]{28, 28}, identity.get(0).input().shape());
        var impossible = new AugmentedImageDataset(image(pixels), 28, 28, 0, 2, 2, 0, 43);
        assertArrayEquals(pixels.toArray(), impossible.get(0).input().toArray());
        var blank = new AugmentedImageDataset(image(Tensor.zeros(784)), 28, 28, 3, .85, 1.15, 10, 43);
        assertEquals(0, blank.get(0).input().sum().scalar());
        assertThrows(IllegalArgumentException.class,
                () -> new AugmentedImageDataset(image(pixels), 28, 28, 0, 0, 1, 0, 43));
    }
}
