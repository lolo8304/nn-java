package ch.lolo.tensor;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class OptimizerKernelTest {
    @Test void fusedUpdatesMatchSnapshotsAcrossLayoutsAndAliases() {
        for (boolean adam : new boolean[]{false, true}) {
            for (int alias = 0; alias < 7; alias++) {
                Tensor p = Tensor.random(1L, -1.0, 1.0, 2, 5, 5).slice(0, 1);
                Tensor m = Tensor.random(2L, -1.0, 1.0, 5, 5).transpose();
                Tensor v = Tensor.random(3L, .1, 1.0, 5, 5).transpose();
                Tensor g = switch (alias) {
                    case 0 -> Tensor.random(4L, -1.0, 1.0, 5, 5).transpose();
                    case 1 -> p;
                    case 2 -> p.transpose();
                    case 3 -> m;
                    case 4 -> m.transpose();
                    case 5 -> v;
                    default -> v.transpose();
                };
                Tensor gg = g.copy(), pp = p.copy();
                Tensor mm = adam ? m.multiply(.8).add(gg.multiply(1 - .8))
                        : m.multiply(.8).subtract(gg.multiply(.03));
                Tensor vv = v.multiply(.95).add(gg.pow(2).multiply(1 - .95));
                Tensor expected = adam ? pp.add(Tensor.generate(i -> -.03 * (mm.get(i) / .2)
                        / (Math.sqrt(vv.get(i) / .05) + 1e-5), p.shape())) : pp.add(mm);
                if (adam) p.adamStep(g, m, v, .03, .8, .95, .2, .05, 1e-5);
                else p.sgdStep(g, m, .03, .8);
                assertArrayEquals(expected.toArray(), p.toArray(), 1e-14);
                assertArrayEquals(mm.toArray(), m.toArray(), 1e-14);
                if (adam) assertArrayEquals(vv.toArray(), v.toArray(), 1e-14);
            }
        }
    }

    @Test void contiguousOffsetAndTailSupportExactAliases() {
        for (boolean adam : new boolean[]{false, true}) {
            for (int alias = 0; alias < 3; alias++) {
                Tensor storage = Tensor.random(11L, .1, 1.0, 4, 19);
                Tensor p = storage.slice(0, 1), m = storage.slice(0, 2), v = storage.slice(0, 3);
                double[] untouched = storage.slice(0, 0).toArray();
                Tensor g = alias == 0 ? p : alias == 1 ? m : v;
                Tensor gg = g.copy();
                Tensor mm = adam ? m.multiply(.8).add(gg.multiply(1 - .8))
                        : m.multiply(.8).subtract(gg.multiply(.03));
                Tensor vv = v.multiply(.95).add(gg.pow(2).multiply(1 - .95));
                Tensor expected = adam ? p.add(Tensor.generate(i -> -.03 * (mm.get(i) / .2)
                        / (Math.sqrt(vv.get(i) / .05) + 1e-5), p.shape())) : p.add(mm);
                if (adam) p.adamStep(g, m, v, .03, .8, .95, .2, .05, 1e-5);
                else p.sgdStep(g, m, .03, .8);
                assertArrayEquals(expected.toArray(), p.toArray(), 1e-14);
                assertArrayEquals(mm.toArray(), m.toArray(), 1e-14);
                if (adam) assertArrayEquals(vv.toArray(), v.toArray(), 1e-14);
                assertArrayEquals(untouched, storage.slice(0, 0).toArray(), 0);
            }
        }
    }

    @Test void rejectsInvalidOutputsBeforeWriting() {
        Tensor p = Tensor.ones(3, 3), g = Tensor.ones(3, 3), m = Tensor.zeros(3, 3);
        assertThrows(IllegalArgumentException.class, () -> p.sgdStep(g, p.transpose(), .1, .9));
        assertThrows(IllegalArgumentException.class, () -> p.adamStep(g, m, m, .1, .9, .99, .1, .01, 1e-8));
        assertThrows(IllegalArgumentException.class, () -> p.adamStep(g, m, Tensor.zeros(9), .1, .9, .99, .1, .01, 1e-8));
        assertArrayEquals(Tensor.ones(3, 3).toArray(), p.toArray(), 0);
        assertArrayEquals(Tensor.zeros(3, 3).toArray(), m.toArray(), 0);
    }
}
