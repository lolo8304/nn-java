package ch.lolo.benchmark;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class TrainingWorkloadTest {
    @Test void completeEpochUpdatesWeightsAndRepeatsFromSeedWithPartialBatch() {
        var first = new TrainingWorkload(65, 17, 32, 42);
        double[] initial = first.parameters.getFirst().value().toArray();
        double loss = first.epoch();
        double[] trained = first.parameters.getFirst().value().toArray();
        assertTrue(Double.isFinite(loss));
        assertFalse(java.util.Arrays.equals(initial, trained));
        assertEquals(3, ((ch.lolo.nn.optim.Adam) first.optimizer).stepCount());
        var repeated = new TrainingWorkload(65, 17, 32, 42);
        assertEquals(loss, repeated.epoch());
        assertArrayEquals(trained, repeated.parameters.getFirst().value().toArray());
    }
}
