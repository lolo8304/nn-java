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
    @Test void classificationReferenceAndFusedPerformEquivalentUpdates() {
        var fused = new TrainingWorkload(65, 17, 32, 42, new ch.lolo.nn.loss.CrossEntropyLoss(), true);
        double[] initial = fused.parameters.getFirst().value().toArray();
        double actual = fused.epoch();
        assertTrue(Double.isFinite(actual));
        assertFalse(java.util.Arrays.equals(initial, fused.parameters.getFirst().value().toArray()));
        assertEquals(3, ((ch.lolo.nn.optim.Adam) fused.optimizer).stepCount());
        var legacy = new TrainingWorkload(65, 17, 32, 42, new ClassificationBenchmarks.LegacyLoss(), true);
        assertEquals(legacy.epoch(), actual, 1e-9);
        for (int i = 0; i < fused.parameters.size(); i++)
            assertArrayEquals(legacy.parameters.get(i).value().toArray(), fused.parameters.get(i).value().toArray(), 1e-9);
    }
}
