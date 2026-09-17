package ch.lolo.nn;

import java.util.ArrayList;
import java.util.List;

public final class TrainingHistory {
    private final List<Double> loss = new ArrayList<>(), accuracy = new ArrayList<>();
    private final List<Double> validationLoss = new ArrayList<>(), validationAccuracy = new ArrayList<>();
    private int bestEpoch;
    private double bestValidationLoss = Double.POSITIVE_INFINITY;

    void add(double l, double a) {
        loss.add(l);
        accuracy.add(a);
    }

    public List<Double> loss() {
        return List.copyOf(loss);
    }

    public List<Double> accuracy() {
        return List.copyOf(accuracy);
    }

    void addValidation(double loss, double accuracy) {
        validationLoss.add(loss);
        validationAccuracy.add(accuracy);
        if (Double.isFinite(loss) && loss < bestValidationLoss) {
            bestValidationLoss = loss;
            bestEpoch = validationLoss.size();
        }
    }

    public List<Double> validationLoss() { return List.copyOf(validationLoss); }

    public List<Double> validationAccuracy() { return List.copyOf(validationAccuracy); }

    /** One-based best epoch, or zero if no finite validation loss was recorded. */
    public int bestEpoch() { return bestEpoch; }

    public double bestValidationLoss() { return bestValidationLoss; }
}
