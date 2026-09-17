package ch.lolo.nn.data;

import java.util.Arrays;
import java.util.Random;

public interface Dataset {
    int size();

    Sample get(int index);

    /**
     * Splits this dataset into disjoint, shuffled training and validation views.
     * Samples are not copied; the source dataset must retain its size and ordering.
     * The local seed makes the split reproducible without changing the training RNG.
     *
     * @param validationFraction fraction reserved for validation, strictly between 0 and 1
     * @param seed seed used to shuffle the sample indices
     * @return training at index 0 and validation at index 1; validation size is rounded
     * @throws IllegalArgumentException if the fraction is invalid or either subset is empty
     */
    default Dataset[] splitTraining(double validationFraction, long seed) {
        if (!(validationFraction > 0 && validationFraction < 1)) {
            throw new IllegalArgumentException("Validation fraction must be between 0 and 1");
        }
        int validationSize = (int) Math.round(size() * validationFraction);
        if (validationSize < 1 || validationSize >= size()) {
            throw new IllegalArgumentException("Training and validation sets must both be nonempty");
        }
        int[] indices = new int[size()];
        for (int i = 0; i < indices.length; i++) indices[i] = i;
        var random = new Random(seed);
        for (int i = indices.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int swap = indices[i];
            indices[i] = indices[j];
            indices[j] = swap;
        }
        int boundary = indices.length - validationSize;
        return new Dataset[]{subset(this, Arrays.copyOfRange(indices, 0, boundary)),
                subset(this, Arrays.copyOfRange(indices, boundary, indices.length))};
    }

    private static Dataset subset(Dataset source, int[] indices) {
        return new Dataset() {
            public int size() { return indices.length; }
            public Sample get(int index) { return source.get(indices[index]); }
        };
    }

}
