package ch.lolo.nn;

import java.util.Random;

public final class NN {
    private static long seed = 42;
    private static Random random = new Random(seed);

    private NN() {
    }

    public static void seed(long s) {
        seed = s;
        random = new Random(s);
    }

    public static long seed() {
        return seed;
    }

    public static Random random() {
        return random;
    }
}
