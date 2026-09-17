package ch.lolo.tensor;

import java.util.function.DoubleBinaryOperator;

/** Identifies operations without trying to infer the contents of a user lambda. */
enum BinaryOp implements DoubleBinaryOperator {
    ADD, SUBTRACT, MULTIPLY, DIVIDE;

    @Override
    public double applyAsDouble(double a, double b) {
        return switch (this) {
            case ADD -> a + b;
            case SUBTRACT -> a - b;
            case MULTIPLY -> a * b;
            case DIVIDE -> a / b;
        };
    }
}
