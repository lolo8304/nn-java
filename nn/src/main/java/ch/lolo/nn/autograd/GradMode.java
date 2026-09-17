package ch.lolo.nn.autograd;

import java.util.Objects;
import java.util.function.Supplier;

/** Controls graph recording on the current thread, independently of layer training mode. */
public final class GradMode {
    private static final ThreadLocal<Boolean> ENABLED = ThreadLocal.withInitial(() -> true);

    private GradMode() {}

    public static boolean isEnabled() {
        return ENABLED.get();
    }

    /**
     * Runs an operation without recording new autograd nodes. Nested calls and exceptions
     * restore the previous mode. Other threads are unaffected. Explicitly constructed
     * parameters keep requiresGrad=true; existing graphs can still be differentiated.
     */
    public static <T> T noGrad(Supplier<T> operation) {
        Objects.requireNonNull(operation, "operation");
        boolean previous = ENABLED.get();
        ENABLED.set(false);
        try {
            return operation.get();
        } finally {
            ENABLED.set(previous);
        }
    }
}
