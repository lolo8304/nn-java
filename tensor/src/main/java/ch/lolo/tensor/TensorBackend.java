package ch.lolo.tensor;

/** Execution backend for all tensors and neural networks in this JVM. */
public enum TensorBackend {
    /** Ordinary Java kernels; no incubator module is required at runtime. */
    JAVA,
    /** Explicit SIMD kernels where supported, with Java fallbacks for other operations. */
    VECTOR
}
