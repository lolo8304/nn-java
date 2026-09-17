package ch.lolo.tensor;

/** Separate JVM entry point for checking optional-module loading and startup selection. */
public final class BackendProbe {
    public static void main(String[] args) {
        System.out.println(Tensor.backend());
        System.out.println(Tensor.ones(3, 5).multiply(2).relu().matmul(Tensor.ones(5, 7)).sum().scalar());
        if (args.length > 0) Tensor.setBackend(TensorBackend.VECTOR);
    }
}
