package ch.lolo.tensor;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;
import static org.junit.jupiter.api.Assertions.*;

class BackendTest {
    private final TensorBackend original = Tensor.backend();

    @AfterEach void restoreBackend() { Tensor.setBackend(original); }

    private void compare(Supplier<Tensor> operation) {
        Tensor.setBackend(TensorBackend.JAVA);
        Tensor expected = operation.get();
        Tensor.setBackend(TensorBackend.VECTOR);
        Tensor actual = operation.get();
        assertArrayEquals(expected.shape(), actual.shape());
        assertArrayEquals(expected.toArray(), actual.toArray());
    }

    @Test void switchValidatesBeforeChangingState() {
        Tensor.setBackend(TensorBackend.JAVA);
        assertThrows(NullPointerException.class, () -> Tensor.setBackend(null));
        assertEquals(TensorBackend.JAVA, Tensor.backend());
        if (!Tensor.isVectorAvailable()) {
            var error = assertThrows(IllegalStateException.class, () -> Tensor.setBackend(TensorBackend.VECTOR));
            assertTrue(error.getMessage().contains("--add-modules jdk.incubator.vector"));
            assertEquals(TensorBackend.JAVA, Tensor.backend());
        } else {
            Tensor existing = Tensor.ones(17);
            Tensor.setBackend(TensorBackend.VECTOR);
            assertEquals(34, existing.multiply(2).sum().scalar());
            Tensor.setBackend(TensorBackend.JAVA);
            assertEquals(34, existing.multiply(2).sum().scalar());
        }
    }

    @Test void arithmeticMatchesForTailsOffsetsViewsAndSpecialValues() {
        if (!Tensor.isVectorAvailable()) return; // Exercised by vectorTest with the module resolved.
        for (int length : new int[]{0, 1, 2, 3, 7, 16, 17, 65, 513}) {
            Tensor a = Tensor.random(1L, 2, length).slice(0, 1);
            Tensor b = Tensor.random(2L, 2, length).slice(0, 1);
            checkArithmetic(a, b);
            Tensor matrix = Tensor.random(3L, 3, length);
            compare(() -> matrix.add(b));
            compare(() -> matrix.subtract(b));
            compare(() -> matrix.multiply(b));
            compare(() -> matrix.divide(b));
            compare(() -> b.subtract(matrix)); // General broadcast fallback preserves operand order.
            compare(() -> b.divide(matrix));
        }
        checkArithmetic(Tensor.random(4L, 5, 7).transpose(), Tensor.random(5L, 5, 7).transpose());
        checkArithmetic(Tensor.scalar(-0.0), Tensor.scalar(2));
        Tensor special = Tensor.of(-0.0, 0.0, Double.NaN, Double.POSITIVE_INFINITY,
                Double.NEGATIVE_INFINITY, Double.MIN_VALUE, -Double.MIN_VALUE,
                Double.MAX_VALUE, -Double.MAX_VALUE);
        checkArithmetic(special, special);
    }

    private void checkArithmetic(Tensor a, Tensor b) {
        double[] beforeA = a.toArray(), beforeB = b.toArray();
        compare(() -> a.add(b)); compare(() -> a.subtract(b));
        compare(() -> a.multiply(b)); compare(() -> a.divide(b));
        compare(a::relu);
        for (double scalar : new double[]{-2, 0, -0.0, Double.POSITIVE_INFINITY}) {
            compare(() -> a.add(scalar)); compare(() -> a.subtract(scalar));
            compare(() -> a.multiply(scalar)); compare(() -> a.divide(scalar));
            compare(() -> Tensor.scalar(scalar).subtract(a));
            compare(() -> Tensor.scalar(scalar).divide(a));
        }
        assertArrayEquals(beforeA, a.toArray());
        assertArrayEquals(beforeB, b.toArray());
    }

    @Test void matmulMatchesForVectorTailsAndStrides() {
        if (!Tensor.isVectorAvailable()) return;
        for (int n : new int[]{0, 1, 3, 17, 65}) {
            Tensor a = Tensor.random(6L, 2, 7, 19).slice(0, 1);
            Tensor b = Tensor.random(7L, 19, 2, n).slice(1, 1);
            compare(() -> a.matmul(b));
            Tensor at = Tensor.random(8L, 19, 7).transpose();
            compare(() -> at.matmul(b));
            Tensor bt = Tensor.random(9L, n, 19).transpose();
            compare(() -> a.matmul(bt));
        }
        compare(() -> Tensor.zeros(7, 0).matmul(Tensor.zeros(0, 17)));
        compare(() -> Tensor.zeros(0, 19).matmul(Tensor.ones(19, 17)));
        compare(() -> Tensor.ones(2, 3, 5).matmul(Tensor.ones(5, 7)));
    }

    @Test void freshJvmsVerifyDefaultOptionalModuleAndProperty() throws Exception {
        runProbe(0, "VECTOR", true, null);
        runProbe(0, "JAVA", false, "java");
        runProbe(1, "--add-modules jdk.incubator.vector", false, null);
        runProbe(0, "JAVA", true, "java");
        runProbe(0, "VECTOR", true, "vector");
        runProbe(1, "--add-modules jdk.incubator.vector", false, "vector");
        runProbe(1, "--add-modules jdk.incubator.vector", false, "java", "switch");
        runProbe(1, "tensor.backend must be java or vector", false, "invalid");
    }

    private void runProbe(int exitCode, String expected, boolean module, String backend, String... args) throws Exception {
        List<String> command = new ArrayList<>();
        command.add(Path.of(System.getProperty("java.home"), "bin", "java").toString());
        if (module) command.addAll(List.of("--add-modules", "jdk.incubator.vector"));
        if (backend != null) command.add("-Dtensor.backend=" + backend);
        command.add("-cp");
        command.add(Path.of(BackendProbe.class.getProtectionDomain().getCodeSource().getLocation().toURI())
                + java.io.File.pathSeparator
                + Path.of(Tensor.class.getProtectionDomain().getCodeSource().getLocation().toURI()));
        command.add(BackendProbe.class.getName());
        command.addAll(List.of(args));
        Process process = new ProcessBuilder(command).redirectErrorStream(true).start();
        try {
            assertTrue(process.waitFor(20, TimeUnit.SECONDS), "Probe timed out");
            String output = new String(process.getInputStream().readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
            assertEquals(exitCode, process.exitValue(), output);
            assertTrue(output.contains(expected), output);
            if (exitCode == 0) assertTrue(output.contains("210.0"), output);
        } finally {
            process.destroyForcibly();
        }
    }
}
