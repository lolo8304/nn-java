package ch.lolo.examples;

import ch.lolo.nn.Sequential;
import ch.lolo.nn.layers.Dense;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.io.DataOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import static org.junit.jupiter.api.Assertions.*;

class MnistResumeTest {
    @Test
    void optionsAndCompatibilityChecks() throws Exception {
        var fresh = MnistExample.Options.parse(new String[]{});
        assertNull(fresh.resume());
        assertEquals(50, fresh.epochs());
        var resume = MnistExample.Options.parse(new String[]{"--resume", "mnist.nn"});
        assertEquals(20, resume.epochs());
        assertEquals(Path.of("mnist-finetuned.nn"), resume.output());
        assertThrows(IllegalArgumentException.class,
                () -> MnistExample.Options.parse(new String[]{"--resume", "mnist.nn", "--output", "./mnist.nn"}));
        assertThrows(IllegalArgumentException.class,
                () -> MnistExample.Options.parse(new String[]{"--epochs", "0"}));
        assertThrows(IllegalArgumentException.class,
                () -> MnistExample.Options.parse(new String[]{"--resume"}));
        assertThrows(IllegalArgumentException.class,
                () -> MnistExample.validateModel(Sequential.of(new Dense(784, 10))));
    }

    @Test
    void resumesThroughExampleWithoutOverwritingOriginal(@TempDir Path directory) throws Exception {
        writeData(directory, "train", 20);
        writeData(directory, "t10k", 2);
        var original = directory.resolve("original.nn");
        var output = directory.resolve("resumed.nn");
        MnistExample.main(new String[]{directory.toString(), "--output", original.toString(), "--epochs", "1"});
        var before = Files.readAllBytes(original);
        var initialSteps = ((ch.lolo.nn.optim.Adam) Sequential.load(original).optimizer()).stepCount();
        MnistExample.main(new String[]{directory.toString(), "--resume", original.toString(),
                "--output", output.toString(), "--epochs", "1"});
        assertArrayEquals(before, Files.readAllBytes(original));
        assertEquals(initialSteps + 1, ((ch.lolo.nn.optim.Adam) Sequential.load(output).optimizer()).stepCount());
    }

    private void writeData(Path directory, String prefix, int count) throws Exception {
        try (var images = new DataOutputStream(Files.newOutputStream(directory.resolve(prefix + "-images-idx3-ubyte")));
             var labels = new DataOutputStream(Files.newOutputStream(directory.resolve(prefix + "-labels-idx1-ubyte")))) {
            images.writeInt(2051); images.writeInt(count); images.writeInt(28); images.writeInt(28);
            labels.writeInt(2049); labels.writeInt(count);
            for (int i = 0; i < count; i++) {
                for (int pixel = 0; pixel < 784; pixel++) images.writeByte(pixel % 28 == 10 + i % 2 ? 255 : 0);
                labels.writeByte(i % 2);
            }
        }
    }
}
