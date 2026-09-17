package ch.lolo.nn.layers;

import ch.lolo.nn.Layer;
import ch.lolo.nn.autograd.Variable;

public final class Flatten implements Layer {
    public Variable forward(Variable x, boolean t) {
        int[] s = x.value().shape();
        if (s.length <= 1) return x;
        int batch = s[0], rest = 1;
        for (int i = 1; i < s.length; i++) rest *= s[i];
        return x.reshape(batch, rest);
    }

    public String type() {
        return "Flatten";
    }
}
