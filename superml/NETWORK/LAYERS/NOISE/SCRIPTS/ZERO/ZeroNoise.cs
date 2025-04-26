using superml.NETWORK.MATH.OBJECTS;

namespace superml.NETWORK.LAYERS.NOISE.SCRIPTS.ZERO;

public class ZeroNoise : INoise {
    public Vector GenerateNoise(int size) {
        var body = new double[size];

        for (var i = 0; i < size; i++)
            body[i] = 0;
        
        return new Vector(body);
    }
}