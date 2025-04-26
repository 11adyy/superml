using superml.NETWORK.MATH.OBJECTS;

namespace superml.NETWORK.LAYERS.NOISE.SCRIPTS;

public interface INoise {
    public Vector GenerateNoise(int size);
}