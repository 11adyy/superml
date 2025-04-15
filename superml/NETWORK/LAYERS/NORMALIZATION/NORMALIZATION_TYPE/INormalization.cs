using superml.NETWORK.MATH.OBJECTS;

namespace superml.NETWORK.LAYERS.NORMALIZATION.NORMALIZATION_TYPE;

public interface INormalization {
    public Tensor Normalize(Tensor tensor);
}