using superml.NETWORK.OBJECTS.MATH_OBJECTS;

namespace superml.NETWORK.MATH.Initialization;

public interface IWeightsInitialization {
    public Matrix Initialize(Matrix matrix);
}