using superml.NETWORK.MATH.OBJECTS;

namespace superml.NETWORK.MATH.Initialization;

public interface IWeightsInitialization {
    public Matrix Initialize(Matrix matrix);
}