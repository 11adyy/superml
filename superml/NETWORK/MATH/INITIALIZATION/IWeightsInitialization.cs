using superml.NETWORK.MATH.OBJECTS;

namespace superml.NETWORK.MATH.Initialization;

public interface IWeightsInitialization {
    /// <summary>
    /// Initialization method for creating start values
    /// </summary>
    /// <param name="matrix"> Matrix for initialization </param>
    /// <returns> Matrix after initialization </returns>
    public Matrix Initialize(Matrix matrix);
}