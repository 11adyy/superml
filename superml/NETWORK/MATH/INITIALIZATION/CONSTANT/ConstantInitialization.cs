using superml.NETWORK.MATH.OBJECTS;

namespace superml.NETWORK.MATH.Initialization.CONSTANT;

public class ConstantInitialization : IWeightsInitialization {
    public ConstantInitialization(double value) => Value = value;
        
    private double Value { get; }
    
    public Matrix Initialize(Matrix matrix) {
        for (var i = 0; i < matrix.Rows; i++)
            for (var j = 0; j < matrix.Columns; j++)
                matrix.Body[i, j] = Value;

        return matrix;
    }
}