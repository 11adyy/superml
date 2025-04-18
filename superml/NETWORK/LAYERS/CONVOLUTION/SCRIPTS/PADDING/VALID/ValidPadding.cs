using superml.NETWORK.MATH.OBJECTS;

namespace superml.NETWORK.LAYERS.CONVOLUTION.SCRIPTS.PADDING.VALID;

public class ValidPadding : Padding {
    protected override Matrix GetPadding(Matrix matrix) => matrix;
}