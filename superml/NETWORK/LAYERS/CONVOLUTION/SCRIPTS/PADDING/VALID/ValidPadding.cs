using superml.NETWORK.OBJECTS.MATH_OBJECTS;

namespace superml.NETWORK.LAYERS.CONVOLUTION.SCRIPTS.PADDING.VALID;

public class ValidPadding : Padding {
    protected override Matrix GetPadding(Matrix matrix, int paddingSize) => matrix;
}