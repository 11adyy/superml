using superml.NETWORK.OBJECTS.MATH_OBJECTS;

namespace superml.NETWORK.OBJECTS.DATA_OBJECTS;

public interface IData {
    public Tensor GetRight();
    public Tensor AsTensor();
    
    public enum Type {
        Array,
        Image
    }
}