using superml.NETWORK.MATH.OBJECTS;

namespace superml.DATA.DATA_OBJECTS;

public interface IData {
    public Tensor GetRight();
    public Tensor AsTensor();
    
    public enum Type {
        Array,
        Image
    }
}