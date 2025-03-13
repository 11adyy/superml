using superml.NETWORK.OBJECTS;

namespace superml.NETWORK.FIT.DATA_OBJECTS;

public interface IData {
    public Tensor GetRight();
    public Tensor AsTensor();
}