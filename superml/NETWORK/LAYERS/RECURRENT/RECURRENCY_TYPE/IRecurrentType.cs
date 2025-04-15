using superml.NETWORK.MATH.OBJECTS;

namespace superml.NETWORK.LAYERS.RECURRENT.RECURRENCY_TYPE;

public interface IRecurrentType {
    public Tensor GetNextLayer(RecurrentLayer layer, Tensor tensor);
    public Tensor BackPropagate(RecurrentLayer layer, Tensor error, double learningRate);
}