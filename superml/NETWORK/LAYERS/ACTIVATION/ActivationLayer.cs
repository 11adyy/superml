using superml.NETWORK.LAYERS.ACTIVATION.ACTIVATION_FUNCTION;
using superml.NETWORK.OBJECTS.MATH_OBJECTS;

namespace superml.NETWORK.LAYERS.ACTIVATION {
    public class ActivationLayer : ILayer {
        public ActivationLayer(Function function) => Function = function;

        private Function Function { get; }

        public Tensor GetNextLayer(Tensor tensor) => Function.Activate(tensor);
        
        public Tensor BackPropagate(Tensor error, double learningRate) => Function.Derivation(error);

        public Tensor GetValues() => null!;
        
        public string GetData() => "";
        
        public string LoadData(string data) => data;
    }
}