using superml.NETWORK.ACTIVATION.INTERFACES;
using superml.NETWORK.LAYERS.INTERFACES;
using superml.NETWORK.OBJECTS;

namespace superml.NETWORK.LAYERS.ACTIVATION {
    public class ActivationLayer : ILayer {
        public ActivationLayer(IFunction function) => Function = function;

        private IFunction Function { get; }

        public Tensor GetNextLayer(Tensor tensor) => Function.Activate(tensor);
        public Tensor BackPropagate(Tensor error) => Function.Derivation(error);

        public Tensor GetValues() => new Tensor(new Matrix(0,0));
        public string GetData() => "";
        public string LoadData(string data) => data;
    }
}