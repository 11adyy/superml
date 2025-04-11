using superml.NETWORK.OBJECTS.MATH_OBJECTS;

namespace superml.NETWORK.LAYERS {
    public interface ILayer {
        public Tensor GetNextLayer(Tensor tensor);
        public Tensor BackPropagate(Tensor error, double learningRate, bool backPropagate);
        public Tensor GetValues();
        public string GetData();
        public string LoadData(string data);
    }
}