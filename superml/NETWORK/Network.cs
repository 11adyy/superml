using superml.NETWORK.ACTIVATION;
using superml.NETWORK.LAYERS.INTERFACES;
using superml.NETWORK.MATH;
using superml.NETWORK.OBJECTS;

namespace superml.NETWORK {
    public class Network {
        public Network(List<ILayer> layers, Function lossFunction) {
            Layers       = layers;
            MainFunction = lossFunction;
        }

        private List<ILayer> Layers { get; }
        private Function MainFunction { get; }

        public Tensor GetLayerData(int layer) => Layers[layer].GetValues();
        
        public int ForwardFeed(Tensor data) {
            try {
                data = Layers.Aggregate(data, (current, layer) => layer.GetNextLayer(current));
                return data.GetMaxIndex();
            }
            catch (Exception) {
                Console.WriteLine("Код ошибки: 1n");
                return 0;
            }
        }

        public void BackPropagation(double expectedAnswer) {
            try {
                var errorTensor = LossFunction.GetErrorTensor(Layers[^1].GetValues(), (int)expectedAnswer, MainFunction);
                for (var i = Layers.Count - 1; i >= 0; i--)
                    errorTensor = Layers[i].BackPropagate(errorTensor);
            }
            catch (Exception) {
                Console.WriteLine("Код ошибки: 2n");
            }
        }
    }
}