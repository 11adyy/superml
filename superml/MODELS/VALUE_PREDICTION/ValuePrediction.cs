using superml.NETWORK;
using superml.NETWORK.LAYERS;
using superml.NETWORK.LAYERS.ACTIVATION.ACTIVATION_FUNCTION.TANGENSOID;
using superml.NETWORK.LAYERS.RECURRENT;
using superml.NETWORK.LAYERS.RECURRENT.RECURRENCY_TYPE.MANY_TO_ONE;
using superml.NETWORK.LAYERS.SOFT_MAX;
using superml.NETWORK.MATH.Initialization.Xavier;

namespace superml.MODELS.VALUE_PREDICTION;

public static class ValuePrediction {
    public static Network SimpleValuePrediction = new Network(new List<ILayer> {
        new RecurrentLayer(new Tangensoid(), new ManyToOne(), 10, new XavierInitialization()),
        new SoftMaxLayer()
    });
    
    public static Network DeepValuePrediction = new Network(new List<ILayer> {
        new RecurrentLayer(new Tangensoid(), new ManyToOne(), 100, new XavierInitialization()),
        new SoftMaxLayer()
    });
}