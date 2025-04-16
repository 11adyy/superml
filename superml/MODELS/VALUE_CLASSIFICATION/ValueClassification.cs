using superml.NETWORK;
using superml.NETWORK.LAYERS;
using superml.NETWORK.LAYERS.ACTIVATION.ACTIVATION_FUNCTION.HYPERBOLIC_TANGENT;
using superml.NETWORK.LAYERS.RECURRENT;
using superml.NETWORK.LAYERS.RECURRENT.RECURRENCY_TYPE.MANY_TO_ONE;
using superml.NETWORK.LAYERS.SOFT_MAX;
using superml.NETWORK.MATH.Initialization.Xavier;

namespace superml.MODELS.VALUE_CLASSIFICATION;

public static class ValueClassification {
    public static Network SimpleValueClassification = new Network(new List<ILayer> {
        new RecurrentLayer(new HyperbolicTangent(), new ManyToOne(), 10, new XavierInitialization()),
        new SoftMaxLayer()
    });
    
    public static Network DeepValueClassification = new Network(new List<ILayer> {
        new RecurrentLayer(new HyperbolicTangent(), new ManyToOne(), 100, new XavierInitialization()),
        new SoftMaxLayer()
    });
}