using superml.NETWORK;
using superml.NETWORK.LAYERS;
using superml.NETWORK.LAYERS.ACTIVATION;
using superml.NETWORK.LAYERS.ACTIVATION.ACTIVATION_FUNCTION.DOUBLE_LEAKY_RELU;
using superml.NETWORK.LAYERS.PERCEPTRON;
using superml.NETWORK.MATH.Initialization.HE;

namespace superml.MODELS.VALUE_PREDICTION;

public static class ClassicValuePrediction {
    public static Network SimpleValuePrediction = new Network(new List<ILayer> {
        new PerceptronLayer(10, 128, new HeInitialization()),
        new ActivationLayer(new DoubleLeakyReLu()),
        new PerceptronLayer(128, 128, new HeInitialization()),
        new ActivationLayer(new DoubleLeakyReLu()),
        new PerceptronLayer(128, 1, new HeInitialization()),
        new PerceptronLayer(1)
    });
    
    public static Network DeepValuePrediction = new Network(new List<ILayer> {
        new PerceptronLayer(10, 128, new HeInitialization()),
        new ActivationLayer(new DoubleLeakyReLu()),
        new PerceptronLayer(128, 128, new HeInitialization()),
        new ActivationLayer(new DoubleLeakyReLu()),
        new PerceptronLayer(128, 128, new HeInitialization()),
        new ActivationLayer(new DoubleLeakyReLu()),
        new PerceptronLayer(128, 128, new HeInitialization()),
        new ActivationLayer(new DoubleLeakyReLu()),
        new PerceptronLayer(128, 128, new HeInitialization()),
        new ActivationLayer(new DoubleLeakyReLu()),
        new PerceptronLayer(128, 1, new HeInitialization()),
        new PerceptronLayer(1)
    });
}