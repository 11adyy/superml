using superml.NETWORK;
using superml.NETWORK.LAYERS;
using superml.NETWORK.LAYERS.ACTIVATION;
using superml.NETWORK.LAYERS.ACTIVATION.ACTIVATION_FUNCTION.DOUBLE_LEAKY_RELU;
using superml.NETWORK.LAYERS.CONVOLUTION;
using superml.NETWORK.LAYERS.FLATTEN;
using superml.NETWORK.LAYERS.PERCEPTRON;
using superml.NETWORK.LAYERS.POOLING;
using superml.NETWORK.LAYERS.POOLING.SCRIPTS.MAX;
using superml.NETWORK.MATH.Initialization.HE;

namespace superml.MODELS.NUMBER_CLASSIFICATION;

public static class CnnClassification {
    public static Network SimpleConvolutionNetwork = new Network(new List<ILayer> {
        new ConvolutionLayer(6, 5,5,3, new HeInitialization(), 1),
        new ActivationLayer(new DoubleLeakyReLu()),
        new PoolingLayer(new MaxPooling(), 2),
        new ConvolutionLayer(16, 5, 5, 6, new HeInitialization(), 1),
        new ActivationLayer(new DoubleLeakyReLu()),
        new PoolingLayer(new MaxPooling(), 2),
        new FlattenLayer(),
        new PerceptronLayer(256, 128, new HeInitialization()),
        new ActivationLayer(new DoubleLeakyReLu()),
        new PerceptronLayer(128, 10, new HeInitialization()),
        new ActivationLayer(new DoubleLeakyReLu()),
        new PerceptronLayer(10)
    });
    
    public static Network DeepConvolutionNetwork = new Network(new List<ILayer> {
        new ConvolutionLayer(16, 5,5,3, new HeInitialization(), 1),
        new ActivationLayer(new DoubleLeakyReLu()),
        new ConvolutionLayer(32, 5, 5, 16, new HeInitialization(), 1),
        new ActivationLayer(new DoubleLeakyReLu()),
        new FlattenLayer(),
        new PerceptronLayer(512, 256, new HeInitialization()),
        new ActivationLayer(new DoubleLeakyReLu()),
        new PerceptronLayer(256, 128, new HeInitialization()),
        new ActivationLayer(new DoubleLeakyReLu()),
        new PerceptronLayer(128, 10, new HeInitialization()),
        new ActivationLayer(new DoubleLeakyReLu()),
        new PerceptronLayer(10)
    });
}