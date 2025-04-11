using System.Drawing;
using System.Drawing.Imaging;
using superml.MODELS.IMAGE_CLASSIFICATION;
using superml.NETWORK;
using superml.NETWORK.LAYERS;
using superml.NETWORK.LAYERS.ACTIVATION;
using superml.NETWORK.LAYERS.ACTIVATION.ACTIVATION_FUNCTION.DOUBLE_LEAKY_RELU;
using superml.NETWORK.LAYERS.ACTIVATION.ACTIVATION_FUNCTION.RELU;
using superml.NETWORK.LAYERS.CONVOLUTION;
using superml.NETWORK.LAYERS.DECONVOLUTION;
using superml.NETWORK.LAYERS.FLATTEN;
using superml.NETWORK.LAYERS.PERCEPTRON;
using superml.NETWORK.LAYERS.POOLING;
using superml.NETWORK.LAYERS.POOLING.SCRIPTS.MAX;
using superml.NETWORK.LAYERS.SOFT_MAX;
using superml.NETWORK.MATH.Initialization.HE;
using superml.NETWORK.MATH.LOSS_FUNCTION.ONE_BY_ONE;
using superml.NETWORK.OBJECTS.MATH_OBJECTS;
using superml.NETWORK.ROUGHEN;
using superml.SCRIPTS.REGION_CONVOLUTION;

namespace UnitTests;

public class NetworkTest {
    [Test]
    public void RCnnTest() {
        var bitmap = (Bitmap)Bitmap.FromFile(@"C://Users//11adyy//Desktop//RCNN_TEST//test.jpg");
        RegionConvolution.ForwardFeed(bitmap, 50, 3, CnnClassification.DeepConvolutionNetwork, .2, 28, 28)
            .Save(@$"D:\загрузки\{Guid.NewGuid()}.png", ImageFormat.Png);
    }

    [Test]
    public void CnnTest() {
        var model = new Network(new List<ILayer> {
            new ConvolutionLayer(8, 9,9,3, new HeInitialization(), 1),
            new ActivationLayer(new DoubleLeakyReLu()),
            new PoolingLayer(new MaxPooling(), 4),
            new ConvolutionLayer(16, 9, 9, 8, new HeInitialization(), 1),
            new ActivationLayer(new DoubleLeakyReLu()),
            new PoolingLayer(new MaxPooling(), 2),
            new FlattenLayer(),
            new PerceptronLayer(144, 128, new HeInitialization()),
            new ActivationLayer(new DoubleLeakyReLu()),
            new PerceptronLayer(128, 2, new HeInitialization()),
            new ActivationLayer(new DoubleLeakyReLu()),
            new PerceptronLayer(2),
            new SoftMaxLayer()
        });

        for (var i = 0; i < 100; i++)
            model.ForwardFeed(new Tensor(new Matrix(64, 64)), AnswerType.Class);
        
        for (var i = 0; i < 100; i++)
            model.BackPropagation(1,1,new OneByOne(), 1);
    }

    [Test]
    public void GeneratorTest() {
        var model = new Network(new List<ILayer> {
            new RoughenLayer(3,3,3),
            new DeconvolutionLayer(16, 2,2,3, new HeInitialization(), 2),
            new ActivationLayer(new ReLu()),
            new DeconvolutionLayer(8, 6, 6, 16, new HeInitialization(), 2),
            new ActivationLayer(new ReLu())
        });
        
        Console.WriteLine(model.ForwardFeed(new Vector(27).FillRandom().AsTensor(3,3,3)).Channels[0].Print());
    }
}