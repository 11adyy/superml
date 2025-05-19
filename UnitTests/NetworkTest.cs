using System.Drawing;
using System.Drawing.Imaging;
using superml.DATA.IMAGE;
using superml.MODELS.IMAGE_CLASSIFICATION;
using superml.NETWORK;
using superml.NETWORK.LAYERS;
using superml.NETWORK.LAYERS.ACTIVATION;
using superml.NETWORK.LAYERS.ACTIVATION.ACTIVATION_FUNCTION.DOUBLE_LEAKY_RELU;
using superml.NETWORK.LAYERS.ACTIVATION.ACTIVATION_FUNCTION.PRELU;
using superml.NETWORK.LAYERS.ACTIVATION.ACTIVATION_FUNCTION.RELU;
using superml.NETWORK.LAYERS.ACTIVATION.ACTIVATION_FUNCTION.SIGMOID;
using superml.NETWORK.LAYERS.CONVOLUTION;
using superml.NETWORK.LAYERS.CONVOLUTION.ADAM.ADAM_CONVOLUTION;
using superml.NETWORK.LAYERS.CONVOLUTION.SCRIPTS.PADDING.SAME;
using superml.NETWORK.LAYERS.CONVOLUTION.SCRIPTS.PADDING.VALID;
using superml.NETWORK.LAYERS.DATA;
using superml.NETWORK.LAYERS.FLATTEN;
using superml.NETWORK.LAYERS.NOISE;
using superml.NETWORK.LAYERS.NOISE.SCRIPTS.GAUSSIAN;
using superml.NETWORK.LAYERS.NORMALIZATION;
using superml.NETWORK.LAYERS.NORMALIZATION.NORMALIZATION_TYPE.ABS;
using superml.NETWORK.LAYERS.NORMALIZATION.NORMALIZATION_TYPE.MIN_MAX;
using superml.NETWORK.LAYERS.PERCEPTRON;
using superml.NETWORK.LAYERS.PERCEPTRON.ADAM.ADAM_PERCEPTRON;
using superml.NETWORK.LAYERS.PERCEPTRON.ADAM.DEFAULT_PERCEPTRON;
using superml.NETWORK.LAYERS.POOLING;
using superml.NETWORK.LAYERS.POOLING.SCRIPTS.MAX;
using superml.NETWORK.LAYERS.ROUGHEN;
using superml.NETWORK.LAYERS.SOFT_MAX;
using superml.NETWORK.LAYERS.TRANSPOSED_CONVOLUTION;
using superml.NETWORK.LAYERS.TRANSPOSED_CONVOLUTION.ADAM.DEFAULT_TRANSPOSED_CONVOLUTION;
using superml.NETWORK.LAYERS.UP_SAMPLING;
using superml.NETWORK.LAYERS.UP_SAMPLING.UP_SAMPLING_TYPE.NEAREST_NEIGHBOR;
using superml.NETWORK.MATH.Initialization.HE;
using superml.NETWORK.MATH.LOSS_FUNCTION.MSE;
using superml.NETWORK.MATH.OBJECTS;
using superml.SCRIPTS.GENERATIVE_ADVERSARIAL_NETWORK;
using superml.SCRIPTS.GENERATIVE_ADVERSARIAL_NETWORK.IMAGES;
using superml.SCRIPTS.REGION_CONVOLUTION;

namespace UnitTests;

public class NetworkTest {
    [Test]
    public void RCnnTest() {
        var bitmap = (Bitmap)Image.FromFile(@"C://Users//11adyy//Desktop//RCNN_TEST//test.jpg");
        RegionConvolution.ForwardFeed(bitmap, 50, 3, CnnClassification.DeepConvolutionNetwork, .2, 28, 28)
            .Save(@$"D:\загрузки\{Guid.NewGuid()}.png", ImageFormat.Png);
    }

    [Test]
    public void CnnTest() {
        var model = new Network(new List<ILayer> {
            new ConvolutionLayer(8, (10, 10, 3), new HeInitialization(), 2, new ValidPadding(), new AdamConvolutionOptimization((.9d,.99d,1e-8))),
            new ActivationLayer(new DoubleLeakyReLu()),
            new PoolingLayer(new MaxPooling(), 2),
            new ConvolutionLayer(16, (3, 3, 8), new HeInitialization(), 2, new ValidPadding(), new AdamConvolutionOptimization((.9d, .99d, 1e-8))),
            new ActivationLayer(new DoubleLeakyReLu()),
            new PoolingLayer(new MaxPooling(), 2),
            new FlattenLayer(),
            new PerceptronLayer(144, 100, new HeInitialization(), new AdamPerceptronOptimization()),
            new ActivationLayer(new DoubleLeakyReLu()),
            new PerceptronLayer(100, 2, new HeInitialization(), new AdamPerceptronOptimization()),
            new ActivationLayer(new DoubleLeakyReLu()),
            new PerceptronLayer(2),
            new SoftMaxLayer()
        });

        for (var i = 0; i < 1; i++) {
            model.ForwardFeed(new Tensor(new Matrix(64, 64)), AnswerType.Class);
            model.BackPropagation(1,1,new Mse(), 1, true);
        }
        
        //Console.WriteLine(model.ForwardFeed(new Tensor(new Matrix(64, 64)), AnswerType.Class));
    }

    [Test]
    public void GeneratorTest() {
        var model = new Network(new List<ILayer> {
            new RoughenLayer(3,3,32),
            new TransposedConvolutionLayer(16, (2,2,32), new HeInitialization(), 2, new NoTransposedConvolutionOptimization()),
            new ActivationLayer(new ReLu()),
            new TransposedConvolutionLayer(8, (6, 6, 16), new HeInitialization(), 2, new NoTransposedConvolutionOptimization()),
            new ActivationLayer(new ReLu()),
            new TransposedConvolutionLayer(4, (6, 6, 8), new HeInitialization(), 2, new NoTransposedConvolutionOptimization()),
            new ActivationLayer(new ReLu()),
            new TransposedConvolutionLayer(3, (6, 6, 4), new HeInitialization(), 2, new NoTransposedConvolutionOptimization()),
            new ActivationLayer(new ReLu()),
            new DataLayer(DataType.InputTensor)
        });
        
        var errorTensor = new Tensor(new List<Matrix> { new (76, 76), new (76, 76), new (76, 76) });
        model.BackPropagation(errorTensor, new Mse(), .1, true);
    }

    [Test]
    public void GanTest() {
        const string path = @"C://Users//11adyy//Desktop//RCNN_TEST//";

        var generator = new Network(new List<ILayer> {
            new NoiseLayer(128, new GaussianNoise()),
            new PerceptronLayer(128, 4800, new HeInitialization(), new AdamPerceptronOptimization()),
            new ActivationLayer(new PReLu(.2d)),
            new RoughenLayer(40,40,3),
            new NormalizationLayer(new Abs()),
            new NormalizationLayer(new MinMax(1)),
            new DataLayer(DataType.InputTensor)
        });
        
        var discriminator = new Network(new List<ILayer> {
            new PerceptronLayer(4800, 2, new HeInitialization(), new AdamPerceptronOptimization()),
            new ActivationLayer(new DoubleLeakyReLu()),
            new PerceptronLayer(2),
            new SoftMaxLayer()
        });

        var network = new ImageGaNetwork(generator, discriminator);
        network.DiscriminatorFitting(1, ImageGaNetwork.LoadReal(path + "faces", 40, 40), .05d);
        network.GeneratorFitting(1000, .5d, 1, @$"C://Users//11adyy//Desktop//RCNN_TEST//answers//{Guid.NewGuid()}.png");
    }

    [Test]
    public void GeneratorBackPropTest() {
        const string path = @"C://Users//11adyy//Desktop//RCNN_TEST//";
        
        var generator = new Network(new List<ILayer> {
            new NoiseLayer(128, new GaussianNoise()),
            new PerceptronLayer(128, 324, new HeInitialization(), new NoPerceptronOptimization()),
            new ActivationLayer(new PReLu(.2d)),
            new RoughenLayer(6,6,9),
            new UpSamplingLayer(new NearestNeighbor(), 2),
            new FlattenLayer(),
            new PerceptronLayer(1296, 2100, new HeInitialization(), new NoPerceptronOptimization()),
            new ActivationLayer(new PReLu(.2d)),
            new PerceptronLayer(2100, 4800, new HeInitialization(), new NoPerceptronOptimization()),
            new ActivationLayer(new Sigmoid()),
            new RoughenLayer(40,40,3),
            new NormalizationLayer(new Abs()),
            new NormalizationLayer(new MinMax(1)),
            new DataLayer(DataType.InputTensor)
        });
        
        var generator1 = new Network(new List<ILayer> {
            new NoiseLayer(144, new GaussianNoise()),
            new RoughenLayer(4,4,9),
            new TransposedConvolutionLayer(6,(4,4,9), new HeInitialization(), 1, new NoTransposedConvolutionOptimization()),
            new ActivationLayer(new PReLu(.2d)),
            new TransposedConvolutionLayer(3,(12,12,6), new HeInitialization(), 1, new NoTransposedConvolutionOptimization()),
            new ActivationLayer(new PReLu(.2d)),
            new TransposedConvolutionLayer(3,(23,23,6), new HeInitialization(), 1, new NoTransposedConvolutionOptimization()),
            new ActivationLayer(new Sigmoid()),
            new NormalizationLayer(new Abs()),
            new NormalizationLayer(new MinMax(1)),
            new DataLayer(DataType.InputTensor)
        });
        
        var generator2 = new Network(new List<ILayer> {
            new NoiseLayer(144, new GaussianNoise()),
            new RoughenLayer(4,4,9),
            new UpSamplingLayer(new NearestNeighbor(), 2), 
            new ConvolutionLayer(6, (3, 3, 9), new HeInitialization(), 1, new SamePadding(new Tensor(3,3,9)), new AdamConvolutionOptimization((.9d,.99d,1e-8))),
            new UpSamplingLayer(new NearestNeighbor(), 2),
            new ConvolutionLayer(3, (3, 3, 6), new HeInitialization(), 1, new SamePadding(new Tensor(3,3,6)), new AdamConvolutionOptimization((.9d,.99d,1e-8))),
            new UpSamplingLayer(new NearestNeighbor(), 2), 
            new NormalizationLayer(new Abs()), 
            new NormalizationLayer(new MinMax(1)),
            new DataLayer(DataType.InputTensor)
        });
        
        for (var i = 0; i < 1000; i++) {
            var answer = generator.ForwardFeed(null!);
            
            if (i % 1 == 0)
                Parser.TensorToImage(answer).Save(@$"C://Users//11adyy//Desktop//RCNN_TEST//answers//2.jpg", ImageFormat.Png);
            generator.BackPropagation(Parser.ImageToTensor(new Bitmap((Bitmap)Bitmap.FromFile(@"C://Users//11adyy//Desktop//RCNN_TEST//answers//Untitled.png"), new Size(40,40))),
                new Mse(), .01, true);
        }
        
    }
}