using System.Drawing;
using superml.DATA.IMAGE;
using superml.NETWORK;
using superml.NETWORK.LAYERS;
using superml.NETWORK.LAYERS.ACTIVATION;
using superml.NETWORK.LAYERS.ACTIVATION.ACTIVATION_FUNCTION.DOUBLE_LEAKY_RELU;
using superml.NETWORK.LAYERS.CONVOLUTION;
using superml.NETWORK.LAYERS.DATA;
using superml.NETWORK.LAYERS.DECONVOLUTION;
using superml.NETWORK.LAYERS.FLATTEN;
using superml.NETWORK.LAYERS.PERCEPTRON;
using superml.NETWORK.LAYERS.POOLING;
using superml.NETWORK.LAYERS.POOLING.SCRIPTS.MAX;
using superml.NETWORK.MATH.Initialization.HE;
using superml.NETWORK.MATH.Initialization.Xavier;
using superml.NETWORK.MATH.LOSS_FUNCTION.ONE_BY_ONE;
using superml.NETWORK.OBJECTS.MATH_OBJECTS;
using superml.NETWORK.ROUGHEN;

namespace superml.SCRIPTS.GENERATIVE_ADVERSARIAL_NETWORK;

public class Network {
    public NETWORK.Network Generator = new NETWORK.Network(new List<ILayer> {
        new RoughenLayer(4,4,9),
        new DeconvolutionLayer(6, 5, 5, 9, new XavierInitialization(), 2),
        new ActivationLayer(new DoubleLeakyReLu()),
        new DeconvolutionLayer(3, 8, 8, 6, new XavierInitialization(), 2),
        new ActivationLayer(new DoubleLeakyReLu()),
        new DataLayer(DataType.InputTensor)
    });

    public NETWORK.Network Discriminator = new NETWORK.Network(new List<ILayer> {
        new DataLayer(DataType.ErrorTensor),
        new ConvolutionLayer(3, 3, 3, 3, new HeInitialization(), 1),
        new ActivationLayer(new DoubleLeakyReLu()),
        new PoolingLayer(new MaxPooling(), 2),
        new ConvolutionLayer(6, 6, 6, 3,new HeInitialization(), 1),
        new ActivationLayer(new DoubleLeakyReLu()),
        new PoolingLayer(new MaxPooling(), 2),
        new FlattenLayer(),
        new PerceptronLayer(96, 48, new HeInitialization()),
        new ActivationLayer(new DoubleLeakyReLu()),
        new PerceptronLayer(48, 24, new HeInitialization()),
        new ActivationLayer(new DoubleLeakyReLu()),
        new PerceptronLayer(24, 12, new HeInitialization()),
        new ActivationLayer(new DoubleLeakyReLu()),
        new PerceptronLayer(12, 2, new HeInitialization()),
        new ActivationLayer(new DoubleLeakyReLu()),
        new PerceptronLayer(2)
    });

    public List<Tensor> GenerateFake(int count) {
        var fake = new List<Tensor>();
        for (var i = 0; i < count; i++) 
            fake.Add(Generator.ForwardFeed(Vector.GenerateGaussianNoise(144).AsTensor(4,4,9)));
        
        return fake;
    }

    public List<Tensor> LoadReal(string directoryPath) {
        var files = Directory.GetFiles(directoryPath);
        return files.Select(file => Parser.ImageToTensor
            (new Bitmap((Bitmap)Bitmap.FromFile(file), new Size(28, 28)))).ToList();
    }
    
    public void DiscriminatorFitting(List<Tensor> realDataSet, List<Tensor> fakeDataSet, double learningRate) {
        for (var i = 0; i < Math.Min(realDataSet.Count, fakeDataSet.Count); i++) {
            switch (new Random().Next() % 100 > 50) {
                case true: // load real 1
                    if (Math.Abs(Discriminator.ForwardFeed(realDataSet[i], AnswerType.Class) - 1) > .1) 
                        Discriminator.BackPropagation(1, 1, new OneByOne(), learningRate, true);
                    break;
                case false: // load fake 0
                    if (Discriminator.ForwardFeed(realDataSet[i], AnswerType.Class) != 0) 
                        Discriminator.BackPropagation(0, 1, new OneByOne(), learningRate, true);
                    break;
            }
        }
    }

    public void GeneratorFitting(int epochs, double learningRate) {
        for (var i = 0; i < epochs; i++) {
            var generatedData = Generator.ForwardFeed(Vector.GenerateGaussianNoise(256).AsTensor(4, 4, 16));
            Discriminator.ForwardFeed(generatedData);
            Discriminator.BackPropagation(1,1, new OneByOne(), learningRate, false);
            var error = Discriminator.GetLayers()[0].GetValues();
            Generator.BackPropagation(error, learningRate, true);
        }
    }

    public Bitmap GenerateTensor() => Parser.TensorToImage(Generator.ForwardFeed(Vector.GenerateGaussianNoise(256).AsTensor(4, 4, 16)));
}