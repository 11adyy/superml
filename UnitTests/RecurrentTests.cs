using superml.NETWORK;
using superml.NETWORK.LAYERS;
using superml.NETWORK.LAYERS.ACTIVATION.ACTIVATION_FUNCTION.DOUBLE_LEAKY_RELU;
using superml.NETWORK.LAYERS.ACTIVATION.ACTIVATION_FUNCTION.TANGENSOID;
using superml.NETWORK.LAYERS.FLATTEN;
using superml.NETWORK.LAYERS.PERCEPTRON;
using superml.NETWORK.LAYERS.RECURRENT;
using superml.NETWORK.LAYERS.RECURRENT.RECURRENCY_TYPE.ManyToMany;
using superml.NETWORK.LAYERS.RECURRENT.RECURRENCY_TYPE.ManyToOne;
using superml.NETWORK.LAYERS.RECURRENT.RECURRENCY_TYPE.OneToMany;
using superml.NETWORK.LAYERS.SOFT_MAX;
using superml.NETWORK.MATH.Initialization.HE;
using superml.NETWORK.MATH.Initialization.Xavier;
using superml.NETWORK.MATH.LOSS_FUNCTION.VALUE_BY_VALUE;
using superml.NETWORK.OBJECTS.MATH_OBJECTS;

namespace UnitTests;

public class RecurrentTests {
    [Test]
    public void ForwardFeed_MTM() {
        var testTensorData = new Tensor(new Matrix(new double[] { 0, 0, 0, 1, 1, 1 }));
        var model = new Network(new List<ILayer> {
            new FlattenLayer(),
            new RecurrentLayer(new DoubleLeakyReLu(), new ManyToMany(), 10, new HeInitialization()),
            new SoftMaxLayer()
        });
        model.ForwardFeed(testTensorData, AnswerType.Class);

        var layers = model.GetLayers();
        Console.WriteLine($"Layer {layers.Count}:\nInput Tensor on layer:\n{layers[^1].GetValues().Channels[0].Print()}\n");
    }

    [Test]
    public void ForwardFeed_MTO() {
        var testTensorData = new Tensor(new Matrix(new double[] { .9, .1, .1, .1, 1, 1 }));
        var model = new Network(new List<ILayer> {
            new FlattenLayer(),
            new RecurrentLayer(new DoubleLeakyReLu(), new ManyToOne(), 10, new HeInitialization()),
            new SoftMaxLayer()
        });
        model.ForwardFeed(testTensorData, AnswerType.Class);

        var layers = model.GetLayers();
        Console.WriteLine($"Layer {layers.Count}:\nInput Tensor on layer:\n{layers[^1].GetValues().Channels[0].Print()}\n");
    }
    
    [Test]
    public void ForwardFeed_OTM() {
        var testTensorData = new Tensor(new Matrix(new double[] { .012d }));
        var model = new Network(new List<ILayer> {
            new FlattenLayer(),
            new RecurrentLayer(new DoubleLeakyReLu(), new OneToMany(), 10, new HeInitialization()),
            new SoftMaxLayer()
        });
        model.ForwardFeed(testTensorData, AnswerType.Class);

        var layers = model.GetLayers();
        Console.WriteLine($"Layer {layers.Count}:\nInput Tensor on layer:\n{layers[^1].GetValues().Channels[0].Print()}\n");
    }
    
    [Test]
    public void BackPropagation_MTM() {
        var testTensorData = new Tensor(new Matrix(new double[] { 70, 10, 30, 21, 14, 77 }));
        var model = new Network(new List<ILayer> {
            new FlattenLayer(),
            new RecurrentLayer(new Tangensoid(), new ManyToMany(), 10, new XavierInitialization()),
            new SoftMaxLayer()
        });
        
        Console.WriteLine();
        Console.WriteLine(model.ForwardFeed(testTensorData, AnswerType.Value));
        Console.WriteLine(model.GetWeights());
        
        for (var i = 0; i < 10; i++) model.BackPropagation(0, 30000, new ValueByValue(), .015d);
        
        Console.WriteLine();
        Console.WriteLine(model.ForwardFeed(testTensorData, AnswerType.Value));
        Console.WriteLine(model.GetWeights());
    }
    
    [Test]
    public void BackPropagation_MTO() {
        var testTensorData = new Tensor(new Matrix(new double[] { 70, 10, 30, 21, 14, 77 }));
        var model = new Network(new List<ILayer> {
            new FlattenLayer(),
            new RecurrentLayer(new DoubleLeakyReLu(), new ManyToOne(), 10, new XavierInitialization()),
            new PerceptronLayer(1)
        });

        Console.WriteLine();
        Console.WriteLine(model.ForwardFeed(testTensorData, AnswerType.Value));
        Console.WriteLine(model.GetWeights());
        
        for (var i = 0; i < 1; i++) model.BackPropagation(0, 30000, new ValueByValue(), .00015d);
        
        Console.WriteLine();
        Console.WriteLine(model.ForwardFeed(testTensorData, AnswerType.Value));
        Console.WriteLine(model.GetWeights());
    }
    
    [Test]
    public void BackPropagation_OTM() {
        var testTensorData = new Tensor(new Matrix(new double[] { .12d }));
        var model = new Network(new List<ILayer> {
            new FlattenLayer(),
            new RecurrentLayer(new DoubleLeakyReLu(), new OneToMany(), 10, new HeInitialization()),
            new SoftMaxLayer()
        });
        
        Console.WriteLine();
        Console.WriteLine(model.ForwardFeed(testTensorData, AnswerType.Value));
        Console.WriteLine(model.GetWeights());
        
        for (var i = 0; i < 10; i++) model.BackPropagation(0, 30000, new ValueByValue(), .015d);
        
        Console.WriteLine();
        Console.WriteLine(model.ForwardFeed(testTensorData, AnswerType.Value));
        Console.WriteLine(model.GetWeights());
    }
}