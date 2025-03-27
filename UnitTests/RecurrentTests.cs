using superml.NETWORK;
using superml.NETWORK.LAYERS;
using superml.NETWORK.LAYERS.ACTIVATION.ACTIVATION_FUNCTION.LEAKY_RELU;
using superml.NETWORK.LAYERS.FLATTEN;
using superml.NETWORK.LAYERS.RECURRENT;
using superml.NETWORK.LAYERS.RECURRENT.RECURRENCY_TYPE;
using superml.NETWORK.LAYERS.SOFT_MAX;
using superml.NETWORK.OBJECTS.MATH_OBJECTS;

namespace UnitTests;

public class RecurrentTests {
    [Test]
    public void ForwardFeed_MTM() {
        var testTensorData = new Tensor(new Matrix(new double[] { 0, 0, 0, 1, 1, 1 }));
        var model = new Network(new List<ILayer> {
            new FlattenLayer(),
            new RecurrentLayer(new LeakyReLu(), RecurrencyType.ManyToMany, 10),
            new SoftMaxLayer()
        });
        var answer = model.ForwardFeed(testTensorData);

        var layers = model.GetLayers();
        Console.WriteLine($"Layer {layers.Count}:\nInput Tensor on layer:\n{layers[^1].GetValues().GetInfo()}\n");
    }

    [Test]
    public void ForwardFeed_MTO() {
        var testTensorData = new Tensor(new Matrix(new double[] { 0, 0, 0, 1, 1, 1 }));
        var model = new Network(new List<ILayer> {
            new FlattenLayer(),
            new RecurrentLayer(new LeakyReLu(), RecurrencyType.ManyToOne, 10),
            new SoftMaxLayer()
        });
        var answer = model.ForwardFeed(testTensorData);

        var layers = model.GetLayers();
        Console.WriteLine($"Layer {layers.Count}:\nInput Tensor on layer:\n{layers[^1].GetValues().GetInfo()}\n");
    }
    
    [Test]
    public void BackPropagation_MTM() {
        var testTensorData = new Tensor(new Matrix(new double[] { 0, 0, 0, 1, 1, 1 }));
        var model = new Network(new List<ILayer> {
            new FlattenLayer(),
            new RecurrentLayer(new LeakyReLu(), RecurrencyType.ManyToMany, 10),
            new SoftMaxLayer()
        });
        var answer = model.ForwardFeed(testTensorData);

        var layers = model.GetLayers();
        for (var layer = 0; layer < layers.Count; layer++) 
            Console.WriteLine($"(BEFORE) Layer {layer + 1}:\nInput Tensor on layer:\n{layers[layer].GetData()}\n");

        model.BackPropagation(0, 1, .15d);
        
        for (var layer = 0; layer < layers.Count; layer++) 
            Console.WriteLine($"(BEFORE) Layer {layer + 1}:\nInput Tensor on layer:\n{layers[layer].GetData()}\n");
    }
    
    [Test]
    public void BackPropagation_MTO() {
        var testTensorData = new Tensor(new Matrix(new double[] { 0, 0, 0, 1, 1, 1 }));
        var model = new Network(new List<ILayer> {
            new FlattenLayer(),
            new RecurrentLayer(new LeakyReLu(), RecurrencyType.ManyToOne, 10),
            new SoftMaxLayer()
        });
        var answer = model.ForwardFeed(testTensorData);

        var layers = model.GetLayers();
        for (var layer = 0; layer < layers.Count; layer++) 
            Console.WriteLine($"(BEFORE) Layer {layer + 1}:\nInput Tensor on layer:\n{layers[layer].GetData()}\n");

        model.BackPropagation(0, 1, .15d);
        
        for (var layer = 0; layer < layers.Count; layer++) 
            Console.WriteLine($"(BEFORE) Layer {layer + 1}:\nInput Tensor on layer:\n{layers[layer].GetData()}\n");
    }
}