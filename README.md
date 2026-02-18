# superml

superml is a C# library for composing and training neural networks from reusable layers. The project includes convolutional, recurrent, perceptron, and generative-adversarial components, along with data utilities for arrays, tensors, images, sound, and CSV files.

## Features

- Layer-based networks with activation, convolution, pooling, recurrent, dropout, flattening, normalization, and up-sampling layers
- Perceptron and convolution optimizers, including Adam implementations
- Weight initialization, loss functions, and regularization options
- Image and value classification and prediction models
- CSV, image, and sound data handling
- Unit tests for network layers and model behavior

## Requirements

Use the .NET SDK version specified in [`global.json`](global.json).

## Build

```sh
dotnet build superml.sln
```

## Compose a network

A network is created by passing an ordered collection of layers to `Network`:

```csharp
using superml.NETWORK;
using superml.NETWORK.LAYERS;
using superml.NETWORK.LAYERS.ACTIVATION.ACTIVATION_FUNCTION.RELU;

var layers = new List<ILayer>
{
    new FlattenLayer(),
    new PerceptronLayer(128),
    new ActivationLayer(new ReLu()),
    new PerceptronLayer(10),
    new SoftMaxLayer(),
};

var network = new Network(layers);
```

Layer constructors take configuration such as dimensions, activation functions, initialization methods, pooling types, and optimizers. Use `ForwardFeed` to run an input through the network and the `BackPropagation` overloads to update its weights from expected outputs or a supplied error. `Network.Fit` and `Network.Test` provide dataset-oriented training and evaluation workflows.

The `UnitTests` project contains additional examples of layer setup, data shapes, and model operations.
