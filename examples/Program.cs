using System;
using BenScr.NeuralNetworks;

var network = new DenseNetwork(2,
    [new LayerDefinition(4, Activation.Tanh), new LayerDefinition(1, Activation.Sigmoid)], seed: 42);
TrainingSample[] xor =
[
    new([0, 0], [0]), new([0, 1], [1]), new([1, 0], [1]), new([1, 1], [0])
];
TrainingResult result = network.Train(xor,
    new TrainingOptions { Epochs = 10_000, LearningRate = 0.5, Seed = 17, TargetLoss = 0.003 });
if (result.FinalLoss > 0.003)
{
    throw new InvalidOperationException("The XOR example did not reach its loss target.");
}
DenseNetwork restored = DenseNetwork.FromJson(network.ToJson());
foreach (TrainingSample sample in xor)
{
    double[] input = sample.GetInput();
    double prediction = restored.Predict(input)[0];
    if (Math.Abs(prediction - sample.GetTarget()[0]) > 0.12 || prediction != network.Predict(input)[0])
    {
        throw new InvalidOperationException("Model roundtrip or XOR classification failed.");
    }
    Console.WriteLine($"{input[0]} XOR {input[1]} -> {prediction:F4}");
}
var linear = new DenseNetwork(1, [new LayerDefinition(1, Activation.Linear)]);
linear.SetParameters([2, 1]);
if (linear.Predict([3])[0] != 7 || linear.ComputeGradient([3], [5])[0] != 12)
{
    throw new InvalidOperationException("Affine prediction or analytic gradient is incorrect.");
}
Console.WriteLine($"All package consumer scenarios passed. XOR MSE: {result.FinalLoss:G6}");
