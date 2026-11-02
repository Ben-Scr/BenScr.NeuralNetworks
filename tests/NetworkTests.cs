using System;
using System.Linq;
using System.Threading.Tasks;
using Xunit;

namespace BenScr.NeuralNetworks.Tests;

public sealed class NetworkTests
{
    [Fact]
    public void AffineForwardUsesOutputMajorWeightsThenBiases()
    {
        var network = new DenseNetwork(2, [new(2, Activation.Linear)]);
        network.SetParameters([2, 3, -1, 4, 5, -2]);
        Assert.Equal([18d, 8d], network.Predict([2, 3]));
        Assert.Equal(4.5, network.EvaluateLoss([2, 3], [15, 8]));
    }

    [Theory]
    [InlineData(Activation.Linear, -2.0)]
    [InlineData(Activation.Relu, 0.0)]
    [InlineData(Activation.Sigmoid, 0.11920292202211755)]
    [InlineData(Activation.Tanh, -0.9640275800758169)]
    public void ActivationsMatchIndependentKnownValues(Activation activation, double expected)
    {
        var network = new DenseNetwork(1, [new(1, activation)]);
        network.SetParameters([1, 0]);
        Assert.Equal(expected, network.Predict([-2])[0], 14);
    }

    [Theory]
    [InlineData(-1000, 0)]
    [InlineData(1000, 1)]
    public void LogisticIsStableForExtremeFiniteInputs(double input, double expected)
    {
        var network = new DenseNetwork(1, [new(1, Activation.Sigmoid)]);
        network.SetParameters([1, 0]);
        Assert.Equal(expected, network.Predict([input])[0]);
    }

    [Theory]
    [InlineData(Activation.Linear)]
    [InlineData(Activation.Tanh)]
    [InlineData(Activation.Sigmoid)]
    [InlineData(Activation.Relu)]
    public void BackpropagationMatchesCentralFiniteDifferences(Activation activation)
    {
        var network = new DenseNetwork(2, [new(3, activation), new(2, activation)], seed: 17);
        double[] parameters = network.GetParameters();
        // Positive biases keep ReLU's test points away from its nondifferentiable corner.
        parameters[6] = 1.1;
        parameters[7] = 0.9;
        parameters[8] = 1.2;
        parameters[^2] = 1.0;
        parameters[^1] = 1.5;
        network.SetParameters(parameters);
        double[] input = [0.2, -0.3];
        double[] target = [0.4, -0.2];
        double[] analytic = network.ComputeGradient(input, target);
        const double Step = 1e-6;
        for (int index = 0; index < parameters.Length; index++)
        {
            double old = parameters[index];
            parameters[index] = old + Step;
            network.SetParameters(parameters);
            double plus = network.EvaluateLoss(input, target);
            parameters[index] = old - Step;
            network.SetParameters(parameters);
            double minus = network.EvaluateLoss(input, target);
            parameters[index] = old;
            double numerical = (plus - minus) / (2 * Step);
            Assert.True(Math.Abs(numerical - analytic[index]) < 1e-7 * (1 + Math.Abs(numerical)),
                $"Parameter {index}: analytic={analytic[index]}, numeric={numerical}");
        }
    }

    [Fact]
    public void ReluGradientAtZeroIsZero()
    {
        var network = new DenseNetwork(1, [new(1, Activation.Relu)]);
        network.SetParameters([0, 0]);
        Assert.Equal([0d, 0d], network.ComputeGradient([1], [1]));
    }

    [Fact]
    public void InitializationIsReproducibleAndDoesNotAliasDefinitionsOrParameters()
    {
        LayerDefinition[] definitions = [new(2, Activation.Tanh), new(1, Activation.Linear)];
        var first = new DenseNetwork(2, definitions, 42);
        var second = new DenseNetwork(2, definitions, 42);
        Assert.Equal(first.GetParameters(), second.GetParameters());
        Assert.NotEqual(first.GetParameters(), new DenseNetwork(2, definitions, 43).GetParameters());
        definitions[0] = new LayerDefinition(100, Activation.Relu);
        Assert.Equal(2, first.Layers[0].Units);
        double[] copy = first.GetParameters();
        copy[0] = 500;
        Assert.NotEqual(500, first.GetParameters()[0]);
        first.SetParameters(copy);
        copy[0] = 100;
        Assert.Equal(500, first.GetParameters()[0]);
    }

    [Fact]
    public void InitialWeightsAreGlorotBoundedAndBiasesAreZero()
    {
        var network = new DenseNetwork(3, [new(4, Activation.Tanh)]);
        double[] values = network.GetParameters();
        foreach (double weight in values.Take(12))
        {
            Assert.InRange(weight, -Math.Sqrt(6.0 / 7), Math.Sqrt(6.0 / 7));
        }
        Assert.All(values.Skip(12), bias => Assert.Equal(0, bias));
    }

    [Fact]
    public void InitializationUsesThePublishedSplitMix64SeedZeroVector()
    {
        var network = new DenseNetwork(1, [new(1, Activation.Linear)], seed: 0);
        // SplitMix64's first output from an all-zero state is E220A8397B1DCDAF.
        double uniform = (0xE220A8397B1DCDAFUL >> 11) / 9007199254740992.0;
        Assert.Equal((2 * uniform - 1) * Math.Sqrt(3), network.GetParameters()[0]);
    }

    [Fact]
    public void MaximumInputWidthAndLayerCountAreInclusive()
    {
        var wide = new DenseNetwork(DenseNetwork.MaxWidth, [new(1, Activation.Linear)]);
        Assert.Equal(0, wide.Predict(new double[DenseNetwork.MaxWidth])[0]);
        var deep = new DenseNetwork(1, Enumerable.Repeat(new LayerDefinition(1, Activation.Linear), DenseNetwork.MaxLayers).ToArray());
        Assert.Equal(DenseNetwork.MaxLayers, deep.Layers.Count);
        Assert.Equal(0, deep.Predict([0])[0]);
    }

    [Fact]
    public void ReadOnlyPredictionIsConcurrentAndIndependent()
    {
        var network = new DenseNetwork(2, [new(5, Activation.Tanh), new(1, Activation.Sigmoid)]);
        double[] expected = network.Predict([0.2, 0.7]);
        Parallel.For(0, 200, _ => Assert.Equal(expected, network.Predict([0.2, 0.7])));
        expected[0] = 100;
        Assert.NotEqual(100, network.Predict([0.2, 0.7])[0]);
    }

    [Fact]
    public void TrainingSampleCopiesAndValidatesItsVectors()
    {
        double[] input = [1, 2];
        double[] target = [3];
        var sample = new TrainingSample(input, target);
        input[0] = 9;
        target[0] = 9;
        sample.GetInput()[0] = 10;
        sample.GetTarget()[0] = 10;
        Assert.Equal([1d, 2d], sample.GetInput());
        Assert.Equal([3d], sample.GetTarget());
        Assert.Throws<ArgumentException>(() => new TrainingSample([], [1]));
        Assert.Throws<ArgumentException>(() => new TrainingSample([1], []));
        Assert.Throws<ArgumentException>(() => new TrainingSample([double.NaN], [1]));
        Assert.Throws<ArgumentException>(() => new TrainingSample([1], [double.PositiveInfinity]));
    }

    [Fact]
    public void InvalidArchitectureIsRejectedBeforeParameterAllocation()
    {
        Assert.Throws<ArgumentNullException>(() => new DenseNetwork(1, null!));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DenseNetwork(0, [new(1, Activation.Linear)]));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DenseNetwork(DenseNetwork.MaxWidth + 1, [new(1, Activation.Linear)]));
        Assert.Throws<ArgumentException>(() => new DenseNetwork(1, []));
        Assert.Throws<ArgumentException>(() => new DenseNetwork(1, [null!]));
        Assert.Throws<ArgumentException>(() => new DenseNetwork(1, [new(0, Activation.Linear)]));
        Assert.Throws<ArgumentException>(() => new DenseNetwork(1, [new(1, (Activation)100)]));
        Assert.Throws<ArgumentException>(() => new DenseNetwork(1, Enumerable.Repeat(new LayerDefinition(1, Activation.Linear), 33).ToArray()));
        Assert.Throws<ArgumentException>(() => new DenseNetwork(4096, [new(4096, Activation.Linear)]));
    }

    [Fact]
    public void InvalidVectorsCannotMutateParameters()
    {
        var network = new DenseNetwork(1, [new(1, Activation.Linear)]);
        double[] before = network.GetParameters();
        Assert.Throws<ArgumentException>(() => network.SetParameters([1]));
        Assert.Throws<ArgumentException>(() => network.SetParameters([1, double.NaN]));
        Assert.Equal(before, network.GetParameters());
        Assert.Throws<ArgumentException>(() => network.Predict([]));
        Assert.Throws<ArgumentException>(() => network.Predict([double.PositiveInfinity]));
        Assert.Throws<ArgumentException>(() => network.EvaluateLoss([1], []));
        Assert.Throws<ArgumentException>(() => network.ComputeGradient([1], [double.NaN]));
    }

    [Fact]
    public void NonfiniteIntermediateArithmeticThrowsInsteadOfPoisoningState()
    {
        var network = new DenseNetwork(1, [new(1, Activation.Linear)]);
        network.SetParameters([double.MaxValue, 0]);
        Assert.Throws<ArithmeticException>(() => network.Predict([2]));
        network.SetParameters([1, 0]);
        Assert.Throws<ArithmeticException>(() => network.EvaluateLoss([double.MaxValue], [-double.MaxValue]));
        Assert.Equal([1d, 0d], network.GetParameters());
    }
}
