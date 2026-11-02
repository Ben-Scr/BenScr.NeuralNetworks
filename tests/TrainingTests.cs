using System;
using System.Linq;
using System.Threading;
using Xunit;

namespace BenScr.NeuralNetworks.Tests;

public sealed class TrainingTests
{
    [Fact]
    public void OneSgdStepMatchesHandCalculation()
    {
        var network = new DenseNetwork(1, [new(1, Activation.Linear)]);
        network.SetParameters([1, 0]);
        TrainingResult result = network.Train([new([2], [3])], new TrainingOptions { Epochs = 1, LearningRate = 0.1 }, TestContext.Current.CancellationToken);
        Assert.Equal(1.4, network.GetParameters()[0], 14);
        Assert.Equal(0.2, network.GetParameters()[1], 14);
        Assert.Equal(1, result.InitialLoss);
        Assert.Equal(0, result.FinalLoss, 14);
        Assert.Equal(1, result.EpochsCompleted);
    }

    [Fact]
    public void BatchUsesMeanGradientAndDoesNotUpdateBetweenItsSamples()
    {
        var network = new DenseNetwork(1, [new(1, Activation.Linear)]);
        network.SetParameters([0, 0]);
        network.Train([new([1], [2]), new([3], [4])],
            new TrainingOptions { Epochs = 1, BatchSize = 10, LearningRate = 0.1, Shuffle = false }, TestContext.Current.CancellationToken);
        Assert.Equal(1.4, network.GetParameters()[0], 14);
        Assert.Equal(0.6, network.GetParameters()[1], 14);
    }

    [Fact]
    public void LastPartialBatchDividesByItsActualCount()
    {
        var network = new DenseNetwork(1, [new(1, Activation.Linear)]);
        network.SetParameters([0, 0]);
        network.Train([new([0], [1]), new([0], [1]), new([0], [1])],
            new TrainingOptions { Epochs = 1, BatchSize = 2, LearningRate = 0.1, Shuffle = false }, TestContext.Current.CancellationToken);
        Assert.Equal(0.36, network.GetParameters()[1], 14);
    }

    [Fact]
    public void LearnsLinearRegression()
    {
        var network = new DenseNetwork(1, [new(1, Activation.Linear)], seed: 7);
        TrainingSample[] samples = Enumerable.Range(-5, 11).Select(x => new TrainingSample([x / 5.0], [2 * x / 5.0 + 1])).ToArray();
        TrainingResult result = network.Train(samples,
            new TrainingOptions { Epochs = 1000, BatchSize = 11, LearningRate = 0.1, TargetLoss = 1e-12 }, TestContext.Current.CancellationToken);
        Assert.True(result.FinalLoss < 1e-12);
        Assert.Equal(1.5, network.Predict([0.25])[0], 5);
        Assert.True(result.FinalLoss < result.InitialLoss);
    }

    [Fact]
    public void LearnsXorAndRoundtripsLearnedPredictions()
    {
        var network = new DenseNetwork(2, [new(4, Activation.Tanh), new(1, Activation.Sigmoid)], seed: 42);
        TrainingSample[] samples = [new([0, 0], [0]), new([0, 1], [1]), new([1, 0], [1]), new([1, 1], [0])];
        TrainingResult result = network.Train(samples,
            new TrainingOptions { Epochs = 10000, LearningRate = 0.5, Seed = 17, TargetLoss = 0.003 }, TestContext.Current.CancellationToken);
        Assert.True(result.FinalLoss <= 0.003, $"Loss: {result.FinalLoss}");
        DenseNetwork loaded = DenseNetwork.FromJson(network.ToJson());
        foreach (TrainingSample sample in samples)
        {
            double prediction = network.Predict(sample.GetInput())[0];
            Assert.InRange(Math.Abs(prediction - sample.GetTarget()[0]), 0, 0.12);
            Assert.Equal(prediction, loaded.Predict(sample.GetInput())[0]);
        }
    }

    [Fact]
    public void FixedSeedTrainingIsReproducible()
    {
        var first = new DenseNetwork(1, [new(3, Activation.Tanh), new(1, Activation.Linear)], 123);
        var second = new DenseNetwork(1, [new(3, Activation.Tanh), new(1, Activation.Linear)], 123);
        TrainingSample[] samples = [new([-1], [1]), new([0], [0]), new([1], [1])];
        var options = new TrainingOptions { Epochs = 20, LearningRate = 0.05, BatchSize = 2, Seed = -2 };
        Assert.Equal(first.Train(samples, options, TestContext.Current.CancellationToken), second.Train(samples, options, TestContext.Current.CancellationToken));
        Assert.Equal(first.GetParameters(), second.GetParameters());
    }

    [Fact]
    public void TargetLossCanStopBeforeAnyUpdate()
    {
        var network = new DenseNetwork(1, [new(1, Activation.Linear)]);
        network.SetParameters([1, 0]);
        TrainingResult result = network.Train([new([2], [2])], new TrainingOptions { TargetLoss = 0 }, TestContext.Current.CancellationToken);
        Assert.Equal(new TrainingResult(0, 0, 0), result);
        Assert.Equal([1d, 0d], network.GetParameters());
    }

    [Fact]
    public void DatasetLossAveragesSamplesAndOutputs()
    {
        var network = new DenseNetwork(1, [new(2, Activation.Linear)]);
        network.SetParameters([0, 0, 0, 0]);
        Assert.Equal(7.5, network.EvaluateLoss([new([0], [1, 2]), new([0], [3, 4])], TestContext.Current.CancellationToken));
    }

    [Fact]
    public void InvalidDatasetAndOptionsLeaveParametersUntouched()
    {
        var network = new DenseNetwork(1, [new(1, Activation.Linear)]);
        double[] before = network.GetParameters();
        Assert.Throws<ArgumentNullException>(() => network.Train(null!, cancellationToken: TestContext.Current.CancellationToken));
        Assert.Throws<ArgumentException>(() => network.Train([], cancellationToken: TestContext.Current.CancellationToken));
        Assert.Throws<ArgumentException>(() => network.Train([null!], cancellationToken: TestContext.Current.CancellationToken));
        Assert.Throws<ArgumentException>(() => network.Train([new([1], [1]), new([1, 2], [1])], cancellationToken: TestContext.Current.CancellationToken));
        Assert.Throws<ArgumentException>(() => network.Train([new([1], [1, 2])], cancellationToken: TestContext.Current.CancellationToken));
        TrainingOptions[] invalid =
        [
            new() { Epochs = 0 }, new() { Epochs = 1000001 }, new() { BatchSize = 0 },
            new() { LearningRate = 0 }, new() { LearningRate = -1 }, new() { LearningRate = double.NaN },
            new() { LearningRate = double.PositiveInfinity }, new() { TargetLoss = -1 }, new() { TargetLoss = double.NaN }
        ];
        foreach (TrainingOptions options in invalid)
        {
            Assert.Throws<ArgumentException>(() => network.Train([new([1], [2])], options, TestContext.Current.CancellationToken));
        }
        Assert.Equal(before, network.GetParameters());
    }

    [Fact]
    public void PreCancellationDoesNotMutateParameters()
    {
        var network = new DenseNetwork(1, [new(1, Activation.Linear)]);
        double[] before = network.GetParameters();
        using var source = new CancellationTokenSource();
        source.Cancel();
        Assert.Throws<OperationCanceledException>(() => network.Train([new([1], [2])], cancellationToken: source.Token));
        Assert.Throws<OperationCanceledException>(() => network.EvaluateLoss([new([1], [2])], source.Token));
        Assert.Equal(before, network.GetParameters());
    }

    [Fact]
    public void OverflowingBatchUpdateIsAtomic()
    {
        var network = new DenseNetwork(1, [new(2, Activation.Linear)]);
        network.SetParameters([0, 0, 0, 0]);
        Assert.Throws<ArithmeticException>(() => network.Train([new([2], [0.1, 2])],
            new TrainingOptions { Epochs = 1, LearningRate = double.MaxValue }, TestContext.Current.CancellationToken));
        Assert.Equal([0d, 0d, 0d, 0d], network.GetParameters());
    }
}
