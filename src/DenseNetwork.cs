using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Threading;

namespace BenScr.NeuralNetworks;

/// <summary>A double-precision feed-forward network of fully connected layers.</summary>
/// <remarks>
/// Concurrent prediction, loss and gradient calls are safe while parameters remain unchanged.
/// Training and parameter mutation require exclusive access. Computation uses the CPU and allocates
/// temporary arrays. Nonfinite arithmetic throws <see cref="ArithmeticException"/>.
/// </remarks>
public sealed partial class DenseNetwork
{
    /// <summary>The inclusive maximum input size or number of units per layer: 4,096.</summary>
    public const int MaxWidth = 4_096;

    /// <summary>The inclusive maximum number of dense layers: 32.</summary>
    public const int MaxLayers = 32;

    /// <summary>The inclusive maximum total number of weights and biases: 1,000,000.</summary>
    public const int MaxParameters = 1_000_000;

    /// <summary>The inclusive maximum number of samples per training or dataset-loss call: 100,000.</summary>
    public const int MaxSamples = 100_000;

    private readonly LayerDefinition[] definitions;
    private readonly ReadOnlyCollection<LayerDefinition> layers;
    private readonly int[] offsets;
    private readonly double[] parameters;

    /// <summary>Constructs a network with zero biases and seeded Glorot-uniform weights.</summary>
    /// <param name="inputSize">Input feature count, from 1 through <see cref="MaxWidth"/>.</param>
    /// <param name="layers">One through <see cref="MaxLayers"/> layer definitions, copied on construction.</param>
    /// <param name="seed">Seed for the version-one SplitMix64 initialization stream.</param>
    /// <exception cref="ArgumentNullException"><paramref name="layers"/> is null.</exception>
    /// <exception cref="ArgumentException">The shape, activation or parameter budget is invalid.</exception>
    public DenseNetwork(int inputSize, IReadOnlyList<LayerDefinition> layers, int seed = 0)
    {
        ArgumentNullException.ThrowIfNull(layers);
        if (inputSize < 1 || inputSize > MaxWidth)
        {
            throw new ArgumentOutOfRangeException(nameof(inputSize));
        }
        if (layers.Count < 1 || layers.Count > MaxLayers)
        {
            throw new ArgumentException("A network requires between 1 and 32 layers.", nameof(layers));
        }

        InputSize = inputSize;
        definitions = new LayerDefinition[layers.Count];
        offsets = new int[layers.Count];
        int previous = inputSize;
        long count = 0;
        for (int layer = 0; layer < layers.Count; layer++)
        {
            LayerDefinition definition = layers[layer] ?? throw new ArgumentException("Layers cannot contain null.", nameof(layers));
            if (definition.Units < 1 || definition.Units > MaxWidth || !Enum.IsDefined(definition.Activation))
            {
                throw new ArgumentException("Layer widths and activations must be supported.", nameof(layers));
            }
            definitions[layer] = definition;
            offsets[layer] = (int)count;
            count += (long)definition.Units * (previous + 1);
            if (count > MaxParameters)
            {
                throw new ArgumentException("The network exceeds the parameter budget.", nameof(layers));
            }
            previous = definition.Units;
        }
        this.layers = Array.AsReadOnly(definitions);
        parameters = new double[(int)count];
        var random = new SeededRandom(seed);
        previous = inputSize;
        for (int layer = 0; layer < definitions.Length; layer++)
        {
            int width = definitions[layer].Units;
            double scale = Math.Sqrt(6.0 / (previous + width));
            for (int weight = 0; weight < previous * width; weight++)
            {
                parameters[offsets[layer] + weight] = (2 * random.NextDouble() - 1) * scale;
            }
            previous = width;
        }
    }

    /// <summary>Gets the required input feature count.</summary>
    public int InputSize { get; }

    /// <summary>Gets the output vector size.</summary>
    public int OutputSize => definitions[^1].Units;

    /// <summary>Gets an immutable view of the copied layer definitions.</summary>
    public IReadOnlyList<LayerDefinition> Layers => layers;

    /// <summary>Gets the total count of weights and biases.</summary>
    public int ParameterCount => parameters.Length;

    /// <summary>Returns an independent copy of all parameters in layer order.</summary>
    /// <remarks>Each layer stores row-major weights (output, input), followed by all biases.</remarks>
    public double[] GetParameters() => (double[])parameters.Clone();

    /// <summary>Replaces all parameters after checking the entire vector.</summary>
    /// <param name="values">Exactly <see cref="ParameterCount"/> finite values in documented layer order.</param>
    /// <exception cref="ArgumentException">The size or a value is invalid. No parameter is changed.</exception>
    public void SetParameters(ReadOnlySpan<double> values)
    {
        ValidateVector(values, ParameterCount, nameof(values));
        values.CopyTo(parameters);
    }

    /// <summary>Computes a new output array without changing parameters.</summary>
    /// <param name="input">Exactly <see cref="InputSize"/> finite feature values.</param>
    /// <exception cref="ArgumentException">Input shape or a value is invalid.</exception>
    /// <exception cref="ArithmeticException">An intermediate result is nonfinite.</exception>
    public double[] Predict(ReadOnlySpan<double> input)
    {
        ValidateVector(input, InputSize, nameof(input));
        return Forward(input)[^1];
    }

    /// <summary>Computes the mean squared error across this sample's outputs.</summary>
    /// <param name="input">Exactly <see cref="InputSize"/> finite feature values.</param>
    /// <param name="target">Exactly <see cref="OutputSize"/> finite target values.</param>
    /// <returns>The mean of squared prediction-minus-target differences.</returns>
    public double EvaluateLoss(ReadOnlySpan<double> input, ReadOnlySpan<double> target)
    {
        ValidateVector(target, OutputSize, nameof(target));
        return Loss(Predict(input), target);
    }

    /// <summary>Computes analytic derivatives of sample mean squared error for every parameter.</summary>
    /// <param name="input">Exactly <see cref="InputSize"/> finite feature values.</param>
    /// <param name="target">Exactly <see cref="OutputSize"/> finite target values.</param>
    /// <returns>A new gradient array in the same order as <see cref="GetParameters"/>.</returns>
    /// <remarks>Parameters remain unchanged; the ReLU derivative at zero is zero.</remarks>
    public double[] ComputeGradient(ReadOnlySpan<double> input, ReadOnlySpan<double> target)
    {
        ValidateVector(input, InputSize, nameof(input));
        ValidateVector(target, OutputSize, nameof(target));
        var gradient = new double[ParameterCount];
        AccumulateGradient(input, target, gradient, 1);
        return gradient;
    }

    /// <summary>Computes mean squared error across all samples and outputs.</summary>
    /// <param name="samples">One through <see cref="MaxSamples"/> immutable pairs matching this network.</param>
    /// <param name="cancellationToken">Cancellation checked between samples.</param>
    /// <returns>The arithmetic mean of each sample's mean squared error.</returns>
    public double EvaluateLoss(IReadOnlyList<TrainingSample> samples, CancellationToken cancellationToken = default)
    {
        TrainingSample[] snapshot = ValidateSamples(samples);
        return DatasetLoss(snapshot, cancellationToken);
    }

    /// <summary>Updates parameters by mini-batch stochastic gradient descent on mean squared error.</summary>
    /// <param name="samples">One through <see cref="MaxSamples"/> immutable training pairs.</param>
    /// <param name="options">Settings copied at entry; null selects the documented defaults.</param>
    /// <param name="cancellationToken">Cancellation checked between samples and before committing each batch.</param>
    /// <returns>Initial/final dataset losses and completed epoch count.</returns>
    /// <remarks>
    /// Invalid samples/options do not change parameters. Each batch update is atomic with respect to
    /// arithmetic failure and cancellation; already completed batches remain applied on failure.
    /// An entire training call is not transactional. Callers must synchronize mutation and reading.
    /// </remarks>
    public TrainingResult Train(IReadOnlyList<TrainingSample> samples, TrainingOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new TrainingOptions();
        ValidateOptions(options);
        cancellationToken.ThrowIfCancellationRequested();
        TrainingSample[] snapshot = ValidateSamples(samples);
        double initial = DatasetLoss(snapshot, cancellationToken);
        double final = initial;
        if (options.TargetLoss is double targetLoss && initial <= targetLoss)
        {
            return new TrainingResult(0, initial, initial);
        }
        var random = new SeededRandom(options.Seed);
        var order = new int[snapshot.Length];
        for (int index = 0; index < order.Length; index++)
        {
            order[index] = index;
        }
        var gradient = new double[ParameterCount];
        var proposed = new double[ParameterCount];
        int completed = 0;
        for (int epoch = 0; epoch < options.Epochs; epoch++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (options.Shuffle)
            {
                for (int index = order.Length - 1; index > 0; index--)
                {
                    int other = random.NextIndex(index + 1);
                    (order[index], order[other]) = (order[other], order[index]);
                }
            }
            for (int start = 0; start < order.Length;)
            {
                int count = Math.Min(options.BatchSize, order.Length - start);
                Array.Clear(gradient);
                for (int position = start; position < start + count; position++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    TrainingSample sample = snapshot[order[position]];
                    AccumulateGradient(sample.Input, sample.Target, gradient, 1.0 / count);
                }
                for (int parameter = 0; parameter < ParameterCount; parameter++)
                {
                    proposed[parameter] = Finite(parameters[parameter] - options.LearningRate * gradient[parameter]);
                }
                cancellationToken.ThrowIfCancellationRequested();
                proposed.CopyTo(parameters, 0);
                start += count;
            }
            completed++;
            final = DatasetLoss(snapshot, cancellationToken);
            if (options.TargetLoss is double threshold && final <= threshold)
            {
                break;
            }
        }
        return new TrainingResult(completed, initial, final);
    }

    internal static void CheckFinite(ReadOnlySpan<double> values, string name)
    {
        foreach (double value in values)
        {
            if (!double.IsFinite(value))
            {
                throw new ArgumentException("All values must be finite.", name);
            }
        }
    }

    private static void ValidateVector(ReadOnlySpan<double> values, int size, string name)
    {
        if (values.Length != size)
        {
            throw new ArgumentException("Vector size does not match the network.", name);
        }
        CheckFinite(values, name);
    }

    private TrainingSample[] ValidateSamples(IReadOnlyList<TrainingSample> samples)
    {
        ArgumentNullException.ThrowIfNull(samples);
        if (samples.Count < 1 || samples.Count > MaxSamples)
        {
            throw new ArgumentException("The dataset must contain between 1 and 100,000 samples.", nameof(samples));
        }
        var snapshot = new TrainingSample[samples.Count];
        for (int index = 0; index < snapshot.Length; index++)
        {
            TrainingSample sample = samples[index] ?? throw new ArgumentException("Samples cannot contain null.", nameof(samples));
            if (sample.InputSize != InputSize || sample.OutputSize != OutputSize)
            {
                throw new ArgumentException("Sample dimensions do not match the network.", nameof(samples));
            }
            snapshot[index] = sample;
        }
        return snapshot;
    }

    private static void ValidateOptions(TrainingOptions options)
    {
        if (options.Epochs < 1 || options.Epochs > 1_000_000 || options.BatchSize < 1 ||
            !double.IsFinite(options.LearningRate) || options.LearningRate <= 0 ||
            (options.TargetLoss is double loss && (!double.IsFinite(loss) || loss < 0)))
        {
            throw new ArgumentException("Epochs, batch size, learning rate or target loss is invalid.", nameof(options));
        }
    }

    private double[][] Forward(ReadOnlySpan<double> input)
    {
        var values = new double[definitions.Length + 1][];
        values[0] = input.ToArray();
        for (int layer = 0; layer < definitions.Length; layer++)
        {
            LayerDefinition definition = definitions[layer];
            double[] previous = values[layer];
            var next = new double[definition.Units];
            int biasOffset = offsets[layer] + definition.Units * previous.Length;
            for (int output = 0; output < next.Length; output++)
            {
                double sum = parameters[biasOffset + output];
                int row = offsets[layer] + output * previous.Length;
                for (int feature = 0; feature < previous.Length; feature++)
                {
                    sum = Finite(sum + parameters[row + feature] * previous[feature]);
                }
                next[output] = Activate(sum, definition.Activation);
            }
            values[layer + 1] = next;
        }
        return values;
    }

    private void AccumulateGradient(ReadOnlySpan<double> input, ReadOnlySpan<double> target,
        double[] gradient, double scale)
    {
        double[][] values = Forward(input);
        var delta = new double[OutputSize];
        for (int output = 0; output < OutputSize; output++)
        {
            double error = Finite(values[^1][output] - target[output]);
            delta[output] = Finite((2.0 / OutputSize) * error * Derivative(values[^1][output], definitions[^1].Activation));
        }
        for (int layer = definitions.Length - 1; layer >= 0; layer--)
        {
            double[] previous = values[layer];
            int biasOffset = offsets[layer] + delta.Length * previous.Length;
            for (int output = 0; output < delta.Length; output++)
            {
                int row = offsets[layer] + output * previous.Length;
                for (int feature = 0; feature < previous.Length; feature++)
                {
                    gradient[row + feature] = Finite(gradient[row + feature] + scale * delta[output] * previous[feature]);
                }
                gradient[biasOffset + output] = Finite(gradient[biasOffset + output] + scale * delta[output]);
            }
            if (layer > 0)
            {
                var previousDelta = new double[previous.Length];
                for (int feature = 0; feature < previous.Length; feature++)
                {
                    double sum = 0;
                    for (int output = 0; output < delta.Length; output++)
                    {
                        sum = Finite(sum + parameters[offsets[layer] + output * previous.Length + feature] * delta[output]);
                    }
                    previousDelta[feature] = Finite(sum * Derivative(previous[feature], definitions[layer - 1].Activation));
                }
                delta = previousDelta;
            }
        }
    }

    private double DatasetLoss(TrainingSample[] samples, CancellationToken cancellationToken)
    {
        double total = 0;
        foreach (TrainingSample sample in samples)
        {
            cancellationToken.ThrowIfCancellationRequested();
            total = Finite(total + Loss(Forward(sample.Input)[^1], sample.Target) / samples.Length);
        }
        return total;
    }

    private static double Loss(ReadOnlySpan<double> output, ReadOnlySpan<double> target)
    {
        double total = 0;
        for (int index = 0; index < output.Length; index++)
        {
            double error = Finite(output[index] - target[index]);
            total = Finite(total + (error / output.Length) * error);
        }
        return total;
    }

    private static double Activate(double value, Activation activation) => activation switch
    {
        Activation.Linear => value,
        Activation.Tanh => Math.Tanh(value),
        Activation.Relu => Math.Max(0, value),
        Activation.Sigmoid => value >= 0 ? 1 / (1 + Math.Exp(-value)) : Math.Exp(value) / (1 + Math.Exp(value)),
        _ => throw new InvalidOperationException("Unsupported activation.")
    };

    private static double Derivative(double activated, Activation activation) => activation switch
    {
        Activation.Linear => 1,
        Activation.Sigmoid => activated * (1 - activated),
        Activation.Tanh => 1 - activated * activated,
        Activation.Relu => activated > 0 ? 1 : 0,
        _ => throw new InvalidOperationException("Unsupported activation.")
    };

    private static double Finite(double value) => double.IsFinite(value)
        ? value : throw new ArithmeticException("Network arithmetic produced a nonfinite value.");
}
