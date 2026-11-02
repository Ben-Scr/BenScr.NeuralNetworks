using System;

namespace BenScr.NeuralNetworks;

/// <summary>An immutable training pair that owns copies of its finite input and target values.</summary>
public sealed class TrainingSample
{
    private readonly double[] input;
    private readonly double[] target;

    /// <summary>Copies nonempty vectors containing only finite values.</summary>
    /// <param name="input">Input features, at most <see cref="DenseNetwork.MaxWidth"/> values.</param>
    /// <param name="target">Expected output, at most <see cref="DenseNetwork.MaxWidth"/> values.</param>
    /// <exception cref="ArgumentException">A vector is empty, too large or contains a nonfinite value.</exception>
    public TrainingSample(ReadOnlySpan<double> input, ReadOnlySpan<double> target)
    {
        Validate(input, nameof(input));
        Validate(target, nameof(target));
        this.input = input.ToArray();
        this.target = target.ToArray();
    }

    /// <summary>Gets the input feature count.</summary>
    public int InputSize => input.Length;

    /// <summary>Gets the target output count.</summary>
    public int OutputSize => target.Length;

    /// <summary>Returns an independent copy of the input vector.</summary>
    public double[] GetInput() => (double[])input.Clone();

    /// <summary>Returns an independent copy of the target vector.</summary>
    public double[] GetTarget() => (double[])target.Clone();

    internal ReadOnlySpan<double> Input => input;
    internal ReadOnlySpan<double> Target => target;

    private static void Validate(ReadOnlySpan<double> values, string name)
    {
        if (values.Length == 0 || values.Length > DenseNetwork.MaxWidth)
        {
            throw new ArgumentException("Vector size must be between 1 and the maximum network width.", name);
        }
        DenseNetwork.CheckFinite(values, name);
    }
}
