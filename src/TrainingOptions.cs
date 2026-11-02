namespace BenScr.NeuralNetworks;

/// <summary>Immutable settings for a single mini-batch gradient-descent call.</summary>
public sealed record TrainingOptions
{
    /// <summary>Gets the maximum epoch count, from 1 through 1,000,000. Default: 1,000.</summary>
    public int Epochs { get; init; } = 1_000;

    /// <summary>Gets the finite, positive gradient-descent step size. Default: 0.1.</summary>
    public double LearningRate { get; init; } = 0.1;

    /// <summary>Gets the positive batch size; the final batch may be smaller. Default: 1.</summary>
    public int BatchSize { get; init; } = 1;

    /// <summary>Gets whether Fisher-Yates shuffling runs before each epoch. Default: true.</summary>
    public bool Shuffle { get; init; } = true;

    /// <summary>Gets the seed for this call's shuffle stream. Default: zero.</summary>
    public int Seed { get; init; }

    /// <summary>Gets the optional finite, nonnegative mean loss at which training stops.</summary>
    /// <remarks>The threshold is checked initially and after complete epochs.</remarks>
    public double? TargetLoss { get; init; }
}
