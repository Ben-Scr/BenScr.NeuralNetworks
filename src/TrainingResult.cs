namespace BenScr.NeuralNetworks;

/// <summary>Loss measurements from a completed training call.</summary>
/// <param name="EpochsCompleted">Completed epochs; zero if the initial loss meets the target.</param>
/// <param name="InitialLoss">Mean squared error over every sample and output before updates.</param>
/// <param name="FinalLoss">Mean squared error after the last complete epoch.</param>
public sealed record TrainingResult(int EpochsCompleted, double InitialLoss, double FinalLoss);
