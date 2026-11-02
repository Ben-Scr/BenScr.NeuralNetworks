namespace BenScr.NeuralNetworks;

/// <summary>Immutable shape and elementwise activation for a dense layer.</summary>
/// <param name="Units">Number of neurons, from 1 through <see cref="DenseNetwork.MaxWidth"/>.</param>
/// <param name="Activation">Activation applied after the affine transformation.</param>
/// <remarks>The owning network validates definitions before allocating its parameters.</remarks>
public sealed record LayerDefinition(int Units, Activation Activation);
