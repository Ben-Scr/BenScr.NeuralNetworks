namespace BenScr.NeuralNetworks;

/// <summary>Elementwise activation functions supported by dense layers.</summary>
public enum Activation
{
    /// <summary>The identity function, suitable for unbounded regression outputs.</summary>
    Linear,
    /// <summary>The logistic sigmoid with outputs in [0, 1].</summary>
    Sigmoid,
    /// <summary>The hyperbolic tangent with outputs in [-1, 1].</summary>
    Tanh,
    /// <summary>The rectified linear function. Its derivative at zero is defined as zero.</summary>
    Relu
}
