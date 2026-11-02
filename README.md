# BenScr.NeuralNetworks

Small CPU dense neural networks for .NET 10: configurable layers, prediction, mini-batch gradient
descent, analytic gradients and validated JSON model persistence. No runtime package dependencies.

Created by Ben Schneider — https://github.com/Ben-Scr

```sh
dotnet add package BenScr.NeuralNetworks --version 1.0.0
```

```csharp
using BenScr.NeuralNetworks;

var network = new DenseNetwork(2,
    [new LayerDefinition(4, Activation.Tanh), new LayerDefinition(1, Activation.Sigmoid)], seed: 42);
TrainingSample[] xor =
[
    new([0, 0], [0]), new([0, 1], [1]), new([1, 0], [1]), new([1, 1], [0])
];
TrainingResult result = network.Train(xor,
    new TrainingOptions { Epochs = 10_000, LearningRate = 0.5, Seed = 17, TargetLoss = 0.003 });
double prediction = network.Predict([0, 1])[0];
string modelJson = network.ToJson();
DenseNetwork restored = DenseNetwork.FromJson(modelJson);
```

The complete [example](https://github.com/Ben-Scr/BenScr.NeuralNetworks/blob/master/examples/Program.cs)
checks all four XOR predictions, persistence and an independently known affine gradient.

## Numerical contract

- Dense layers apply an affine transform followed by `Linear`, `Sigmoid`, `Tanh` or `Relu`.
- All calculations and parameters are `double`. Inputs, targets and stored parameters must be finite.
  Nonfinite intermediate arithmetic throws `ArithmeticException`; normalize features and choose an
  appropriate learning rate. The library does not silently clip gradients or outputs.
- Loss is mean squared error averaged over output elements and then samples. Gradient descent uses
  the mean gradient of each batch; the final short batch uses its actual sample count.
- `ComputeGradient` returns analytic derivatives for a single sample without updating the model.
  ReLU's derivative at zero is defined as zero.
- `GetParameters`, `SetParameters` and `ComputeGradient` use the same order: layer by layer, each
  layer's row-major weights `(output, input)` followed by its biases. Getters return copies.
- Weights use Glorot uniform initialization, with bounds `sqrt(6 / (fanIn + fanOut))`; biases start at
  zero. Initialization and Fisher-Yates shuffling use the documented version-one SplitMix64 stream.
  Each `Train` call starts its own shuffle stream. Reproducibility applies to the same library,
  runtime, architecture and options. Cross-platform bit-identical floating-point math is not promised.
- Training stops after the specified epochs or when `TargetLoss` is reached initially or after an
  entire epoch. Loss values are measured on the whole supplied dataset, not a validation split.
  Convergence depends on the data and settings; training success is not guaranteed for arbitrary data.

`TrainingSample` copies its vectors. Networks copy definitions and expose an immutable view. Dataset
collections must not be changed while a call validates them; calls then use their own sample-reference
snapshot. Concurrent predictions, loss calculations and gradients are supported while parameters are
unchanged. `Train` and `SetParameters` require exclusive access; callers provide synchronization.

Cancellation is checked between samples and before every batch commit. An invalid dataset/options or
pre-cancelled token causes no update. Each batch is committed only after all candidate parameters are
finite. Completed batches remain applied if a later batch fails or cancellation is requested; an
entire training call is not transactional. `OperationCanceledException` carries the supplied token.

## Scope and limits

This is an in-memory, CPU-only dense-network implementation for small experiments, learning tasks and
regression models. It does not supply GPU kernels, tensors, automatic differentiation for arbitrary
graphs, convolution/recurrent layers, pretrained models, distributed training, ONNX interchange or a
general deep-learning runtime. No Native AOT, trimming or allocation-free guarantee is made.

| Limit | Inclusive maximum |
| --- | --- |
| Input size / layer width | 4,096 |
| Dense layers | 32 |
| Total weights and biases | 1,000,000 |
| Samples in one call | 100,000 |
| Epochs in one training call | 1,000,000 |
| Model JSON length | 33,554,432 UTF-16 code units |
| Model JSON nesting depth | 8 |

These are validation ceilings, not performance promises. Training work still scales with sample count,
epoch count and parameter count. Prediction/training allocate working arrays; datasets and model JSON
also require memory.

## Model format

`ToJson` writes compact invariant-culture JSON with `formatVersion: 1`, `inputSize`, `layers` and
`parameters`. Layer entries contain `units` and the exact activation name. `FromJson` validates sizes,
parameter count, finite numbers, supported version and activations. Unknown, missing and duplicate
members are rejected. Invalid JSON/model data throws `System.Text.Json.JsonException` (or a subtype).

The format contains architecture and parameters only. Training data, optimizer/shuffle state and
application labels are not persisted. Store the string with standard `File`/`Stream` APIs as needed.

## Build and verify

Requirements: .NET SDK 10.0.303, .NET 10 runtime, Git and PowerShell 7. SDK and test dependency versions
are pinned; restore lock files are committed. Tests use xUnit v3 4.0.2 with Microsoft Testing Platform.

```powershell
dotnet restore tests/BenScr.NeuralNetworks.Tests.csproj --locked-mode
dotnet test --project tests/BenScr.NeuralNetworks.Tests.csproj -c Release --no-restore
pwsh -File tools/Verify.ps1
```

The verifier checks the public API baseline, tests, example, final archives, original icon and license,
portable PDB identities/Source Link checksums, and an isolated package consumer with an empty cache.
CI runs it on Windows and Ubuntu x64. Release manifests record actually executed checks and exact
source commits; see [release instructions](https://github.com/Ben-Scr/BenScr.NeuralNetworks/blob/master/RELEASE.md).

The original author license is in [LICENSE.txt](https://github.com/Ben-Scr/BenScr.NeuralNetworks/blob/master/LICENSE.txt)
and embedded in the package. The BenScr icon is embedded unchanged. See
[provenance](https://github.com/Ben-Scr/BenScr.NeuralNetworks/blob/master/docs/PROVENANCE.md) and
[changelog](https://github.com/Ben-Scr/BenScr.NeuralNetworks/blob/master/CHANGELOG.md).
