# Compatibility

BenScr.NeuralNetworks 1.0.0 is a new library and model format. It does not replace or import any
pre-existing BenScr neural-network package, private repository or external model format.

The public API baseline is `docs/PublicApi.txt`. Model format version 1 stores architecture and
parameters; exact fields, numeric meaning and validation are documented in README. Unsupported model
versions are rejected. Future incompatible API or model-format changes require explicit migration
instructions and appropriate package versioning.
