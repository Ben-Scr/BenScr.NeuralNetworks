# Provenance

Created by Ben Schneider — https://github.com/Ben-Scr

This repository contains a new implementation of standard dense-layer forward propagation and
backpropagation for mean squared error. No legacy model files or application-specific source are
included. The training RNG implements the published SplitMix64 arithmetic; it is not cryptographic.
Glorot uniform initialization scales weights by fan-in/fan-out. These standard algorithms are
implemented directly; no third-party algorithm source is copied into the library.

Build and test packages retain their own licenses and are not runtime dependencies. Package auditing
and build configuration adapt the existing BenScr.MathParser verification tooling under the same
author license. The original custom author license is retained in LICENSE.txt.

The common original icon is copied unchanged to `branding/BenScr_Icon_Master.png` and embedded as
`icon.png`. It is a transparent 1254 x 1254 PNG, 795819 bytes, SHA-256
`a5dcabceb766e13c1840a95130ee98a458e12e3713fd9b07f3ceae75a1f5b993`.

The repository and NuGet target ID were checked before creation. The package's nuspec, portable PDBs
and release manifest identify the exact source commit used for a build.
