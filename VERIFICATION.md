# Verification record

Initial Windows x64 validation on 2026-11-02 passed 59 tests with zero skipped or failed tests,
and the complete XOR/persistence/reference example passed. All projects built with zero warnings.

Tests include independent central finite differences for every parameter across all four activation
functions, an affine calculation, exact SGD/mean-batch updates, final partial-batch scaling, seeded
initialization/training, linear regression and XOR convergence. Additional tests cover copied ownership,
parallel prediction, invalid sizes/options, arithmetic overflow, atomic failed updates, cancellation,
model version/shape/finite validation, duplicate/unknown JSON properties, culture and size/depth limits.

The xUnit test project uses the .NET 10 Microsoft Testing Platform runner. All tests are discoverable;
the verifier rejects an empty test run or any incomplete outcome. The public API baseline was reviewed
and is checked against the built library.

Release verification is performed against a clean committed checkout. Its generated manifest records
the exact source commit and nupkg/snupkg hashes, package audit and fresh package-consumer execution.
`artifacts/latest-verification.txt` locates the corresponding output. Windows and Ubuntu CI run the
same verifier; their actual results are linked in the final release manifest. NuGet availability and
public-feed checks are recorded separately from local pack success.

Interactive debugger stepping, other architectures, Native AOT and trimming were not executed.
