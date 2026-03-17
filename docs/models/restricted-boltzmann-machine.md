# Restricted Boltzmann Machine

## Primary Task

- Generation
- Pattern reconstruction
- Unsupervised feature learning

## Current Runtime

- Browser UI: Wasm-backed worker
- Terminal tests: Rust unit tests, Wasm regression checks, CLI smoke tests on bundled data
- Legacy TypeScript backend/training path removed; remaining TypeScript in `src/core/rbm.ts` is UI/helper-only

## Model Notes

- Bipartite visible/hidden architecture
- Supports:
  - Bernoulli visible model
  - Gaussian visible model
- Browser page now separates architecture, training, and Gibbs playback more clearly

## Current Assumptions

- Training epoch should preserve finite weights, biases, and visible reconstructions
- Epoch counter must advance exactly once per epoch call
- Gaussian visible reconstructions should stay bounded in `[0, 1]`

## Current Limitations

- Full Boltzmann Machine is not implemented yet
- Worker message contract is still implicit in implementation instead of explicitly tested

## Current Tests

- Rust unit tests for finite metrics, tensor shape checks, epoch advance, and Gaussian bounded reconstruction
- Wasm regression checks for Bernoulli and Gaussian paths
- CLI smoke tests over bundled MNIST data
- Browser route smoke validation in the devcontainer

## TODO

- Add deterministic tiny-dataset training baselines with tolerance windows
- Add explicit worker contract tests for training/playback transitions
- Add richer dataset breakdown and worker contract coverage
