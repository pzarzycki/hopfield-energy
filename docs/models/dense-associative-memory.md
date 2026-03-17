# Dense Associative Memory

## Primary Task

- Associative retrieval
- Pattern reconstruction

## Current Runtime

- Browser UI: Wasm-backed worker
- Terminal tests: Rust unit tests, Wasm regression checks, CLI smoke tests on bundled data
- Legacy TypeScript training path removed; remaining TypeScript in `src/core/denseAssociativeMemory.ts` is UI/helper-only

## Model Notes

- Krotov-style bipartite associative memory
- Supports nonlinearities:
  - ReLU power
  - Signed power
  - Softmax
- Hidden competition sharpness is controlled by the power / inverse-temperature-style setting
- Browser page now separates architecture, training, and retrieval playback more clearly

## Current Assumptions

- Higher sharpness should increase winner concentration for the same query
- Softmax hidden activations should remain normalized
- Training epoch should keep all reported metrics finite

## Current Limitations

- Layered neuron variants are not implemented yet
- Worker protocol coverage still relies on runtime smoke checks rather than dedicated contract tests

## Current Tests

- Rust unit tests for finite training metrics, sharpness concentration, and softmax normalization
- Wasm regression checks for concentration and normalization invariants
- CLI smoke tests over bundled MNIST data
- Browser route smoke validation in the devcontainer

## TODO

- Add layered DAM variants
- Add deterministic retrieval baselines across nonlinearity choices
- Add worker contract tests and broader deterministic retrieval baselines
