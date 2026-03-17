# Hopfield Network

## Primary Task

- Associative retrieval
- Pattern reconstruction

## Current Runtime

- Browser UI: Wasm-backed worker
- Terminal tests: Rust unit tests, Wasm regression checks, browser build validation
- Legacy browser-side TypeScript engine removed; `src/core/hopfield.ts` now carries shared state types only

## Model Notes

- Classical bipolar Hopfield network over `784` visible neurons in the MNIST/Fashion-MNIST setup
- Stored dataset images are binarized for the Hopfield page
- Supports multiple learning rules:
  - Hebbian
  - Pseudoinverse
  - Storkey
  - Krauth-Mezard
  - Unlearning
- Supports multiple convergence rules:
  - Asynchronous random
  - Synchronous
  - Stochastic

## Current Assumptions

- Memory patterns are bipolar in `{-1, +1}`
- Asynchronous updates should not increase energy
- Learned weight matrix must remain symmetric with zero diagonal

## Current Limitations

- UI still mixes setup and inference controls too tightly
- Worker protocol is model-specific rather than documented as a reusable test contract
- Browser page still carries presentation-era wording and layout debt

## Current Tests

- Rust unit tests for retrieval, symmetry, zero diagonal, and async energy descent
- Wasm regression checks for deterministic invariants
- Browser route smoke validation through the devcontainer

## TODO

- Add more deterministic golden cases for each learning rule
- Add worker contract tests for initialize / setQuery / step / play / pause / reset
- Separate architecture setup from inference controls in the UI
