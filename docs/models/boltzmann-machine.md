# Boltzmann Machine

## Primary Task

- Generation
- Pattern reconstruction
- Unsupervised feature learning

## Current Runtime

- Not implemented yet

## Current Assumptions

- This model will not be forced into classification as a primary task
- It should follow the same project principles:
  - Wasm compute core
  - terminal-verifiable tests
  - UI focused on inspection and education

## Current Limitations

- No Wasm implementation yet
- No browser worker yet
- No CLI smoke tests yet

## TODO

- Define architecture and observability surface before UI work
- Add deterministic toy-system tests before scaling to MNIST/Fashion-MNIST
- Keep training and generation controls clearly separated in the final page
