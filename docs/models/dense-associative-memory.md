# Dense Associative Memory

## Primary Task

- Associative retrieval
- Pattern reconstruction

## Current Runtime

- Browser UI: Wasm-backed worker
- Terminal validation: Rust unit tests, Wasm regression checks, DAM smoke test on bundled MNIST

## Data Contract

- External input/output space: grayscale `Float32` values in `[0, 1]`
- Current bundled DAM datasets: `MNIST` and `Fashion-MNIST`
- DAM does **not** currently expose a separate binary-input mode
- Internal learning / scoring space: contrast-centered transform `x_c = 2x - 1`

That means the user-facing query editor, datasets, reconstructions, and feature previews all stay grayscale `[0,1]`, while the Wasm core performs hidden scoring and prototype updates in a centered contrast space. This is intentional and documented, not an implicit conversion.

## Model Notes

- The core is a two-layer bipartite associative memory with hidden prototype slots.
- `ReLU power` is the canonical DAM path in this project.
- `ReLU power` retrieval is sparse winner-take-all style: only the strongest positively aligned hidden slot survives.
- `Signed power` follows the same sparse winner path but preserves sign.
- `Softmax` is kept as an exploratory diffuse-attention variant and is not the default.
- Training uses competitive prototype updates with a winner slot and a weaker anti-Hebbian runner-up term.

## Current Assumptions

- Strong hidden competition is necessary to avoid diffuse attractor collapse.
- Centered internal contrast space is necessary for grayscale datasets with dominant background pixels.
- DAM is educational and inspectable first; exact paper-faithful Krotov objectives may still evolve.

## Current Limitations

- Layered DAM variants are not implemented yet.
- `Softmax` is available for comparison but is not the recommended default retrieval mode.
- The current trainer is competitive and practical, but still simpler than a full research-grade Krotov implementation.
- Worker protocol coverage still relies on runtime validation rather than dedicated contract tests.

## Current Tests

- Rust unit tests for:
  - finite training metrics
  - ordered-dataset seed spread
  - sharpness concentration
  - softmax normalization
  - non-collapse on a small toy retrieval set
- Wasm regression checks for concentration and normalization invariants
- `npm run dam:smoke` on bundled MNIST with explicit collapse guards:
  - minimum exact exemplar matches
  - minimum distinct matched labels
  - minimum distinct winning hidden slots

## TODO

- Add layered DAM variants
- Add dedicated worker contract tests
- Expand deterministic regression coverage across nonlinearity choices
- Revisit the trainer if stronger class-selective retrieval is needed on bundled datasets
