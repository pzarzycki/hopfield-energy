# Dense Hopfield Network

## Primary Task

- Associative retrieval
- Pattern reconstruction

## Current Runtime

- Browser UI: Wasm-backed worker
- Terminal tests: Rust unit tests, Wasm regression checks, production build validation
- Legacy browser-side TypeScript retrieval engine removed; `src/core/denseHopfield.ts` now carries shared snapshot types only

## Model Notes

- Continuous-state retrieval over stored grayscale exemplar memories
- Retrieval is implemented as softmax attention over stored memories
- Inverse temperature `beta` controls attention sharpness

## Current Assumptions

- Attention weights should form a normalized distribution
- Higher `beta` should sharpen the dominant memory weight for the same query
- The page emphasizes retrieval dynamics, not capacity scaling

## Current Limitations

- Uses one exemplar per class in the current educational setup
- UI still blends configuration and retrieval controls in one flow
- No dedicated worker-contract test suite yet

## Current Tests

- Rust unit tests for finite retrieval state, normalized attention, and beta sharpening
- Wasm regression checks for attention normalization and sharpening
- Browser route smoke and basic interaction validation

## TODO

- Add deterministic convergence baselines for selected toy memories
- Expand retrieval tests for noisy/occluded memory recovery
- Restructure UI into setup, retrieval, and inspection sections
