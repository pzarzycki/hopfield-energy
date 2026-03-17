# Energy-Based Memory Models

Interactive and terminal-testable implementations of classical and modern energy-based memory models, from Hopfield networks to RBMs and Dense Associative Memory.

🚀 **Live Demo:** [https://pzarzycki.github.io/hopfield-energy/](https://pzarzycki.github.io/hopfield-energy/)
[![GitHub Pages CI](https://github.com/pzarzycki/hopfield-energy/actions/workflows/deploy.yml/badge.svg)](https://github.com/pzarzycki/hopfield-energy/actions/workflows/deploy.yml)

## Overview

This project explores energy-based models with two execution targets:

- **Browser UI** for interactive visualization, training controls, and query playback
- **CLI smoke tests** for model/data validation without React or a browser

The current model set includes:

- **Hopfield Network**
- **Dense Hopfield Network**
- **Restricted Boltzmann Machine**
- **Dense Associative Memory**

Per-model notes now live in:

- [docs/models/hopfield.md](C:\PRJ\hopfield-energy\docs\models\hopfield.md)
- [docs/models/dense-hopfield.md](C:\PRJ\hopfield-energy\docs\models\dense-hopfield.md)
- [docs/models/restricted-boltzmann-machine.md](C:\PRJ\hopfield-energy\docs\models\restricted-boltzmann-machine.md)
- [docs/models/dense-associative-memory.md](C:\PRJ\hopfield-energy\docs\models\dense-associative-memory.md)
- [docs/models/boltzmann-machine.md](C:\PRJ\hopfield-energy\docs\models\boltzmann-machine.md)

## Current Wasm Status

Rust/Wasm compute baselines now exist in [wasm-core](C:\PRJ\hopfield-energy\wasm-core) for all currently implemented model families:

- Hopfield
- Dense Hopfield
- Restricted Boltzmann Machine
- Dense Associative Memory

Current status:

- Rust crate builds successfully in the devcontainer
- Rust unit tests pass
- `wasm-pack build` succeeds for both web and node targets
- Node-side Wasm smoke paths exist for Hopfield, Dense Hopfield, RBM, and DAM
- deterministic Wasm regression checks now exist across all currently implemented model families
- browser pages for Hopfield, Dense Hopfield, RBM, and DAM now execute through Wasm-backed workers
- RBM and DAM now separate architecture, training, and query playback more clearly in the browser UI
- legacy TypeScript training/backend paths for the implemented models have been removed or reduced to lightweight UI/helper types
- dataset examples are now accessed through a consistent dataset help dialog instead of occupying the main page layout
- query editors now absorb example selection and degradation controls instead of keeping a separate memory-seed panel
- top hero cards no longer show backend labels
- large connectome and weight heatmaps now scale to their parent panels instead of relying on horizontal scrolling
- Hopfield pattern-set choices are now limited to synthetic toy pattern sets; MNIST and Fashion-MNIST are no longer shown there
- Vite dev-server file watching now uses polling inside the devcontainer because Docker Desktop bind mounts on Windows can miss filesystem change events
- the top network selector is no longer sticky
- the npm Wasm commands use explicit devcontainer Rust toolchain paths, so `npm run verify:all` works from terminal inside the container without extra shell setup

## Progress

Current verified state:

- devcontainer image builds successfully
- container-only validation works with container-owned `node_modules` and `wasm-core/target` volumes
- app build passes inside the devcontainer
- `npm run verify:all` passes inside the devcontainer
- DAM and RBM terminal smoke tests pass inside the devcontainer
- Rust/Wasm smoke tests pass for Hopfield, Dense Hopfield, RBM, and DAM inside the devcontainer
- Rust/Wasm invariant and regression checks pass for Hopfield, Dense Hopfield, RBM, and DAM inside the devcontainer
- the app is served successfully from the devcontainer and reachable from the host browser
- Hopfield, Dense Hopfield, RBM, and Dense Associative Memory routes were browser-verified from the host-facing dev URL
- Hopfield and Dense Hopfield were exercised interactively from the browser after the worker migration
- page headers now state each model's primary task explicitly instead of forcing classification language onto unsupervised models

Current migration status:

- Hopfield, Dense Hopfield, RBM, and DAM browser execution now runs through Wasm-backed workers
- workers are still model-specific runtime adapters
- the remaining architectural work is unifying those worker contracts behind the shared session API and adding the not-yet-implemented Boltzmann Machine family

## Architecture Document

The current architectural plan lives in [Architecture.md](C:\PRJ\hopfield-energy\Architecture.md).

It defines:

- the target separation between UI, runtime adapters, model API, and data pipeline
- the shared observability/snapshot model required for educational visualizations
- the intended backend direction for future Wasm or WebGPU execution
- the migration plan for RBM, Boltzmann Machine, and Dense Associative Memory variants

## Architecture

The project is being organized around a strict separation of concerns:

- **Core model logic** in `src/core/*`
  This contains training loops, retrieval dynamics, metrics, and numerical routines.
- **Data pipeline logic** in `src/data/*`
  This contains dataset archive parsing, sample selection, and runtime-specific loading adapters.
- **Browser runtime adapters** in workers/pages
  Workers coordinate long-running training/playback in the browser, but should not own the model logic itself.
- **CLI runtime adapters** in `scripts/*`
  These run the same shared core/data modules directly from Node for smoke tests and future regression checks.
- **React pages** in `src/pages/*`
  These should focus on presentation, controls, and visualization only.

That direction is now active in the browser/runtime layer for:

- **Hopfield**
- **Dense Hopfield**
- **Dense Associative Memory**
- **Restricted Boltzmann Machine**

## Shared Dataset Pipeline

The project uses bundled binary dataset archives in `public/datasets/`.

Shared modules:

- [src/data/datasetArchives.ts](C:\PRJ\hopfield-energy\src\data\datasetArchives.ts)
  Browser-safe archive parsing and helper utilities
- [scripts/shared/datasetArchivesNode.ts](C:\PRJ\hopfield-energy\scripts\shared\datasetArchivesNode.ts)
  Node adapter for loading the same archives from the filesystem

This means the browser and CLI paths now consume the same archive format and sample-selection logic.

## Local Development

### Prerequisites

- Node.js 20 or later
- npm

### Setup

```bash
git clone https://github.com/pzarzycki/hopfield-energy.git
cd hopfield-energy
npm install
```

## Dev Container

This repo now includes a custom dev container in `.devcontainer/` with:

- recent Node and npm
- Git
- Rust
- `wasm-pack`
- Python
- `ripgrep`
- host Git config mounted into the container
- host Codex/Codex-agent config mounted into the container
- container-owned `node_modules` and `wasm-core/target` volumes to avoid host/runtime cross-platform conflicts

The intended workflow is:

- source code stays in the mounted repo/worktree
- builds/tests/tooling run inside the container
- the same container definition can later be reused across multiple worktrees and agents
- `PROJECT_RUNTIME=devcontainer` is the canonical in-container runtime signal

HTTP ports exposed/forwarded by default inside the devcontainer:

- `5173` for `npm run dev`
- `4173` for `npm run preview`

The terminal-native way to use it is via the `devcontainer` CLI if installed locally.

Inside the devcontainer, the main validation command is:

```bash
npm run verify:all
```

That runs:

- Rust/Wasm unit tests
- Wasm package builds for web and node
- deterministic Wasm regression assertions
- Hopfield Wasm smoke test
- Dense Hopfield Wasm smoke test
- RBM Wasm smoke test
- DAM Wasm smoke test
- DAM CLI smoke test
- RBM CLI smoke test
- production app build

### Run The App

```bash
npm run dev
```

Inside the devcontainer the app listens on `http://localhost:5173/`. If you publish that container port to a different host port, open the mapped host URL instead.

### Production Build

Inside the devcontainer:

```bash
npm run build
```

This is the final step of `npm run verify:all`, which runs the full validation pipeline (Rust tests, Wasm builds, smoke tests, and finally the production Vite build).

### GitHub Pages Deployment

The project publishes to GitHub Pages via GitHub Actions using a **devcontainer-aligned Docker environment** to ensure dev/prod parity.

**Workflow behavior:**

- **Pull requests** to `main`: Run the full `npm run verify:all` validation pipeline in a containerized environment. No deployment occurs.
- **Push to `main`**: Run the same validation pipeline and automatically publish the resulting `dist/` to GitHub Pages.
- **Workflow dispatch**: Manual trigger to build and deploy immediately.

**Validation pipeline in CI** (`.github/workflows/deploy.yml`):

The GitHub Actions workflow runs inside a container (`mcr.microsoft.com/devcontainers/javascript-node:1-22-bookworm`) with Rust and `wasm-pack` installed, matching the local devcontainer setup. It executes:

```bash
npm ci
npm run verify:all
```

This ensures that:
- Rust/Wasm code compiles and tests pass
- Generated Wasm binaries are rebuilt and regression-tested
- TypeScript is type-checked
- ESLint passes
- The production app builds successfully
- Wasm smoke tests pass

Only if all steps succeed does the build artifact upload to GitHub Pages and (on `main` pushes) trigger deployment.

**Live site:** The app publishes to https://pzarzycki.github.io/hopfield-energy/ with the correct base path configured in `vite.config.ts`.

## CLI Smoke Tests

The smoke tests train models directly in Node using the shared core/data modules.

### Dense Associative Memory

```bash
npm run dam:smoke
```

Example:

```bash
npm run dam:smoke -- --dataset mnist --epochs 3 --hidden-units 64 --sharpness 6
```

### Restricted Boltzmann Machine

```bash
npm run rbm:smoke
```

Example:

```bash
npm run rbm:smoke -- --dataset mnist --epochs 3 --hidden-units 64 --visible-model bernoulli
```

Both commands emit JSON summaries with:

- training configuration
- reconstruction metrics before and after training
- final epoch metrics
- simple noisy-retrieval evaluation against the stored reference exemplars

### Wasm Core

These commands are intended to run inside the devcontainer:

```bash
npm run wasm:test
npm run wasm:build
npm run wasm:regression
npm run wasm:dam:smoke
```

## Current Runtime Status

- **Browser UI:** available and Wasm-backed for all currently implemented model families
- **CLI smoke tests:** available for DAM and RBM
- **Rust/Wasm core:** baselines available for Hopfield, Dense Hopfield, RBM, and DAM
- **Web workers:** used for browser-side long-running model execution
- **WebGPU:** partial/experimental in the codebase, not yet the default cross-runtime execution path

The next architectural step is to make every model follow the same pattern:

1. Shared core model module
2. Shared dataset/runtime pipeline
3. Browser worker adapter
4. CLI smoke/regression command
5. Optional WebGPU backend behind the same model interface

## Project Structure

```text
hopfield-energy/
├── public/
│   └── datasets/                 # bundled binary dataset archives
├── scripts/
│   ├── dam-smoke.ts              # DAM CLI smoke test
│   ├── rbm-smoke.ts              # RBM CLI smoke test
│   └── build_dataset_archives.py
├── src/
│   ├── core/                     # model/training/retrieval logic
│   ├── data/                     # shared dataset parsing/loading
│   ├── features/                 # reusable visualization widgets
│   ├── pages/                    # React presentation pages
│   ├── workers/                  # browser runtime worker adapters
│   ├── App.tsx
│   ├── App.css
│   ├── index.css
│   └── main.tsx
├── package.json
└── vite.config.ts
```

## Deployment

The site is deployed to GitHub Pages from the production build output.

## License

MIT
