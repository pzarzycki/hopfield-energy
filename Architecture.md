# Architecture

## Purpose

This project is an educational workbench for energy-based memory models. The primary product is not only pattern reconstruction or generation, but also **introspection**: the engineer should be able to inspect network structure, neuron states, energy terms, weights, biases, phases, and intermediate dynamics at each step.

The architecture must therefore optimize for:

- **Clean decomposition**
- **One computational implementation per model family**
- **UI/runtime separation**
- **CLI testability**
- **Inspectability**
- **Future acceleration paths such as Wasm or WebGPU**

## Design Principles

1. **Presentation is not computation**
   React pages render controls, charts, matrices, and canvases. They do not own model math, dataset logic, or training loops.

2. **Data pipeline is shared**
   Browser, CLI, and future headless test runners must use the same dataset archive format, parsing, and pattern transforms.

3. **Model API is runtime-neutral**
   UI, worker, and CLI should depend on a shared model/session contract, not on implementation details.

4. **Toy examples still need serious observability**
   All models must expose rich snapshots of their internal state, not only final outputs.

5. **Acceleration is a backend concern**
   Wasm, WebGPU, or other compute strategies sit behind a model backend contract. The page and CLI should not know which one is active beyond metadata.

6. **No duplicated “mock” model implementations**
   We should not evolve separate UI math, worker math, and CLI math for the same model.

## Conceptual Layers

```text
UI / Presentation
  React pages, controls, plots, matrices, canvases

Runtime Adapters
  Browser workers, CLI commands, future headless-browser harnesses

Model Service API
  Shared session-oriented contract for training, reconstruction, generation, and inspection

Model Backends
  Concrete compute engine implementations (initially TypeScript, later Wasm or WebGPU-backed)

Data Pipeline
  Dataset archives, parsing, sampling, representation transforms, augmentation/noise
```

## Repository Shape

Target structure:

```text
src/
  core/
    api/                  # shared runtime-neutral interfaces and types
    models/               # model-family contracts and orchestration
    backends/             # concrete compute backends
    observability/        # snapshot/trace/tensor helpers
  data/
    archives/             # archive schema, parsing, browser-side loading
    transforms/           # binary, bipolar, grayscale, noise, corruption
    sampling/             # class-balanced views, training/reference/query subsets
  runtime/
    browser/              # worker adapters and browser wiring
    cli/                  # thin CLI entrypoints and runtime adapters
  pages/                  # React-only presentation
  features/               # reusable widgets
scripts/
  shared/                 # Node-only adapters for CLI/runtime
  smoke/                  # model smoke tests
```

The current repo is not yet fully migrated to this layout, but this is the intended direction.

## Data API

The data layer exists to provide a **stable pattern source** to all models and runtimes.

### Core Types

```ts
type DatasetId = "mnist" | "fashion-mnist" | string;

type PatternEncoding = "grayscale-f32" | "binary-f32" | "bipolar-i8";

interface DatasetSample {
  id: string;
  label: number;
  labelName: string;
  pattern: Uint8Array | Float32Array | Int8Array;
}

interface DatasetArchive {
  id: DatasetId;
  name: string;
  rows: number;
  cols: number;
  memorySamplesPerClass: number;
  samples: DatasetSample[];
}
```

### Data Responsibilities

- Parse dataset archives
- Load datasets from browser assets or Node filesystem adapters
- Produce transformed views:
  - grayscale
  - binarized
  - bipolar
- Produce subsets:
  - training set
  - memory/reference set
  - evaluation set
- Apply controlled perturbations:
  - corruption
  - occlusion
  - inversion
  - quantization

### Data Services

```ts
interface DatasetRepository {
  loadArchive(id: DatasetId): Promise<DatasetArchive>;
}

interface PatternViewRequest {
  encoding: PatternEncoding;
  threshold?: number;
}

interface DatasetView<TPattern> {
  archive: DatasetArchive;
  patterns: TPattern[];
  labels: string[];
  rows: number;
  cols: number;
}

interface DatasetPipeline {
  materialize<TPattern>(archive: DatasetArchive, request: PatternViewRequest): DatasetView<TPattern>;
  selectReferenceSamples<TPattern>(view: DatasetView<TPattern>, samplesPerClass?: number): DatasetView<TPattern>;
  applyNoise(pattern: Float32Array, options: NoiseOptions): Float32Array;
}
```

### Rule

Pages and model sessions must not parse archives directly. They go through the data API.

## Model API

The model layer must present a shared shape across:

- Hopfield variants
- Restricted Boltzmann Machine
- Boltzmann Machine
- Dense Associative Memory
- Future layered Dense Associative Memory variants

### Model Families

```ts
type ModelFamily =
  | "hopfield"
  | "dense-hopfield"
  | "rbm"
  | "boltzmann-machine"
  | "dense-associative-memory";
```

### Topology Model

Because this project is educational, topology must be explicit and inspectable.

```ts
interface LayerSpec {
  id: string;
  label: string;
  role: "visible" | "hidden" | "memory" | "latent";
  units: number;
  shape?: { rows: number; cols: number };
  stateKinds: Array<"activation" | "probability" | "sample" | "bias">;
}

interface ConnectionSpec {
  id: string;
  from: string;
  to: string;
  kind: "weights" | "symmetric-weights" | "bias" | "velocity";
  rows: number;
  cols: number;
}

interface ModelTopology {
  family: ModelFamily;
  title: string;
  layers: LayerSpec[];
  connections: ConnectionSpec[];
}
```

This allows the UI to render architecture cards, labels, matrices, and layer summaries in a model-agnostic way.

### Session-Oriented Execution

Each model should run through an inspectable session.

```ts
interface ModelSessionConfig {
  seed?: number;
  backend?: string;
}

interface TrainOptions {
  epochs?: number;
  steps?: number;
  batchSize?: number;
}

interface ReconstructionOptions {
  maxSteps: number;
  tolerance: number;
}

interface GenerationOptions {
  maxSteps: number;
  temperature?: number;
}

interface ModelSession {
  getTopology(): ModelTopology;
  getBackendInfo(): BackendInfo;
  initialize(init: ModelInitialization): Promise<void>;
  setQuery(pattern: Float32Array): Promise<void>;
  trainEpoch(): Promise<ModelTrace>;
  reconstruct(options: ReconstructionOptions): Promise<ModelTrace>;
  generate(options: GenerationOptions): Promise<ModelTrace>;
  step(): Promise<ModelSnapshot>;
  reset(): Promise<ModelSnapshot>;
  inspect(): Promise<ModelSnapshot>;
  exportState(): Promise<ModelArtifact>;
  dispose(): Promise<void>;
}
```

### Initialization Contract

Different models need different hyperparameters, but the initialization should still follow a common envelope:

```ts
interface ModelInitialization {
  family: ModelFamily;
  architecture: Record<string, number | string | boolean>;
  training: Record<string, number | string | boolean>;
  dataset?: PreparedDataset;
}
```

Model-specific configuration can be strongly typed behind this envelope, but the session API remains common.

## Observability API

This is the most important part for the UI.

Every model must expose internal details through a snapshot schema rather than ad hoc page-specific objects.

### Snapshot Model

```ts
interface NamedTensor {
  id: string;
  label: string;
  role: "state" | "parameter" | "metric" | "auxiliary";
  shape: number[];
  format: "f32" | "u8" | "i8" | "u32";
  values: Float32Array | Uint8Array | Int8Array | Uint32Array;
}

interface LayerSnapshot {
  layerId: string;
  activations?: Float32Array;
  probabilities?: Float32Array;
  samples?: Uint8Array | Float32Array;
  bias?: Float32Array;
}

interface ConnectionSnapshot {
  connectionId: string;
  values: Float32Array;
  rows: number;
  cols: number;
}

interface ScalarMetric {
  id: string;
  label: string;
  value: number;
}

interface ModelSnapshot {
  phase: "idle" | "training-positive" | "training-negative" | "reconstruction" | "generation";
  step: number;
  converged: boolean;
  matchedPatternIndex?: number;
  visiblePattern?: Float32Array;
  outputPattern?: Float32Array;
  layers: LayerSnapshot[];
  connections: ConnectionSnapshot[];
  metrics: ScalarMetric[];
}
```

### Trace Model

Training and playback should return a trace of rich snapshots, not just a final tensor.

```ts
interface ModelTrace {
  topology: ModelTopology;
  snapshots: ModelSnapshot[];
  final: ModelSnapshot;
}
```

The UI can then decide whether to:

- animate step-by-step
- plot metrics over time
- render matrices
- render hidden grids
- expose raw tensors for debugging

## Model-Specific Expectations

### Hopfield / Dense Hopfield

- Visible state
- Energy trace
- Update deltas
- Similarity or attention state
- Stored memory set metadata

### RBM / Boltzmann Machine

- Visible probabilities/samples
- Hidden probabilities/samples
- Weight matrix
- Bias vectors
- Positive and negative phase views
- Reconstruction error
- Contrastive gap
- Free energy
- Optional persistent chains later

### Dense Associative Memory

- Visible reconstruction
- Hidden activations
- Hidden scores
- Prototype matrix
- Winner index / entropy / activation mass
- Iterative replay trace
- Future layered variants: per-layer hidden states and inter-layer matrices

## Backend Contract

Backends implement model execution, not the page.

```ts
interface BackendInfo {
  kind: string;
  version?: string;
  capabilities: string[];
}

interface ModelBackend {
  createSession(config?: ModelSessionConfig): Promise<ModelSession>;
}
```

Initial backend options:

- `typescript-core`
  Current practical baseline while the architecture is still evolving
- `wasm-core`
  Active migration target for a single shared implementation across browser and CLI
- `webgpu`
  Optional acceleration path when it can be kept as a thin runtime around one compute core

### Important Rule

We should not evolve divergent, model-specific page math and worker math. Backends should sit behind the session contract so browser workers and CLI both talk to the same model service API.

## Runtime Adapters

### Browser

- React page calls browser runtime adapter
- Adapter talks to worker or directly to backend
- Worker owns scheduling, timers, cancellation, and message passing
- Worker does not own model math

### CLI

- CLI command loads dataset through Node adapter
- CLI creates model session
- CLI trains or reconstructs
- CLI prints structured JSON traces or summaries

### Headless Testing

If we later choose a browser-only compute backend such as WebGPU, CLI should run through a headless runtime adapter rather than a second model implementation.

## Dev Container Requirements

The development container should support:

- Node and npm
- TypeScript and `tsx`
- Git
- Rust toolchain and `wasm-pack`
- Python for dataset tooling if still needed
- mounted host Git config
- mounted host Codex/Codex-agent config

The container is a toolchain boundary, not a source-of-truth boundary:

- source code stays in the mounted workspace
- builds/tests/tooling run inside the container

## Migration Plan

### Phase 1: API Stabilization

- Move to `core/api` contracts
- Normalize dataset pipeline entrypoints
- Normalize snapshot/trace schema
- Make pages depend on generic snapshot structures where practical

### Phase 2: Runtime Cleanup

- Convert workers into thin runtime adapters
- Move CLI smoke scripts onto shared session APIs
- Keep browser and CLI using the same model service shapes

### Phase 3: Backend Consolidation

- Decide whether the long-term shared compute core is:
  - Wasm-first
  - WebGPU-first via headless browser runtime
- Port one model family end-to-end before migrating all models

### Phase 4: Model Expansion

- Boltzmann Machine
- Layered Dense Associative Memory
- richer generation workflows
- richer snapshot visualizations

## Recommended Immediate Next Steps

1. Create `core/api` types for topology, snapshots, traces, and sessions.
2. Refactor DAM and RBM workers to return `ModelSnapshot`-shaped data.
3. Normalize dataset access behind repository + pipeline helpers.
4. Decide between:
   - Wasm shared compute core
   - headless-browser WebGPU runtime
5. Only then migrate model internals.

## Current Implementation Note

An initial Rust/Wasm Dense Associative Memory core now exists under `wasm-core/` and is validated inside the devcontainer with:

- `npm run wasm:test`
- `npm run wasm:build`

This should be treated as the first concrete backend migration step.
