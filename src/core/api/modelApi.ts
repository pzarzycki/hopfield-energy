export type ModelFamily =
  | "hopfield"
  | "dense-hopfield"
  | "rbm"
  | "boltzmann-machine"
  | "dense-associative-memory";

export interface LayerSpec {
  id: string;
  label: string;
  role: "visible" | "hidden" | "memory" | "latent";
  units: number;
  shape?: { rows: number; cols: number };
  stateKinds: Array<"activation" | "probability" | "sample" | "bias">;
}

export interface ConnectionSpec {
  id: string;
  from: string;
  to: string;
  kind: "weights" | "symmetric-weights" | "bias" | "velocity";
  rows: number;
  cols: number;
}

export interface ModelTopology {
  family: ModelFamily;
  title: string;
  layers: LayerSpec[];
  connections: ConnectionSpec[];
}

export interface LayerSnapshot {
  layerId: string;
  activations?: Float32Array;
  probabilities?: Float32Array;
  samples?: Uint8Array | Float32Array;
  bias?: Float32Array;
}

export interface ConnectionSnapshot {
  connectionId: string;
  values: Float32Array;
  rows: number;
  cols: number;
}

export interface ScalarMetric {
  id: string;
  label: string;
  value: number;
}

export interface ModelSnapshot {
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

export interface ModelTrace {
  topology: ModelTopology;
  snapshots: ModelSnapshot[];
  final: ModelSnapshot;
}

export interface BackendInfo {
  kind: string;
  version?: string;
  capabilities: string[];
}

export interface ModelSessionConfig {
  seed?: number;
  backend?: string;
}

export interface ReconstructionOptions {
  maxSteps: number;
  tolerance: number;
}

export interface GenerationOptions {
  maxSteps: number;
  temperature?: number;
}

export interface ModelInitialization {
  family: ModelFamily;
  architecture: Record<string, number | string | boolean>;
  training: Record<string, number | string | boolean>;
  dataset?: {
    patterns: Float32Array[];
    labels?: string[];
    referencePatterns?: Float32Array[];
  };
}

export interface ModelSession {
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
  dispose(): Promise<void>;
}
