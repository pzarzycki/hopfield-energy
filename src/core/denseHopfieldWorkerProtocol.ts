import type { DenseSnapshot } from "./denseHopfield";
import type { DatasetName } from "../data/datasetArchives";

export interface DenseHopfieldMemoryView {
  labels: string[];
  patterns: Float32Array[];
  similarityMatrix: Float32Array;
  maxSimilarityAbs: number;
  description: string;
}

export interface DenseHopfieldInitializeMessage {
  type: "initialize";
  datasetName: DatasetName;
  beta: number;
}

export interface DenseHopfieldSetQueryMessage {
  type: "setQuery";
  pattern: Float32Array;
}

export interface DenseHopfieldSetBetaMessage {
  type: "setBeta";
  beta: number;
}

export interface DenseHopfieldStepMessage {
  type: "step";
}

export interface DenseHopfieldPlayMessage {
  type: "play";
  intervalMs: number;
  maxSteps: number;
}

export interface DenseHopfieldPauseMessage {
  type: "pause";
}

export interface DenseHopfieldResetMessage {
  type: "reset";
}

export type DenseHopfieldWorkerRequest =
  | DenseHopfieldInitializeMessage
  | DenseHopfieldSetQueryMessage
  | DenseHopfieldSetBetaMessage
  | DenseHopfieldStepMessage
  | DenseHopfieldPlayMessage
  | DenseHopfieldPauseMessage
  | DenseHopfieldResetMessage;

export interface DenseHopfieldInitializedMessage {
  type: "initialized";
  backend: "wasm-core";
  memorySet: DenseHopfieldMemoryView;
  snapshot: DenseSnapshot;
}

export interface DenseHopfieldSnapshotMessage {
  type: "snapshot";
  snapshot: DenseSnapshot;
}

export interface DenseHopfieldPausedMessage {
  type: "paused";
}

export interface DenseHopfieldErrorMessage {
  type: "error";
  message: string;
}

export type DenseHopfieldWorkerResponse =
  | DenseHopfieldInitializedMessage
  | DenseHopfieldSnapshotMessage
  | DenseHopfieldPausedMessage
  | DenseHopfieldErrorMessage;
