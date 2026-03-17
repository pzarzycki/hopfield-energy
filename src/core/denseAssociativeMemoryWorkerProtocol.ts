import type { DatasetName } from "../data/datasetArchives";
import type {
  DenseAssociativeActivation,
  DenseAssociativeEpochMetrics,
  DenseAssociativeFeatureMap,
  DenseAssociativeMemoryModel,
  DenseAssociativeSnapshot,
} from "./denseAssociativeMemory";

export interface DenseAssociativeInitializeMessage {
  type: "initialize";
  datasetName: DatasetName;
  hiddenUnits: number;
  epochs: number;
  learningRate: number;
  batchSize: number;
  sharpness: number;
  activation: DenseAssociativeActivation;
  momentum: number;
  weightDecay: number;
}

export interface DenseAssociativeSetQueryMessage {
  type: "setQuery";
  pattern: Float32Array;
}

export interface DenseAssociativeStartTrainingMessage {
  type: "startTraining";
  intervalMs: number;
}

export interface DenseAssociativePauseTrainingMessage {
  type: "pauseTraining";
}

export interface DenseAssociativeTrainEpochMessage {
  type: "trainEpoch";
}

export interface DenseAssociativeResetTrainingMessage {
  type: "resetTraining";
}

export interface DenseAssociativeStartPlaybackMessage {
  type: "startPlayback";
  intervalMs: number;
  maxSteps: number;
}

export interface DenseAssociativePausePlaybackMessage {
  type: "pausePlayback";
}

export interface DenseAssociativeStepPlaybackMessage {
  type: "stepPlayback";
}

export interface DenseAssociativeResetPlaybackMessage {
  type: "resetPlayback";
}

export type DenseAssociativeWorkerRequest =
  | DenseAssociativeInitializeMessage
  | DenseAssociativeSetQueryMessage
  | DenseAssociativeStartTrainingMessage
  | DenseAssociativePauseTrainingMessage
  | DenseAssociativeTrainEpochMessage
  | DenseAssociativeResetTrainingMessage
  | DenseAssociativeStartPlaybackMessage
  | DenseAssociativePausePlaybackMessage
  | DenseAssociativeStepPlaybackMessage
  | DenseAssociativeResetPlaybackMessage;

export interface DenseAssociativeWorkerStatePayload {
  backend: "cpu" | "wasm-core";
  model: DenseAssociativeMemoryModel;
  snapshot: DenseAssociativeSnapshot;
  featureMaps: DenseAssociativeFeatureMap[];
  epoch: number;
  trainingError: number;
}

export interface DenseAssociativeInitializedMessage extends DenseAssociativeWorkerStatePayload {
  type: "initialized";
}

export interface DenseAssociativeTrainingEpochMessage extends DenseAssociativeWorkerStatePayload {
  type: "trainingEpoch";
  metrics: DenseAssociativeEpochMetrics;
}

export interface DenseAssociativeSnapshotMessage {
  type: "snapshot";
  snapshot: DenseAssociativeSnapshot;
}

export interface DenseAssociativeTrainingPausedMessage {
  type: "trainingPaused";
}

export interface DenseAssociativePlaybackPausedMessage {
  type: "playbackPaused";
}

export interface DenseAssociativeErrorMessage {
  type: "error";
  message: string;
}

export type DenseAssociativeWorkerResponse =
  | DenseAssociativeInitializedMessage
  | DenseAssociativeTrainingEpochMessage
  | DenseAssociativeSnapshotMessage
  | DenseAssociativeTrainingPausedMessage
  | DenseAssociativePlaybackPausedMessage
  | DenseAssociativeErrorMessage;
