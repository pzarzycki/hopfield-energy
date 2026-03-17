import type { DatasetName } from "../data/datasetArchives";
import type { RBMEpochMetrics, RBMFeatureMap, RBMModel, RBMSnapshot, RBMVisibleModel } from "./rbm";

export type RBMBackendKind = "wasm-core";

export interface RBMWorkerInitializeMessage {
  type: "initialize";
  datasetName: DatasetName;
  visibleModel: RBMVisibleModel;
  hiddenUnits: number;
  epochs: number;
  learningRate: number;
  batchSize: number;
  cdSteps: number;
  momentum: number;
  weightDecay: number;
}

export interface RBMWorkerSetQueryMessage {
  type: "setQuery";
  pattern: Float32Array;
}

export interface RBMWorkerStartTrainingMessage {
  type: "startTraining";
  intervalMs: number;
}

export interface RBMWorkerPauseTrainingMessage {
  type: "pauseTraining";
}

export interface RBMWorkerTrainEpochMessage {
  type: "trainEpoch";
}

export interface RBMWorkerResetTrainingMessage {
  type: "resetTraining";
}

export interface RBMWorkerStartPlaybackMessage {
  type: "startPlayback";
  intervalMs: number;
  maxSteps: number;
}

export interface RBMWorkerPausePlaybackMessage {
  type: "pausePlayback";
}

export interface RBMWorkerStepPlaybackMessage {
  type: "stepPlayback";
}

export interface RBMWorkerResetPlaybackMessage {
  type: "resetPlayback";
}

export type RBMWorkerRequest =
  | RBMWorkerInitializeMessage
  | RBMWorkerSetQueryMessage
  | RBMWorkerStartTrainingMessage
  | RBMWorkerPauseTrainingMessage
  | RBMWorkerTrainEpochMessage
  | RBMWorkerResetTrainingMessage
  | RBMWorkerStartPlaybackMessage
  | RBMWorkerPausePlaybackMessage
  | RBMWorkerStepPlaybackMessage
  | RBMWorkerResetPlaybackMessage;

export interface RBMWorkerStatePayload {
  backend: RBMBackendKind;
  model: RBMModel;
  snapshot: RBMSnapshot;
  featureMaps: RBMFeatureMap[];
  epoch: number;
  trainingError: number;
}

export interface RBMWorkerInitializedMessage extends RBMWorkerStatePayload {
  type: "initialized";
}

export interface RBMWorkerTrainingEpochMessage extends RBMWorkerStatePayload {
  type: "trainingEpoch";
  metrics: RBMEpochMetrics;
}

export interface RBMWorkerSnapshotMessage {
  type: "snapshot";
  snapshot: RBMSnapshot;
}

export interface RBMWorkerTrainingPausedMessage {
  type: "trainingPaused";
}

export interface RBMWorkerPlaybackPausedMessage {
  type: "playbackPaused";
}

export interface RBMWorkerErrorMessage {
  type: "error";
  message: string;
}

export type RBMWorkerResponse =
  | RBMWorkerInitializedMessage
  | RBMWorkerTrainingEpochMessage
  | RBMWorkerSnapshotMessage
  | RBMWorkerTrainingPausedMessage
  | RBMWorkerPlaybackPausedMessage
  | RBMWorkerErrorMessage;
