import type { HopfieldInitResult, HopfieldSnapshot } from "./hopfield";
import type { ConvergenceRuleConfig, LearningRuleConfig } from "./hopfieldRules";

export interface WorkerInitializeMessage {
  type: "initialize";
  patternSetId: string;
  learningConfig: LearningRuleConfig;
}

export interface WorkerSetQueryMessage {
  type: "setQuery";
  pattern: Int8Array;
}

export interface WorkerStepMessage {
  type: "step";
  convergenceConfig: ConvergenceRuleConfig;
}

export interface WorkerPlayMessage {
  type: "play";
  convergenceConfig: ConvergenceRuleConfig;
  intervalMs: number;
  maxSteps: number;
}

export interface WorkerPauseMessage {
  type: "pause";
}

export interface WorkerResetMessage {
  type: "reset";
}

export type WorkerRequest =
  | WorkerInitializeMessage
  | WorkerSetQueryMessage
  | WorkerStepMessage
  | WorkerPlayMessage
  | WorkerPauseMessage
  | WorkerResetMessage;

export interface WorkerReadyMessage extends HopfieldInitResult {
  type: "ready";
  backend: "wasm-core";
  snapshot: HopfieldSnapshot;
}

export interface WorkerSnapshotMessage {
  type: "snapshot";
  snapshot: HopfieldSnapshot;
}

export interface WorkerPausedMessage {
  type: "paused";
}

export interface WorkerErrorMessage {
  type: "error";
  message: string;
}

export type WorkerResponse =
  | WorkerReadyMessage
  | WorkerSnapshotMessage
  | WorkerPausedMessage
  | WorkerErrorMessage;
