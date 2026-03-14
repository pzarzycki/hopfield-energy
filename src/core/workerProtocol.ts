import type { HopfieldInitResult, HopfieldSnapshot, UpdateRule } from "./hopfield";

export interface WorkerInitializeMessage {
  type: "initialize";
  patternSetId: string;
  updateRule: UpdateRule;
}

export interface WorkerSetQueryMessage {
  type: "setQuery";
  pattern: Int8Array;
}

export interface WorkerStepMessage {
  type: "step";
}

export interface WorkerPlayMessage {
  type: "play";
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
