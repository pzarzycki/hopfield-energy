export interface HopfieldSnapshot {
  state: Int8Array;
  energy: number;
  step: number;
  changedCount: number;
  matchedPatternIndex: number;
  converged: boolean;
}

export interface HopfieldInitResult {
  weights: Float32Array;
  maxWeightAbs: number;
  patternCount: number;
}
