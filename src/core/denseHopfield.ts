export interface DenseSnapshot {
  state: Float32Array;
  energy: number;
  step: number;
  delta: number;
  matchedPatternIndex: number;
  converged: boolean;
  attention: Float32Array;
  entropy: number;
  topAttention: number;
}

export function createDenseSnapshot(state: Float32Array): DenseSnapshot {
  return {
    state,
    energy: 0,
    step: 0,
    delta: 0,
    matchedPatternIndex: -1,
    converged: false,
    attention: new Float32Array(0),
    entropy: 0,
    topAttention: 0,
  };
}
