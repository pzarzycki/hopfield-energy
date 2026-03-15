import { PATTERN_SIZE } from "./patternSets";

export interface DenseMemorySet {
  labels: string[];
  patterns: Float32Array[];
  embeddings: Float32Array[];
  similarityMatrix: Float32Array;
  maxSimilarityAbs: number;
  description: string;
}

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

function toCentered(image: Float32Array): Float32Array {
  const centered = new Float32Array(image.length);
  for (let index = 0; index < image.length; index += 1) {
    centered[index] = image[index] * 2 - 1;
  }
  return centered;
}

function dot(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let index = 0; index < a.length; index += 1) {
    sum += a[index] * b[index];
  }
  return sum;
}

function softmax(scores: Float32Array): Float32Array {
  let max = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < scores.length; index += 1) {
    max = Math.max(max, scores[index]);
  }

  const weights = new Float32Array(scores.length);
  let total = 0;
  for (let index = 0; index < scores.length; index += 1) {
    const value = Math.exp(scores[index] - max);
    weights[index] = value;
    total += value;
  }

  if (total === 0) {
    return weights;
  }

  for (let index = 0; index < weights.length; index += 1) {
    weights[index] /= total;
  }

  return weights;
}

function logSumExp(scores: Float32Array): number {
  let max = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < scores.length; index += 1) {
    max = Math.max(max, scores[index]);
  }

  let total = 0;
  for (let index = 0; index < scores.length; index += 1) {
    total += Math.exp(scores[index] - max);
  }

  return max + Math.log(total || 1);
}

function computeEnergy(centeredState: Float32Array, scores: Float32Array, beta: number): number {
  const safeBeta = Math.max(beta, 1e-6);
  return dot(centeredState, centeredState) / (2 * PATTERN_SIZE) - logSumExp(scores) / safeBeta;
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

export function buildDenseMemorySet(labels: string[], patterns: Float32Array[]): DenseMemorySet {
  const embeddings = patterns.map((pattern) => toCentered(pattern));
  const similarityMatrix = new Float32Array(patterns.length * patterns.length);
  let maxSimilarityAbs = 1;

  for (let row = 0; row < embeddings.length; row += 1) {
    for (let column = 0; column < embeddings.length; column += 1) {
      const similarity = dot(embeddings[row], embeddings[column]) / PATTERN_SIZE;
      similarityMatrix[row * embeddings.length + column] = similarity;
      maxSimilarityAbs = Math.max(maxSimilarityAbs, Math.abs(similarity));
    }
  }

  return {
    labels,
    patterns,
    embeddings,
    similarityMatrix,
    maxSimilarityAbs,
    description: `${patterns.length} stored memories over ${PATTERN_SIZE} pixels each.`,
  };
}

export function evaluateDenseState(state: Float32Array, memorySet: DenseMemorySet, beta: number, step: number): DenseSnapshot {
  const centeredState = toCentered(state);
  const scores = new Float32Array(memorySet.embeddings.length);
  for (let index = 0; index < memorySet.embeddings.length; index += 1) {
    scores[index] = (beta * dot(memorySet.embeddings[index], centeredState)) / PATTERN_SIZE;
  }

  const attention = softmax(scores);
  let matchedPatternIndex = -1;
  let topAttention = Number.NEGATIVE_INFINITY;
  let entropy = 0;
  for (let index = 0; index < attention.length; index += 1) {
    const weight = attention[index];
    if (weight > topAttention) {
      topAttention = weight;
      matchedPatternIndex = index;
    }
    if (weight > 0) {
      entropy -= weight * Math.log(weight);
    }
  }

  return {
    state: state.slice(),
    energy: computeEnergy(centeredState, scores, beta),
    step,
    delta: 0,
    matchedPatternIndex,
    converged: false,
    attention,
    entropy,
    topAttention: topAttention > Number.NEGATIVE_INFINITY ? topAttention : 0,
  };
}

export function stepDenseHopfield(
  state: Float32Array,
  memorySet: DenseMemorySet,
  beta: number,
  step: number,
  tolerance: number,
): DenseSnapshot {
  const current = evaluateDenseState(state, memorySet, beta, step);
  const nextState = new Float32Array(PATTERN_SIZE);

  for (let memoryIndex = 0; memoryIndex < memorySet.patterns.length; memoryIndex += 1) {
    const weight = current.attention[memoryIndex];
    const pattern = memorySet.patterns[memoryIndex];
    for (let pixelIndex = 0; pixelIndex < PATTERN_SIZE; pixelIndex += 1) {
      nextState[pixelIndex] += weight * pattern[pixelIndex];
    }
  }

  let delta = 0;
  for (let index = 0; index < nextState.length; index += 1) {
    delta += Math.abs(nextState[index] - state[index]);
  }
  delta /= PATTERN_SIZE;

  const next = evaluateDenseState(nextState, memorySet, beta, step + 1);
  next.delta = delta;
  next.converged = delta <= tolerance;
  return next;
}
