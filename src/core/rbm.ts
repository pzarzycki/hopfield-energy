import { PATTERN_SIDE, PATTERN_SIZE } from "./patternSets";

export type RBMVisibleModel = "bernoulli" | "gaussian";

export type RBMBackendKind = "wasm-core";

export interface RBMModel {
  visibleBias: Float32Array;
  hiddenBias: Float32Array;
  weights: Float32Array;
  visibleUnits: number;
  hiddenUnits: number;
  visibleModel: RBMVisibleModel;
}

export interface RBMTrainingConfig {
  hiddenUnits: number;
  epochs: number;
  learningRate: number;
  visibleModel: RBMVisibleModel;
  batchSize?: number;
  cdSteps?: number;
  momentum?: number;
  weightDecay?: number;
}

export interface RBMFeatureMap {
  hiddenIndex: number;
  map: Float32Array;
}

export interface RBMEpochMetrics {
  epoch: number;
  reconstructionError: number;
  contrastiveGap: number;
  freeEnergy: number;
  hiddenActivation: number;
  weightMeanAbs: number;
}

export interface RBMSnapshot {
  visible: Float32Array;
  reconstruction: Float32Array;
  hiddenProbabilities: Float32Array;
  hiddenState: Uint8Array;
  freeEnergy: number;
  reconstructionError: number;
  step: number;
  converged: boolean;
  matchedPatternIndex: number;
}

const EPSILON = 1e-6;

export function quantizeVisiblePattern(pattern: Float32Array, visibleModel: RBMVisibleModel, threshold = 0.5): Float32Array {
  if (visibleModel === "gaussian") {
    return pattern.slice();
  }

  return Float32Array.from(pattern, (value) => (value >= threshold ? 1 : 0));
}

export function buildFeatureStrength(model: RBMModel, hiddenUnits = model.hiddenUnits): Float32Array {
  const featureStrength = new Float32Array(hiddenUnits);
  for (let hiddenIndex = 0; hiddenIndex < hiddenUnits; hiddenIndex += 1) {
    const rowOffset = hiddenIndex * PATTERN_SIZE;
    let norm = 0;
    for (let visibleIndex = 0; visibleIndex < PATTERN_SIZE; visibleIndex += 1) {
      norm += Math.abs(model.weights[rowOffset + visibleIndex]);
    }
    featureStrength[hiddenIndex] = norm / PATTERN_SIZE;
  }

  return featureStrength;
}

export function buildFeatureMaps(model: RBMModel, featureStrength: Float32Array): RBMFeatureMap[] {
  return Array.from({ length: model.hiddenUnits }, (_, hiddenIndex) => ({ hiddenIndex }))
    .sort((left, right) => featureStrength[right.hiddenIndex] - featureStrength[left.hiddenIndex])
    .map(({ hiddenIndex }) => {
      const map = new Float32Array(PATTERN_SIZE);
      const rowOffset = hiddenIndex * PATTERN_SIZE;
      let maxAbs = EPSILON;
      for (let visibleIndex = 0; visibleIndex < PATTERN_SIZE; visibleIndex += 1) {
        maxAbs = Math.max(maxAbs, Math.abs(model.weights[rowOffset + visibleIndex]));
      }

      for (let visibleIndex = 0; visibleIndex < PATTERN_SIZE; visibleIndex += 1) {
        const weight = model.weights[rowOffset + visibleIndex];
        map[visibleIndex] = weight / (2 * maxAbs) + 0.5;
      }

      return { hiddenIndex, map };
    });
}

export function buildHiddenGrid(probabilities: Float32Array): Float32Array {
  const side = Math.ceil(Math.sqrt(probabilities.length));
  const padded = new Float32Array(side * side);
  padded.set(probabilities);
  return padded;
}

export function getHiddenGridSide(hiddenUnits: number): number {
  return Math.ceil(Math.sqrt(hiddenUnits));
}

export function applyVisibleNoise(
  pattern: Float32Array,
  corruptionPercent: number,
  obfuscationPercent: number,
  visibleModel: RBMVisibleModel,
): Float32Array {
  const next = pattern.slice();
  const total = next.length;
  const corruptionCount = Math.round((total * corruptionPercent) / 100);
  const obfuscationCount = Math.round((total * obfuscationPercent) / 100);
  const indices = new Uint16Array(total);

  for (let index = 0; index < total; index += 1) {
    indices[index] = index;
  }

  for (let index = total - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    const current = indices[index];
    indices[index] = indices[swapIndex];
    indices[swapIndex] = current;
  }

  for (let index = 0; index < corruptionCount; index += 1) {
    const target = indices[index];
    next[target] = visibleModel === "bernoulli" ? (next[target] > 0.5 ? 0 : 1) : 1 - next[target];
  }

  for (let index = 0; index < obfuscationCount; index += 1) {
    next[indices[corruptionCount + index]] = 0;
  }

  return next;
}

export function createBlankVisiblePattern(): Float32Array {
  return new Float32Array(PATTERN_SIZE);
}

export function visibleDensity(pattern: Float32Array): number {
  let total = 0;
  for (let index = 0; index < pattern.length; index += 1) {
    total += pattern[index];
  }
  return total / pattern.length;
}

export { PATTERN_SIDE };
