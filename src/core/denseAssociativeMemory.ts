import { PATTERN_SIDE, PATTERN_SIZE } from "./patternSets";

export type DenseAssociativeActivation = "relu-power" | "signed-power" | "softmax";

export interface DenseAssociativeMemoryModel {
  weights: Float32Array;
  visibleUnits: number;
  hiddenUnits: number;
  activation: DenseAssociativeActivation;
  sharpness: number;
}

export interface DenseAssociativeMemoryTrainingConfig {
  hiddenUnits: number;
  epochs: number;
  learningRate: number;
  batchSize?: number;
  sharpness: number;
  activation: DenseAssociativeActivation;
  momentum?: number;
  weightDecay?: number;
}

export interface DenseAssociativeFeatureMap {
  hiddenIndex: number;
  map: Float32Array;
}

export interface DenseAssociativeEpochMetrics {
  epoch: number;
  reconstructionError: number;
  contrastiveGap: number;
  hiddenActivation: number;
  winnerShare: number;
  weightMeanAbs: number;
  energy: number;
}

export interface DenseAssociativeSnapshot {
  visible: Float32Array;
  reconstruction: Float32Array;
  hiddenActivations: Float32Array;
  hiddenScores: Float32Array;
  energy: number;
  reconstructionError: number;
  step: number;
  converged: boolean;
  matchedPatternIndex: number;
  topHiddenIndex: number;
  topHiddenActivation: number;
  hiddenEntropy: number;
}

const EPSILON = 1e-6;

export function buildDenseAssociativeFeatureStrength(model: DenseAssociativeMemoryModel): Float32Array {
  const featureStrength = new Float32Array(model.hiddenUnits);
  for (let hiddenIndex = 0; hiddenIndex < model.hiddenUnits; hiddenIndex += 1) {
    const rowOffset = hiddenIndex * model.visibleUnits;
    let total = 0;
    for (let visibleIndex = 0; visibleIndex < model.visibleUnits; visibleIndex += 1) {
      total += Math.abs(model.weights[rowOffset + visibleIndex]);
    }
    featureStrength[hiddenIndex] = total / model.visibleUnits;
  }
  return featureStrength;
}

export function buildDenseAssociativeFeatureMaps(model: DenseAssociativeMemoryModel, featureStrength: Float32Array): DenseAssociativeFeatureMap[] {
  return Array.from({ length: model.hiddenUnits }, (_, hiddenIndex) => ({ hiddenIndex }))
    .sort((left, right) => featureStrength[right.hiddenIndex] - featureStrength[left.hiddenIndex])
    .map(({ hiddenIndex }) => {
      const rowOffset = hiddenIndex * model.visibleUnits;
      const map = new Float32Array(model.visibleUnits);
      let maxAbs = EPSILON;
      for (let visibleIndex = 0; visibleIndex < model.visibleUnits; visibleIndex += 1) {
        maxAbs = Math.max(maxAbs, Math.abs(model.weights[rowOffset + visibleIndex]));
      }
      for (let visibleIndex = 0; visibleIndex < model.visibleUnits; visibleIndex += 1) {
        map[visibleIndex] = model.weights[rowOffset + visibleIndex] / (2 * maxAbs) + 0.5;
      }
      return { hiddenIndex, map };
    });
}

export function buildDenseAssociativeHiddenGrid(hiddenActivations: Float32Array): Float32Array {
  const side = Math.ceil(Math.sqrt(hiddenActivations.length));
  const padded = new Float32Array(side * side);
  padded.set(hiddenActivations);
  return padded;
}

export function getDenseAssociativeHiddenGridSide(hiddenUnits: number): number {
  return Math.ceil(Math.sqrt(hiddenUnits));
}

export function applyDenseAssociativeNoise(pattern: Float32Array, corruptionPercent: number, obfuscationPercent: number): Float32Array {
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
    next[target] = 1 - next[target];
  }

  for (let index = 0; index < obfuscationCount; index += 1) {
    next[indices[corruptionCount + index]] = 0;
  }

  return next;
}

export function createBlankDenseAssociativePattern(): Float32Array {
  return new Float32Array(PATTERN_SIZE);
}

export { PATTERN_SIDE };
