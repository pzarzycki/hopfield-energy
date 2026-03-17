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

function clamp01(value: number): number {
  if (value <= 0) {
    return 0;
  }
  if (value >= 1) {
    return 1;
  }
  return value;
}

function clampSigned(value: number): number {
  if (value <= -1) {
    return -1;
  }
  if (value >= 1) {
    return 1;
  }
  return value;
}

function toCentered(value: number): number {
  return value * 2 - 1;
}

function meanAbsoluteDifference(a: Float32Array, b: Float32Array): number {
  let total = 0;
  for (let index = 0; index < a.length; index += 1) {
    total += Math.abs(a[index] - b[index]);
  }
  return total / Math.max(a.length, 1);
}

function centeredDotSample(sample: Float32Array, weights: Float32Array, rowOffset: number, visibleUnits: number): number {
  let total = 0;
  for (let visibleIndex = 0; visibleIndex < visibleUnits; visibleIndex += 1) {
    total += toCentered(sample[visibleIndex]) * weights[rowOffset + visibleIndex];
  }
  return total / visibleUnits;
}

function centeredDot(a: Float32Array, b: Float32Array): number {
  let total = 0;
  for (let index = 0; index < a.length; index += 1) {
    total += toCentered(a[index]) * toCentered(b[index]);
  }
  return total / Math.max(a.length, 1);
}

function softmax(scores: Float32Array, beta: number): Float32Array {
  let maxScore = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < scores.length; index += 1) {
    maxScore = Math.max(maxScore, beta * scores[index]);
  }

  const output = new Float32Array(scores.length);
  let total = 0;
  for (let index = 0; index < scores.length; index += 1) {
    const value = Math.exp(beta * scores[index] - maxScore);
    output[index] = value;
    total += value;
  }

  const scale = total > EPSILON ? 1 / total : 0;
  for (let index = 0; index < output.length; index += 1) {
    output[index] *= scale;
  }
  return output;
}

function computePotential(scores: Float32Array, activation: DenseAssociativeActivation, sharpness: number): number {
  if (activation === "softmax") {
    let maxScore = Number.NEGATIVE_INFINITY;
    for (let index = 0; index < scores.length; index += 1) {
      maxScore = Math.max(maxScore, sharpness * scores[index]);
    }

    let total = 0;
    for (let index = 0; index < scores.length; index += 1) {
      total += Math.exp(sharpness * scores[index] - maxScore);
    }
    return (maxScore + Math.log(total || 1)) / Math.max(sharpness, EPSILON);
  }

  const exponent = Math.max(1, Math.round(sharpness));
  let total = 0;
  for (let index = 0; index < scores.length; index += 1) {
    const score = scores[index];
    if (activation === "relu-power") {
      total += Math.pow(Math.max(0, score), exponent + 1) / (exponent + 1);
      continue;
    }
    total += Math.pow(Math.abs(score), exponent + 1) / (exponent + 1);
  }
  return total;
}

function computeHiddenScores(visible: Float32Array, model: DenseAssociativeMemoryModel): Float32Array {
  const scores = new Float32Array(model.hiddenUnits);
  for (let hiddenIndex = 0; hiddenIndex < model.hiddenUnits; hiddenIndex += 1) {
    const rowOffset = hiddenIndex * model.visibleUnits;
    scores[hiddenIndex] = centeredDotSample(visible, model.weights, rowOffset, model.visibleUnits);
  }
  return scores;
}

function computeHiddenActivations(scores: Float32Array, activation: DenseAssociativeActivation, sharpness: number): Float32Array {
  if (activation === "softmax") {
    return softmax(scores, Math.max(1, sharpness));
  }

  const exponent = Math.max(1, Math.round(sharpness));
  const hidden = new Float32Array(scores.length);
  for (let index = 0; index < scores.length; index += 1) {
    const score = scores[index];
    if (activation === "relu-power") {
      hidden[index] = Math.pow(Math.max(0, score), exponent);
      continue;
    }
    hidden[index] = Math.sign(score) * Math.pow(Math.abs(score), exponent);
  }
  return hidden;
}

function reconstructVisibleFromHidden(hidden: Float32Array, model: DenseAssociativeMemoryModel): Float32Array {
  const centered = new Float32Array(model.visibleUnits);
  let activationMass = 0;
  for (let hiddenIndex = 0; hiddenIndex < model.hiddenUnits; hiddenIndex += 1) {
    activationMass += Math.abs(hidden[hiddenIndex]);
    const rowOffset = hiddenIndex * model.visibleUnits;
    for (let visibleIndex = 0; visibleIndex < model.visibleUnits; visibleIndex += 1) {
      centered[visibleIndex] += model.weights[rowOffset + visibleIndex] * hidden[hiddenIndex];
    }
  }

  const scale = activationMass > EPSILON ? 1 / activationMass : 0;
  const visible = new Float32Array(model.visibleUnits);
  for (let visibleIndex = 0; visibleIndex < model.visibleUnits; visibleIndex += 1) {
    visible[visibleIndex] = clamp01(clampSigned(centered[visibleIndex] * scale) * 0.5 + 0.5);
  }
  return visible;
}

export function computeAssociativeEnergy(visible: Float32Array, model: DenseAssociativeMemoryModel): number {
  const centeredNorm = centeredDot(visible, visible) * 0.5;
  const scores = computeHiddenScores(visible, model);
  return centeredNorm - computePotential(scores, model.activation, model.sharpness);
}

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

export function summarizeDenseAssociativeState(visible: Float32Array, model: DenseAssociativeMemoryModel, referencePatterns: Float32Array[], step: number): DenseAssociativeSnapshot {
  const hiddenScores = computeHiddenScores(visible, model);
  const hiddenActivations = computeHiddenActivations(hiddenScores, model.activation, model.sharpness);
  const reconstruction = reconstructVisibleFromHidden(hiddenActivations, model);

  let topHiddenIndex = -1;
  let topHiddenActivation = Number.NEGATIVE_INFINITY;
  let activationMass = 0;
  let entropy = 0;
  for (let hiddenIndex = 0; hiddenIndex < hiddenActivations.length; hiddenIndex += 1) {
    const magnitude = Math.abs(hiddenActivations[hiddenIndex]);
    activationMass += magnitude;
    if (magnitude > topHiddenActivation) {
      topHiddenActivation = magnitude;
      topHiddenIndex = hiddenIndex;
    }
  }

  const normalization = activationMass > EPSILON ? activationMass : 1;
  for (let hiddenIndex = 0; hiddenIndex < hiddenActivations.length; hiddenIndex += 1) {
    const probability = Math.abs(hiddenActivations[hiddenIndex]) / normalization;
    if (probability > 0) {
      entropy -= probability * Math.log(probability);
    }
  }

  let matchedPatternIndex = -1;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let index = 0; index < referencePatterns.length; index += 1) {
    const distance = meanAbsoluteDifference(reconstruction, referencePatterns[index]);
    if (distance < bestDistance) {
      bestDistance = distance;
      matchedPatternIndex = index;
    }
  }

  return {
    visible: visible.slice(),
    reconstruction,
    hiddenActivations,
    hiddenScores,
    energy: computeAssociativeEnergy(visible, model),
    reconstructionError: meanAbsoluteDifference(visible, reconstruction),
    step,
    converged: false,
    matchedPatternIndex,
    topHiddenIndex,
    topHiddenActivation: topHiddenActivation > Number.NEGATIVE_INFINITY ? topHiddenActivation : 0,
    hiddenEntropy: entropy,
  };
}

export function createBlankDenseAssociativePattern(): Float32Array {
  return new Float32Array(PATTERN_SIZE);
}

export { PATTERN_SIDE };
