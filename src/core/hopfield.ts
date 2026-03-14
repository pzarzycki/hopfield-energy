import numeric from "numeric";

import type { ConvergenceRuleConfig, LearningRuleConfig } from "./hopfieldRules";
import { PATTERN_SIZE } from "./patternSets";

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

export class HopfieldEngine {
  readonly size: number;
  readonly weights: Float32Array;
  readonly state: Int8Array;
  private readonly scratchIndices: Uint16Array;
  private patterns: Int8Array[] = [];
  private previousState: Int8Array;
  private stepCount = 0;

  constructor(size = PATTERN_SIZE) {
    this.size = size;
    this.weights = new Float32Array(size * size);
    this.state = new Int8Array(size).fill(-1);
    this.previousState = this.state.slice();
    this.scratchIndices = new Uint16Array(size);
    for (let index = 0; index < size; index += 1) {
      this.scratchIndices[index] = index;
    }
  }

  train(patterns: Int8Array[], learningConfig: LearningRuleConfig): HopfieldInitResult {
    this.patterns = patterns.map((pattern) => pattern.slice());
    this.weights.fill(0);

    switch (learningConfig.rule) {
      case "hebbian":
        this.trainHebbian(patterns);
        break;
      case "pseudoinverse":
        this.trainPseudoinverse(patterns);
        break;
      case "storkey":
        this.trainStorkey(patterns);
        break;
      case "krauth-mezard":
        this.trainKrauthMezard(patterns, learningConfig.kappa, learningConfig.epsilon, learningConfig.maxEpochs);
        break;
      case "unlearning":
        this.trainUnlearning(patterns, learningConfig.epsilon, learningConfig.steps);
        break;
    }

    this.zeroDiagonal();

    return {
      weights: this.weights.slice(),
      maxWeightAbs: this.getMaxWeightAbs(),
      patternCount: patterns.length,
    };
  }

  setState(nextState: Int8Array): HopfieldSnapshot {
    this.state.set(nextState);
    this.previousState = this.state.slice();
    this.stepCount = 0;
    return this.snapshot(0);
  }

  step(convergenceConfig: ConvergenceRuleConfig): HopfieldSnapshot {
    this.previousState = this.state.slice();

    let changedCount = 0;
    switch (convergenceConfig.rule) {
      case "async-random":
        changedCount = this.stepAsyncRandom(this.state);
        break;
      case "synchronous":
        changedCount = this.stepSynchronous();
        break;
      case "stochastic":
        changedCount = this.stepStochastic(Math.max(0.01, convergenceConfig.temperature));
        break;
    }

    this.stepCount += 1;
    return this.snapshot(changedCount);
  }

  reset(initialState: Int8Array): HopfieldSnapshot {
    return this.setState(initialState);
  }

  private trainHebbian(patterns: Int8Array[]): void {
    const scale = 1 / this.size;

    for (const pattern of patterns) {
      for (let row = 0; row < this.size; row += 1) {
        const rowOffset = row * this.size;
        for (let column = row + 1; column < this.size; column += 1) {
          const weight = pattern[row] * pattern[column] * scale;
          this.weights[rowOffset + column] += weight;
          this.weights[column * this.size + row] += weight;
        }
      }
    }
  }

  private trainPseudoinverse(patterns: Int8Array[]): void {
    const patternCount = patterns.length;
    if (patternCount === 0) {
      return;
    }

    const overlap = Array.from({ length: patternCount }, () => new Array<number>(patternCount).fill(0));
    for (let mu = 0; mu < patternCount; mu += 1) {
      for (let nu = mu; nu < patternCount; nu += 1) {
        let dot = 0;
        for (let index = 0; index < this.size; index += 1) {
          dot += patterns[mu][index] * patterns[nu][index];
        }
        overlap[mu][nu] = dot;
        overlap[nu][mu] = dot;
      }
    }

    const inverse = this.invertWithRegularization(overlap);

    for (let row = 0; row < this.size; row += 1) {
      const rowOffset = row * this.size;
      for (let column = row + 1; column < this.size; column += 1) {
        let value = 0;
        for (let mu = 0; mu < patternCount; mu += 1) {
          for (let nu = 0; nu < patternCount; nu += 1) {
            value += patterns[mu][row] * inverse[mu][nu] * patterns[nu][column];
          }
        }
        this.weights[rowOffset + column] = value;
        this.weights[column * this.size + row] = value;
      }
    }
  }

  private trainStorkey(patterns: Int8Array[]): void {
    const scale = 1 / this.size;
    const localFields = new Float32Array(this.size);

    for (const pattern of patterns) {
      for (let row = 0; row < this.size; row += 1) {
        localFields[row] = this.getLocalField(row, pattern);
      }

      for (let row = 0; row < this.size; row += 1) {
        const rowOffset = row * this.size;
        for (let column = row + 1; column < this.size; column += 1) {
          const delta =
            (pattern[row] * pattern[column] - pattern[row] * localFields[column] - localFields[row] * pattern[column]) *
            scale;
          this.weights[rowOffset + column] += delta;
          this.weights[column * this.size + row] += delta;
        }
      }

      this.zeroDiagonal();
    }
  }

  private trainKrauthMezard(patterns: Int8Array[], kappa: number, epsilon: number, maxEpochs: number): void {
    const scale = epsilon / this.size;

    for (let epoch = 0; epoch < maxEpochs; epoch += 1) {
      let converged = true;

      for (const pattern of patterns) {
        for (let row = 0; row < this.size; row += 1) {
          const stability = pattern[row] * this.getLocalField(row, pattern);
          if (stability > kappa) {
            continue;
          }

          const rowOffset = row * this.size;
          const factor = scale * pattern[row];
          for (let column = 0; column < this.size; column += 1) {
            if (column === row) {
              continue;
            }
            this.weights[rowOffset + column] += factor * pattern[column];
          }
          converged = false;
        }
      }

      this.symmetrizeWeights();
      this.zeroDiagonal();
      if (converged) {
        break;
      }
    }
  }

  private trainUnlearning(patterns: Int8Array[], epsilon: number, steps: number): void {
    this.trainHebbian(patterns);
    const scale = epsilon / this.size;

    for (let iteration = 0; iteration < steps; iteration += 1) {
      const state = this.createRandomState();
      this.runAsyncToConvergence(state, 18);
      if (this.isStoredPattern(state)) {
        continue;
      }

      for (let row = 0; row < this.size; row += 1) {
        const rowOffset = row * this.size;
        for (let column = row + 1; column < this.size; column += 1) {
          const delta = scale * state[row] * state[column];
          this.weights[rowOffset + column] -= delta;
          this.weights[column * this.size + row] -= delta;
        }
      }
      this.zeroDiagonal();
    }
  }

  private computeEnergy(): number {
    let energy = 0;
    for (let row = 0; row < this.size; row += 1) {
      const rowOffset = row * this.size;
      for (let column = row + 1; column < this.size; column += 1) {
        energy -= this.weights[rowOffset + column] * this.state[row] * this.state[column];
      }
    }
    return energy;
  }

  private getMatchedPatternIndex(): number {
    let bestIndex = -1;
    let bestDistance = Number.POSITIVE_INFINITY;

    for (let patternIndex = 0; patternIndex < this.patterns.length; patternIndex += 1) {
      let distance = 0;
      const pattern = this.patterns[patternIndex];
      for (let index = 0; index < this.size; index += 1) {
        if (pattern[index] !== this.state[index]) {
          distance += 1;
        }
      }

      if (distance < bestDistance) {
        bestDistance = distance;
        bestIndex = patternIndex;
      }
    }

    return bestIndex;
  }

  private getMaxWeightAbs(): number {
    let maxAbs = 0;
    for (let index = 0; index < this.weights.length; index += 1) {
      const value = Math.abs(this.weights[index]);
      if (value > maxAbs) {
        maxAbs = value;
      }
    }
    return maxAbs;
  }

  private snapshot(changedCount: number): HopfieldSnapshot {
    return {
      state: this.state.slice(),
      energy: this.computeEnergy(),
      step: this.stepCount,
      changedCount,
      matchedPatternIndex: this.getMatchedPatternIndex(),
      converged: this.isConverged(),
    };
  }

  private isConverged(): boolean {
    for (let index = 0; index < this.size; index += 1) {
      if (this.previousState[index] !== this.state[index]) {
        return false;
      }
    }
    return true;
  }

  private shuffleIndices(): void {
    for (let index = this.scratchIndices.length - 1; index > 0; index -= 1) {
      const swapIndex = Math.floor(Math.random() * (index + 1));
      const current = this.scratchIndices[index];
      this.scratchIndices[index] = this.scratchIndices[swapIndex];
      this.scratchIndices[swapIndex] = current;
    }
  }

  private stepAsyncRandom(targetState: Int8Array): number {
    this.shuffleIndices();

    let changedCount = 0;
    for (let order = 0; order < this.scratchIndices.length; order += 1) {
      const neuronIndex = this.scratchIndices[order];
      const nextValue = this.getLocalField(neuronIndex, targetState) >= 0 ? 1 : -1;
      if (nextValue !== targetState[neuronIndex]) {
        targetState[neuronIndex] = nextValue;
        changedCount += 1;
      }
    }

    return changedCount;
  }

  private stepSynchronous(): number {
    const nextState = new Int8Array(this.size);
    let changedCount = 0;

    for (let neuronIndex = 0; neuronIndex < this.size; neuronIndex += 1) {
      const nextValue = this.getLocalField(neuronIndex, this.state) >= 0 ? 1 : -1;
      nextState[neuronIndex] = nextValue;
      if (nextValue !== this.state[neuronIndex]) {
        changedCount += 1;
      }
    }

    this.state.set(nextState);
    return changedCount;
  }

  private stepStochastic(temperature: number): number {
    let changedCount = 0;

    for (let stepIndex = 0; stepIndex < this.size; stepIndex += 1) {
      const neuronIndex = Math.floor(Math.random() * this.size);
      const field = this.getLocalField(neuronIndex, this.state);
      const probability = 1 / (1 + Math.exp((-2 * field) / temperature));
      const nextValue = Math.random() < probability ? 1 : -1;
      if (nextValue !== this.state[neuronIndex]) {
        this.state[neuronIndex] = nextValue;
        changedCount += 1;
      }
    }

    return changedCount;
  }

  private getLocalField(neuronIndex: number, sourceState: Int8Array): number {
    let weightedSum = 0;
    const rowOffset = neuronIndex * this.size;

    for (let targetIndex = 0; targetIndex < this.size; targetIndex += 1) {
      weightedSum += this.weights[rowOffset + targetIndex] * sourceState[targetIndex];
    }

    return weightedSum;
  }

  private symmetrizeWeights(): void {
    for (let row = 0; row < this.size; row += 1) {
      const rowOffset = row * this.size;
      for (let column = row + 1; column < this.size; column += 1) {
        const value = (this.weights[rowOffset + column] + this.weights[column * this.size + row]) * 0.5;
        this.weights[rowOffset + column] = value;
        this.weights[column * this.size + row] = value;
      }
    }
  }

  private zeroDiagonal(): void {
    for (let row = 0; row < this.size; row += 1) {
      this.weights[row * this.size + row] = 0;
    }
  }

  private invertWithRegularization(matrix: number[][]): number[][] {
    const attempts = [0, 1e-8, 1e-6, 1e-4];

    for (const regularization of attempts) {
      try {
        const candidate = matrix.map((row, rowIndex) =>
          row.map((value, columnIndex) => value + (rowIndex === columnIndex ? regularization : 0)),
        );
        return numeric.inv(candidate);
      } catch {
        continue;
      }
    }

    throw new Error("Unable to invert pattern overlap matrix for the pseudoinverse rule.");
  }

  private createRandomState(): Int8Array {
    const state = new Int8Array(this.size);
    for (let index = 0; index < this.size; index += 1) {
      state[index] = Math.random() < 0.5 ? -1 : 1;
    }
    return state;
  }

  private runAsyncToConvergence(state: Int8Array, maxSweeps: number): void {
    for (let sweep = 0; sweep < maxSweeps; sweep += 1) {
      if (this.stepAsyncRandom(state) === 0) {
        break;
      }
    }
  }

  private isStoredPattern(candidate: Int8Array): boolean {
    return this.patterns.some((pattern) => {
      for (let index = 0; index < this.size; index += 1) {
        if (pattern[index] !== candidate[index]) {
          return false;
        }
      }
      return true;
    });
  }
}
