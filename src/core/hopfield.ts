import { PATTERN_SIZE } from "./patternSets";

export type UpdateRule = "hebbian";

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

  train(patterns: Int8Array[], updateRule: UpdateRule): HopfieldInitResult {
    this.patterns = patterns.map((pattern) => pattern.slice());
    this.weights.fill(0);

    if (updateRule === "hebbian") {
      this.trainHebbian(patterns);
    }

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

  step(): HopfieldSnapshot {
    this.previousState = this.state.slice();
    this.shuffleIndices();

    let changedCount = 0;

    for (let order = 0; order < this.scratchIndices.length; order += 1) {
      const neuronIndex = this.scratchIndices[order];
      let weightedSum = 0;
      const rowOffset = neuronIndex * this.size;

      for (let targetIndex = 0; targetIndex < this.size; targetIndex += 1) {
        weightedSum += this.weights[rowOffset + targetIndex] * this.state[targetIndex];
      }

      const nextValue = weightedSum >= 0 ? 1 : -1;
      if (nextValue !== this.state[neuronIndex]) {
        this.state[neuronIndex] = nextValue;
        changedCount += 1;
      }
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
}
