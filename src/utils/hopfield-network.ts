// Core Hopfield Network implementation with step-by-step tracking

export interface NetworkState {
    state: number[];
    energy: number;
    step: number;
    updatedNeuron?: number;
}

export class HopfieldNetwork {
    private size: number;
    private weights: number[][];
    private memories: number[][];
    public currentState: number[];
    public history: NetworkState[];
    private stepCount: number;

    constructor(size: number) {
        this.size = size;
        this.weights = Array(size).fill(0).map(() => Array(size).fill(0));
        this.memories = [];
        this.currentState = Array(size).fill(-1);
        this.history = [];
        this.stepCount = 0;
    }

    // Train the network using Hebbian learning
    train(patterns: number[][]): void {
        if (patterns.length === 0) return;

        this.memories = patterns.map(p => [...p]);
        const numPatterns = patterns.length;

        // Initialize weights to zero
        for (let i = 0; i < this.size; i++) {
            for (let j = 0; j < this.size; j++) {
                this.weights[i][j] = 0;
            }
        }

        // Calculate mean activity for each neuron across all patterns
        const means = new Array(this.size).fill(0);
        for (let i = 0; i < this.size; i++) {
            for (const pattern of patterns) {
                means[i] += pattern[i];
            }
            means[i] /= numPatterns;
        }

        // Center the patterns: ξ~ = ξ - m
        const centeredPatterns = patterns.map(pattern => 
            pattern.map((val, i) => val - means[i])
        );

        // Zero-mean Hebbian learning: w_ij = (1/N) * Σ(ξ~_i^p * ξ~_j^p)
        for (const centered of centeredPatterns) {
            for (let i = 0; i < this.size; i++) {
                for (let j = 0; j < this.size; j++) {
                    if (i !== j) { // No self-connections
                        this.weights[i][j] += (centered[i] * centered[j]) / numPatterns;
                    }
                }
            }
        }
    }

    // Initialize the network with a query pattern
    setState(pattern: number[]): void {
        this.currentState = [...pattern];
        this.stepCount = 0;
        this.history = [];

        // Record initial state
        this.history.push({
            state: [...this.currentState],
            energy: this.computeEnergy(this.currentState),
            step: this.stepCount
        });
    }

    // Compute energy of a given state
    computeEnergy(state: number[]): number {
        let energy = 0;

        // Avoid double counting by only summing upper triangle
        for (let i = 0; i < this.size; i++) {
            for (let j = i + 1; j < this.size; j++) {
                energy -= this.weights[i][j] * state[i] * state[j];
            }
        }

        return energy;
    }

    // Update a single neuron (asynchronous update)
    updateNeuron(index: number): boolean {
        let sum = 0;

        for (let j = 0; j < this.size; j++) {
            sum += this.weights[index][j] * this.currentState[j];
        }

        const newValue = sum >= 0 ? 1 : -1;
        const changed = newValue !== this.currentState[index];

        if (changed) {
            this.currentState[index] = newValue;
        }

        return changed;
    }

    // Perform one update step (update ALL neurons in random order)
    step(): NetworkState {
        // Shuffle neuron indices for random asynchronous update order
        const indices = Array.from({ length: this.size }, (_, i) => i);
        this.shuffleArray(indices);

        for (const idx of indices) {
            this.updateNeuron(idx);
        }

        this.stepCount++;

        const currentEnergy = this.computeEnergy(this.currentState);
        const state: NetworkState = {
            state: [...this.currentState],
            energy: currentEnergy,
            step: this.stepCount
        };

        this.history.push(state);
        return state;
    }

    // Check if network has converged (stable for last 2 steps)
    hasConverged(): boolean {
        if (this.history.length < 2) return false;

        const len = this.history.length;
        return this.arraysEqual(
            this.history[len - 1].state,
            this.history[len - 2].state
        );
    }

    // Find which memory pattern the current state is closest to
    getMatchedPattern(): number {
        let minDistance = Infinity;
        let matchedIndex = -1;

        for (let i = 0; i < this.memories.length; i++) {
            const distance = this.hammingDistance(this.currentState, this.memories[i]);
            if (distance < minDistance) {
                minDistance = distance;
                matchedIndex = i;
            }
        }

        return matchedIndex;
    }

    // Calculate Hamming distance between two patterns
    private hammingDistance(pattern1: number[], pattern2: number[]): number {
        let distance = 0;
        for (let i = 0; i < pattern1.length; i++) {
            if (pattern1[i] !== pattern2[i]) {
                distance++;
            }
        }
        return distance;
    }

    // Check if two arrays are equal
    private arraysEqual(arr1: number[], arr2: number[]): boolean {
        if (arr1.length !== arr2.length) return false;
        for (let i = 0; i < arr1.length; i++) {
            if (arr1[i] !== arr2[i]) return false;
        }
        return true;
    }

    // Shuffle array in place (Fisher-Yates)
    private shuffleArray<T>(array: T[]): void {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
    }

    // Reset to a specific history step
    resetToStep(stepIndex: number): void {
        if (stepIndex >= 0 && stepIndex < this.history.length) {
            const targetState = this.history[stepIndex];
            this.currentState = [...targetState.state];
            this.stepCount = targetState.step;
            this.history = this.history.slice(0, stepIndex + 1);
        }
    }

    // Get current energy
    getCurrentEnergy(): number {
        return this.computeEnergy(this.currentState);
    }
}
