import { readFileSync } from 'fs';
import numeric from 'numeric';

function grayscaleToBipolar(pattern: number[], threshold = 128): number[] {
    return pattern.map(v => v >= threshold ? 1 : -1);
}

const tsContent = readFileSync('src/utils/mnist-data.ts', 'utf-8');
const b64Match = tsContent.match(/const B64_DIGITS: string\[\] = \[\n([\s\S]*?)\];/);
const b64Lines = b64Match![1].match(/"([^"]+)"/g)!;

function decodeB64(b64str: string): number[] {
    const buf = Buffer.from(b64str.replace(/"/g, ''), 'base64');
    return Array.from(buf);
}
const digits = b64Lines.map(s => grayscaleToBipolar(decodeB64(s)));

class HopfieldNetwork {
    size: number;
    weights: number[][];
    memories: number[][];
    currentState: number[];

    constructor(size: number) {
        this.size = size;
        this.weights = Array(size).fill(0).map(() => Array(size).fill(0));
        this.memories = [];
        this.currentState = Array(size).fill(-1);
    }

    trainProjection(patterns: number[][]): void {
        this.memories = patterns.map(p => [...p]);
        const numPatterns = patterns.length;
        const N = this.size;

        // X is N x P matrix
        // X^T is P x N
        // Compute overlap matrix Q = X^T * X
        const Q: number[][] = Array(numPatterns).fill(0).map(() => Array(numPatterns).fill(0));
        for (let mu = 0; mu < numPatterns; mu++) {
            for (let nu = 0; nu < numPatterns; nu++) {
                let dot = 0;
                for (let i = 0; i < N; i++) {
                    dot += patterns[mu][i] * patterns[nu][i];
                }
                Q[mu][nu] = dot;
            }
        }

        // Invert Q using numeric.js
        const Qinv = numeric.inv(Q);

        // Compute weights W = X * Qinv * X^T
        for (let i = 0; i < N; i++) {
            for (let j = i; j < N; j++) {
                if (i === j) {
                    this.weights[i][j] = 0;
                    continue;
                }
                
                let sum = 0;
                for (let mu = 0; mu < numPatterns; mu++) {
                    for (let nu = 0; nu < numPatterns; nu++) {
                        sum += patterns[mu][i] * Qinv[mu][nu] * patterns[nu][j];
                    }
                }
                this.weights[i][j] = sum;
                this.weights[j][i] = sum;
            }
        }
    }

    setState(pattern: number[]): void {
        this.currentState = [...pattern];
    }

    updateNeuron(index: number): boolean {
        let sum = 0;
        for (let j = 0; j < this.size; j++) {
            sum += this.weights[index][j] * this.currentState[j];
        }
        
        let newValue = this.currentState[index];
        if (sum > 0) newValue = 1;
        else if (sum < 0) newValue = -1;

        const changed = newValue !== this.currentState[index];
        if (changed) this.currentState[index] = newValue;
        return changed;
    }

    step(): boolean {
        const indices = Array.from({ length: this.size }, (_, i) => i);
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }
        let anyChanged = false;
        for (const idx of indices) {
            if (this.updateNeuron(idx)) anyChanged = true;
        }
        return anyChanged;
    }

    hammingDistance(a: number[], b: number[]): number {
        let d = 0;
        for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) d++;
        return d;
    }
}

const net = new HopfieldNetwork(784);
net.trainProjection(digits);

for (let i = 0; i < 10; i++) {
    net.setState([...digits[i]]);
    net.step();
    const hamming = net.hammingDistance(net.currentState, digits[i]);
    console.log(`Digit ${i} fixed-point error: ${hamming} bits`);
}

// Test noisy recall
for (let i = 0; i < 10; i++) {
    const noisy = digits[i].map(v => Math.random() < 0.2 ? -v : v);
    net.setState(noisy);
    const startHam = net.hammingDistance(net.currentState, digits[i]);
    for (let s = 0; s < 50; s++) if (!net.step()) break;
    const endHam = net.hammingDistance(net.currentState, digits[i]);
    console.log(`Digit ${i} noisy recall (20% noise): start=${startHam}, end=${endHam}`);
}
