import assert from "node:assert/strict";

type HopfieldWasmModule = typeof import("../../wasm-core/pkg-node/hopfield_energy_wasm.js");

const wasm = (await import("../../wasm-core/pkg-node/hopfield_energy_wasm.js")) as HopfieldWasmModule;

function approxEqual(actual: number, expected: number, tolerance = 1e-6, message?: string) {
  assert.ok(Math.abs(actual - expected) <= tolerance, message ?? `expected ${actual} ~= ${expected} (tol ${tolerance})`);
}

function sum(values: ArrayLike<number>): number {
  let total = 0;
  for (let index = 0; index < values.length; index += 1) {
    total += values[index] ?? 0;
  }
  return total;
}

function topShare(values: ArrayLike<number>): number {
  let total = 0;
  let best = 0;
  for (let index = 0; index < values.length; index += 1) {
    const magnitude = Math.abs(values[index] ?? 0);
    total += magnitude;
    best = Math.max(best, magnitude);
  }
  return total > 0 ? best / total : 0;
}

function runHopfieldChecks() {
  const patterns = Int8Array.from([
    1, -1, 1, -1,
    -1, 1, -1, 1,
  ]);
  const core = new wasm.HopfieldCore(4, patterns, 2);
  core.train(wasm.HopfieldLearningRule.Hebbian, 0, 0, 1, 1);

  const weights = core.weights();
  assert.equal(weights.length, 16);
  for (let row = 0; row < 4; row += 1) {
    approxEqual(weights[row * 4 + row] ?? 0, 0, 1e-6, "Hopfield diagonal should be zero");
    for (let col = 0; col < 4; col += 1) {
      approxEqual(weights[row * 4 + col] ?? 0, weights[col * 4 + row] ?? 0, 1e-6, "Hopfield weights must be symmetric");
    }
  }

  core.set_state(Int8Array.from([1, -1, -1, -1]));
  const beforeEnergy = core.energy();
  core.step(wasm.HopfieldConvergenceRule.AsyncRandom, 1);
  const afterEnergy = core.energy();
  assert.ok(afterEnergy <= beforeEnergy + 1e-6, `Hopfield async energy increased: ${beforeEnergy} -> ${afterEnergy}`);
}

function runDenseHopfieldChecks() {
  const memories = Float32Array.from([
    1, 0, 1, 0,
    0, 1, 0, 1,
  ]);

  const softCore = new wasm.DenseHopfieldCore(memories, 2, 4, 2);
  softCore.set_state(Float32Array.from([1, 0, 0.8, 0]));

  const sharpCore = new wasm.DenseHopfieldCore(memories, 2, 4, 16);
  sharpCore.set_state(Float32Array.from([1, 0, 0.8, 0]));

  approxEqual(sum(softCore.attention()), 1, 1e-6, "Dense Hopfield attention must sum to 1");
  assert.ok(
    sharpCore.top_attention() >= softCore.top_attention(),
    `Expected higher beta to sharpen attention: ${sharpCore.top_attention()} < ${softCore.top_attention()}`,
  );
}

function runRbmChecks() {
  const bernoulliSamples = Float32Array.from([
    1, 0, 1, 0,
    0, 1, 0, 1,
    1, 1, 0, 0,
    0, 0, 1, 1,
  ]);
  const bernoulliCore = new wasm.RbmCore(4, 3, wasm.RbmVisibleModelKind.Bernoulli, bernoulliSamples, 4, bernoulliSamples, 4);
  bernoulliCore.set_query(Float32Array.from([1, 0, 1, 0]));

  const beforeEpoch = bernoulliCore.epoch();
  bernoulliCore.train_epoch(0.05, 1, 2, 0.72, 0.00015);
  assert.equal(bernoulliCore.epoch(), beforeEpoch + 1, "RBM epoch must advance after training");
  assert.equal(bernoulliCore.weights().length, 12, "RBM weights must match hidden x visible");
  assert.equal(bernoulliCore.visible_bias().length, 4, "RBM visible bias length mismatch");
  assert.equal(bernoulliCore.hidden_bias().length, 3, "RBM hidden bias length mismatch");

  const gaussianSamples = Float32Array.from([
    1.0, 0.2, 0.8, 0.0,
    0.0, 0.9, 0.1, 1.0,
    0.7, 0.8, 0.2, 0.1,
    0.1, 0.0, 0.8, 0.9,
  ]);
  const gaussianCore = new wasm.RbmCore(4, 3, wasm.RbmVisibleModelKind.Gaussian, gaussianSamples, 4, gaussianSamples, 4);
  gaussianCore.set_query(Float32Array.from([1.0, 0.2, 0.8, 0.0]));
  gaussianCore.train_epoch(0.03, 1, 2, 0.72, 0.00015);
  gaussianCore.step(1e-4);

  assert.ok(Number.isFinite(gaussianCore.free_energy()), "Gaussian RBM free energy must be finite");
  assert.ok(Number.isFinite(gaussianCore.reconstruction_error()), "Gaussian RBM reconstruction error must be finite");
  assert.ok(gaussianCore.reconstruction().every((value) => Number.isFinite(value) && value >= 0 && value <= 1), "Gaussian reconstruction must stay in [0,1]");
}

function runDamChecks() {
  const samples = Float32Array.from([
    1, 0, 1, 0,
    0, 1, 0, 1,
    1, 1, 0, 0,
    0, 0, 1, 1,
  ]);

  const softCore = new wasm.DenseAssociativeMemoryCore(4, 3, wasm.DenseAssociativeActivationKind.ReluPower, 2, samples, 4, samples, 4);
  softCore.set_query(Float32Array.from([1, 0, 0.8, 0]));

  const sharpCore = new wasm.DenseAssociativeMemoryCore(4, 3, wasm.DenseAssociativeActivationKind.ReluPower, 12, samples, 4, samples, 4);
  sharpCore.set_query(Float32Array.from([1, 0, 0.8, 0]));

  assert.ok(
    topShare(sharpCore.hidden_activations()) >= topShare(softCore.hidden_activations()),
    "Higher DAM sharpness should not reduce winner concentration",
  );

  const softmaxCore = new wasm.DenseAssociativeMemoryCore(4, 3, wasm.DenseAssociativeActivationKind.Softmax, 6, samples, 4, samples, 4);
  softmaxCore.set_query(Float32Array.from([1, 0, 1, 0]));
  const activations = softmaxCore.hidden_activations();
  approxEqual(sum(activations), 1, 1e-5, "DAM softmax activations must sum to 1");
  assert.ok(activations.every((value) => Number.isFinite(value) && value >= 0), "DAM softmax activations must be finite and non-negative");
}

runHopfieldChecks();
runDenseHopfieldChecks();
runRbmChecks();
runDamChecks();

console.log(
  JSON.stringify(
    {
      suite: "wasm-regression",
      status: "passed",
      models: ["hopfield", "dense-hopfield", "rbm", "dense-associative-memory"],
      assertions: [
        "Hopfield symmetry and async energy descent",
        "Dense Hopfield normalized attention and beta sharpening",
        "RBM tensor shapes, epoch advance, Gaussian bounded reconstruction",
        "DAM sharpness concentration and softmax normalization",
      ],
    },
    null,
    2,
  ),
);
