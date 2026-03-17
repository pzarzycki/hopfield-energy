type DenseHopfieldWasmModule = typeof import("../../wasm-core/pkg-node/hopfield_energy_wasm.js");

const wasm = (await import("../../wasm-core/pkg-node/hopfield_energy_wasm.js")) as DenseHopfieldWasmModule;

const memories = [
  1, 0, 1, 0,
  0, 1, 0, 1,
];

const core = new wasm.DenseHopfieldCore(memories, 2, 4, 8);
core.set_state([1, 0, 0.8, 0]);

const before = {
  energy: core.energy(),
  entropy: core.entropy(),
  topAttention: core.top_attention(),
  matchedPatternIndex: core.matched_pattern_index(),
};

core.step(1e-4);

const after = {
  energy: core.energy(),
  entropy: core.entropy(),
  topAttention: core.top_attention(),
  matchedPatternIndex: core.matched_pattern_index(),
  delta: core.delta(),
  converged: core.converged(),
  step: core.step_index(),
};

console.log(
  JSON.stringify(
    {
      backend: "wasm-core",
      model: "dense-hopfield",
      primaryTasks: ["associative retrieval", "pattern reconstruction"],
      before,
      after,
    },
    null,
    2,
  ),
);
