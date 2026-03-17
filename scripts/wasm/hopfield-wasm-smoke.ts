type HopfieldWasmModule = typeof import("../../wasm-core/pkg-node/hopfield_energy_wasm.js");

const wasm = (await import("../../wasm-core/pkg-node/hopfield_energy_wasm.js")) as HopfieldWasmModule;

const patterns = [
  1, -1, 1, -1,
  -1, 1, -1, 1,
];

const core = new wasm.HopfieldCore(4, patterns, 2);
core.train(wasm.HopfieldLearningRule.Hebbian, 0, 0, 1, 1);
core.set_state([1, -1, 1, -1]);

const before = {
  energy: core.energy(),
  matchedPatternIndex: core.matched_pattern_index(),
  converged: core.converged(),
};

core.step(wasm.HopfieldConvergenceRule.Synchronous, 1);

const after = {
  energy: core.energy(),
  matchedPatternIndex: core.matched_pattern_index(),
  converged: core.converged(),
  changedCount: core.changed_count(),
  step: core.step_index(),
  maxWeightAbs: core.max_weight_abs(),
};

console.log(
  JSON.stringify(
    {
      backend: "wasm-core",
      model: "hopfield",
      primaryTasks: ["associative retrieval", "pattern reconstruction"],
      before,
      after,
    },
    null,
    2,
  ),
);
