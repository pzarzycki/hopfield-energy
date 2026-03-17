type RbmWasmModule = typeof import("../../wasm-core/pkg-node/hopfield_energy_wasm.js");

const wasm = (await import("../../wasm-core/pkg-node/hopfield_energy_wasm.js")) as RbmWasmModule;

const samples = [
  1, 0, 1, 0,
  0, 1, 0, 1,
  1, 1, 0, 0,
  0, 0, 1, 1,
];

const core = new wasm.RbmCore(
  4,
  3,
  wasm.RbmVisibleModelKind.Bernoulli,
  samples,
  4,
  samples,
  4,
);

core.set_query([1, 0, 1, 0]);
const before = {
  freeEnergy: core.free_energy(),
  reconstructionError: core.reconstruction_error(),
  matchedPatternIndex: core.matched_pattern_index(),
};

const metrics = core.train_epoch(0.05, 1, 2, 0.72, 0.00015);
core.step(1e-4);

const after = {
  freeEnergy: core.free_energy(),
  reconstructionError: core.reconstruction_error(),
  matchedPatternIndex: core.matched_pattern_index(),
  step: core.step_index(),
  epoch: core.epoch(),
  metrics,
};

console.log(
  JSON.stringify(
    {
      backend: "wasm-core",
      model: "rbm",
      primaryTasks: ["generation", "pattern reconstruction", "unsupervised feature learning"],
      before,
      after,
    },
    null,
    2,
  ),
);
