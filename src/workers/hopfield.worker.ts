/// <reference lib="webworker" />

import { createBlankPattern, loadPatternSetById } from "../core/patternSets";
import type { ConvergenceRuleConfig, LearningRuleConfig } from "../core/hopfieldRules";
import type { WorkerRequest, WorkerResponse } from "../core/workerProtocol";

let initialState = createBlankPattern();
let playTimer: number | null = null;
let wasmModulePromise: Promise<typeof import("../../wasm-core/pkg/hopfield_energy_wasm.js")> | null = null;
let core: InstanceType<(typeof import("../../wasm-core/pkg/hopfield_energy_wasm.js"))["HopfieldCore"]> | null = null;
let currentPatternCount = 0;
const wasmBinaryUrl = new URL("../../wasm-core/pkg/hopfield_energy_wasm_bg.wasm", import.meta.url);

function post(message: WorkerResponse, transfer: Transferable[] = []): void {
  self.postMessage(message, transfer);
}

function pausePlayback(): void {
  if (playTimer !== null) {
    self.clearInterval(playTimer);
    playTimer = null;
  }
}

async function loadWasmModule() {
  if (!wasmModulePromise) {
    wasmModulePromise = import("../../wasm-core/pkg/hopfield_energy_wasm.js").then(async (module) => {
      await module.default(wasmBinaryUrl);
      return module;
    });
  }
  return wasmModulePromise;
}

function flattenPatterns(patterns: Int8Array[]): Int8Array {
  return Int8Array.from(patterns.flatMap((pattern) => Array.from(pattern)));
}

function parseLearningRule(
  module: typeof import("../../wasm-core/pkg/hopfield_energy_wasm.js"),
  config: LearningRuleConfig,
) {
  switch (config.rule) {
    case "pseudoinverse":
      return module.HopfieldLearningRule.Pseudoinverse;
    case "storkey":
      return module.HopfieldLearningRule.Storkey;
    case "krauth-mezard":
      return module.HopfieldLearningRule.KrauthMezard;
    case "unlearning":
      return module.HopfieldLearningRule.Unlearning;
    case "hebbian":
    default:
      return module.HopfieldLearningRule.Hebbian;
  }
}

function parseConvergenceRule(
  module: typeof import("../../wasm-core/pkg/hopfield_energy_wasm.js"),
  config: ConvergenceRuleConfig,
) {
  switch (config.rule) {
    case "synchronous":
      return module.HopfieldConvergenceRule.Synchronous;
    case "stochastic":
      return module.HopfieldConvergenceRule.Stochastic;
    case "async-random":
    default:
      return module.HopfieldConvergenceRule.AsyncRandom;
  }
}

function ensureCore(): NonNullable<typeof core> {
  if (!core) {
    throw new Error("Hopfield worker received a command before initialization completed.");
  }
  return core;
}

function createSnapshot() {
  const activeCore = ensureCore();
  return {
    state: Int8Array.from(activeCore.state()),
    energy: activeCore.energy(),
    step: activeCore.step_index(),
    changedCount: activeCore.changed_count(),
    matchedPatternIndex: activeCore.matched_pattern_index(),
    converged: activeCore.converged(),
  };
}

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const message = event.data;

  void (async () => {
    try {
      if (message.type === "initialize") {
        pausePlayback();

        const patternSet = await loadPatternSetById(message.patternSetId);
        const module = await loadWasmModule();
        core?.free();
        core = new module.HopfieldCore(patternSet.patterns[0]?.length ?? initialState.length, flattenPatterns(patternSet.patterns), patternSet.patterns.length);
        currentPatternCount = patternSet.patterns.length;
        core.train(
          parseLearningRule(module, message.learningConfig),
          message.learningConfig.rule === "krauth-mezard" ? message.learningConfig.kappa : 0,
          "epsilon" in message.learningConfig ? message.learningConfig.epsilon : 0,
          "maxEpochs" in message.learningConfig ? message.learningConfig.maxEpochs : 1,
          "steps" in message.learningConfig ? message.learningConfig.steps : 1,
        );
        initialState = createBlankPattern();
        core.set_state(Int8Array.from(initialState));
        const snapshot = createSnapshot();

        post(
          {
            type: "ready",
            backend: "wasm-core",
            weights: core.weights(),
            maxWeightAbs: core.max_weight_abs(),
            patternCount: currentPatternCount,
            snapshot,
          },
          [snapshot.state.buffer],
        );
        return;
      }

      if (message.type === "setQuery") {
        pausePlayback();
        initialState = message.pattern.slice();
        ensureCore().set_state(Int8Array.from(initialState));
        const snapshot = createSnapshot();
        post({ type: "snapshot", snapshot }, [snapshot.state.buffer]);
        return;
      }

      if (message.type === "reset") {
        pausePlayback();
        ensureCore().set_state(Int8Array.from(initialState));
        const snapshot = createSnapshot();
        post({ type: "snapshot", snapshot }, [snapshot.state.buffer]);
        return;
      }

      if (message.type === "step") {
        pausePlayback();
        const module = await loadWasmModule();
        ensureCore().step(
          parseConvergenceRule(module, message.convergenceConfig),
          message.convergenceConfig.rule === "stochastic" ? message.convergenceConfig.temperature : 0.1,
        );
        const snapshot = createSnapshot();
        post({ type: "snapshot", snapshot }, [snapshot.state.buffer]);
        return;
      }

      if (message.type === "play") {
        pausePlayback();

        let iteration = 0;
        playTimer = self.setInterval(() => {
          void (async () => {
            const module = await loadWasmModule();
            ensureCore().step(
              parseConvergenceRule(module, message.convergenceConfig),
              message.convergenceConfig.rule === "stochastic" ? message.convergenceConfig.temperature : 0.1,
            );
            const snapshot = createSnapshot();
            post({ type: "snapshot", snapshot }, [snapshot.state.buffer]);

            iteration += 1;
            if (snapshot.converged || iteration >= message.maxSteps) {
              pausePlayback();
              post({ type: "paused" });
            }
          })().catch((error: unknown) => {
            pausePlayback();
            post({
              type: "error",
              message: error instanceof Error ? error.message : "Unknown worker error",
            });
          });
        }, message.intervalMs);
        return;
      }

      if (message.type === "pause") {
        pausePlayback();
        post({ type: "paused" });
      }
    } catch (error) {
      pausePlayback();
      post({
        type: "error",
        message: error instanceof Error ? error.message : "Unknown worker error",
      });
    }
  })();
};
