/// <reference lib="webworker" />

import { grayscaleToFloat32, loadDatasetArchive, selectMemorySamples, type DatasetName } from "../data/datasetArchives";
import {
  buildFeatureMaps,
  buildFeatureStrength,
  createBlankVisiblePattern,
  quantizeVisiblePattern,
  type RBMEpochMetrics,
  type RBMModel,
  type RBMSnapshot,
  type RBMTrainingConfig,
  type RBMVisibleModel,
} from "../core/rbm";
import type {
  RBMWorkerRequest,
  RBMWorkerResponse,
} from "../core/rbmWorkerProtocol";

type RbmWasmModule = typeof import("../../wasm-core/pkg/hopfield_energy_wasm.js");
type RbmCore = InstanceType<RbmWasmModule["RbmCore"]>;

const TOLERANCE = 1e-3;

let datasetName: DatasetName = "mnist";
let config: RBMTrainingConfig | null = null;
let trainingSamples: Float32Array[] = [];
let referencePatterns: Float32Array[] = [];
let currentQuery = createBlankVisiblePattern();
let currentSnapshot: RBMSnapshot | null = null;
let trainingTimer: number | null = null;
let playbackTimer: number | null = null;
let currentEpoch = 0;
let trainingInFlight = false;

let wasmModulePromise: Promise<RbmWasmModule> | null = null;
let core: RbmCore | null = null;
const wasmBinaryUrl = new URL("../../wasm-core/pkg/hopfield_energy_wasm_bg.wasm", import.meta.url);

function post(message: RBMWorkerResponse): void {
  self.postMessage(message);
}

function cloneSnapshot(snapshot: RBMSnapshot): RBMSnapshot {
  return {
    ...snapshot,
    visible: snapshot.visible.slice(),
    reconstruction: snapshot.reconstruction.slice(),
    hiddenProbabilities: snapshot.hiddenProbabilities.slice(),
    hiddenState: snapshot.hiddenState.slice(),
  };
}

function pauseTraining(postMessage = false): void {
  if (trainingTimer !== null) {
    self.clearInterval(trainingTimer);
    trainingTimer = null;
  }
  trainingInFlight = false;
  if (postMessage) {
    post({ type: "trainingPaused" });
  }
}

function pausePlayback(postMessage = false): void {
  if (playbackTimer !== null) {
    self.clearInterval(playbackTimer);
    playbackTimer = null;
  }
  if (postMessage) {
    post({ type: "playbackPaused" });
  }
}

async function loadWasmModule(): Promise<RbmWasmModule> {
  if (!wasmModulePromise) {
    wasmModulePromise = import("../../wasm-core/pkg/hopfield_energy_wasm.js").then(async (module) => {
      const wasm = module as RbmWasmModule;
      await wasm.default(wasmBinaryUrl);
      return wasm;
    });
  }
  return wasmModulePromise;
}

function parseVisibleModel(module: RbmWasmModule, visibleModel: RBMVisibleModel) {
  return visibleModel === "gaussian" ? module.RbmVisibleModelKind.Gaussian : module.RbmVisibleModelKind.Bernoulli;
}

function flattenPatterns(patterns: Float32Array[]): Float32Array {
  return Float32Array.from(patterns.flatMap((pattern) => Array.from(pattern)));
}

function createModelFromCore(stateConfig: RBMTrainingConfig, stateCore: RbmCore): RBMModel {
  return {
    visibleBias: stateCore.visible_bias(),
    hiddenBias: stateCore.hidden_bias(),
    weights: stateCore.weights(),
    visibleUnits: currentQuery.length,
    hiddenUnits: stateConfig.hiddenUnits,
    visibleModel: stateConfig.visibleModel,
  };
}

function createSnapshotFromCore(stateCore: RbmCore, step: number): RBMSnapshot {
  return {
    visible: stateCore.visible(),
    reconstruction: stateCore.reconstruction(),
    hiddenProbabilities: stateCore.hidden_probabilities(),
    hiddenState: Uint8Array.from(stateCore.hidden_state()),
    freeEnergy: stateCore.free_energy(),
    reconstructionError: stateCore.reconstruction_error(),
    step,
    converged: stateCore.converged(),
    matchedPatternIndex: stateCore.matched_pattern_index(),
  };
}

async function initializeWorker(nextDatasetName: DatasetName, nextConfig: RBMTrainingConfig): Promise<void> {
  pauseTraining(false);
  pausePlayback(false);

  datasetName = nextDatasetName;
  config = { ...nextConfig };

  const archive = await loadDatasetArchive(datasetName);
  trainingSamples = archive.samples.map((sample) => quantizeVisiblePattern(grayscaleToFloat32(sample.pattern), nextConfig.visibleModel));
  referencePatterns = selectMemorySamples(archive).map((sample) => quantizeVisiblePattern(grayscaleToFloat32(sample.pattern), nextConfig.visibleModel));
  currentEpoch = 0;
  currentQuery = createBlankVisiblePattern();

  const module = await loadWasmModule();
  core?.free();
  core = new module.RbmCore(
    currentQuery.length,
    nextConfig.hiddenUnits,
    parseVisibleModel(module, nextConfig.visibleModel),
    flattenPatterns(trainingSamples),
    trainingSamples.length,
    flattenPatterns(referencePatterns),
    referencePatterns.length,
  );
  core.set_query(currentQuery.slice());
  currentSnapshot = createSnapshotFromCore(core, 0);
  const model = createModelFromCore(nextConfig, core);

  post({
    type: "initialized",
    backend: "wasm-core",
    model,
    snapshot: cloneSnapshot(currentSnapshot),
    featureMaps: buildFeatureMaps(model, buildFeatureStrength(model)),
    epoch: currentEpoch,
    trainingError: 0,
  });
}

function ensureState(): { core: RbmCore; config: RBMTrainingConfig; snapshot: RBMSnapshot } {
  if (!core || !config || !currentSnapshot) {
    throw new Error("RBM worker received a command before initialization completed.");
  }
  return { core, config, snapshot: currentSnapshot };
}

async function trainSingleEpoch(): Promise<boolean> {
  if (trainingInFlight) {
    return true;
  }

  trainingInFlight = true;

  try {
    const state = ensureState();
    if (currentEpoch >= state.config.epochs || trainingSamples.length === 0) {
      pauseTraining(true);
      return false;
    }

    const metricsVector = state.core.train_epoch(
      state.config.learningRate,
      state.config.batchSize ?? 25,
      state.config.cdSteps ?? 3,
      state.config.momentum ?? 0.72,
      state.config.weightDecay ?? 0.00015,
    );
    currentEpoch = state.core.epoch();
    state.core.set_query(currentQuery.slice());
    currentSnapshot = createSnapshotFromCore(state.core, 0);
    const model = createModelFromCore(state.config, state.core);
    const metrics: RBMEpochMetrics = {
      epoch: metricsVector[0] ?? currentEpoch,
      reconstructionError: metricsVector[1] ?? state.core.reconstruction_error(),
      contrastiveGap: metricsVector[2] ?? 0,
      freeEnergy: metricsVector[3] ?? state.core.free_energy(),
      hiddenActivation: metricsVector[4] ?? 0,
      weightMeanAbs: metricsVector[5] ?? 0,
    };

    post({
      type: "trainingEpoch",
      backend: "wasm-core",
      model,
      snapshot: cloneSnapshot(currentSnapshot),
      featureMaps: buildFeatureMaps(model, buildFeatureStrength(model)),
      epoch: currentEpoch,
      trainingError: metrics.reconstructionError,
      metrics,
    });

    if (currentEpoch >= state.config.epochs) {
      pauseTraining(true);
      return false;
    }

    return true;
  } finally {
    trainingInFlight = false;
  }
}

function setQuery(pattern: Float32Array): void {
  const state = ensureState();
  pausePlayback(false);
  currentQuery = quantizeVisiblePattern(pattern, state.config.visibleModel);
  state.core.set_query(currentQuery.slice());
  currentSnapshot = createSnapshotFromCore(state.core, 0);
  post({ type: "snapshot", snapshot: cloneSnapshot(currentSnapshot) });
}

function resetPlayback(): void {
  const state = ensureState();
  pausePlayback(false);
  state.core.set_query(currentQuery.slice());
  currentSnapshot = createSnapshotFromCore(state.core, 0);
  post({ type: "snapshot", snapshot: cloneSnapshot(currentSnapshot) });
}

function stepPlayback(): void {
  const state = ensureState();
  pausePlayback(false);
  state.core.step(TOLERANCE);
  currentSnapshot = createSnapshotFromCore(state.core, state.snapshot.step + 1);
  post({ type: "snapshot", snapshot: cloneSnapshot(currentSnapshot) });
}

self.onmessage = (event: MessageEvent<RBMWorkerRequest>) => {
  const message = event.data;

  void (async () => {
    try {
      if (message.type === "initialize") {
        await initializeWorker(message.datasetName, {
          hiddenUnits: message.hiddenUnits,
          epochs: message.epochs,
          learningRate: message.learningRate,
          visibleModel: message.visibleModel,
          batchSize: message.batchSize,
          cdSteps: message.cdSteps,
          momentum: message.momentum,
          weightDecay: message.weightDecay,
        });
        return;
      }

      if (message.type === "setQuery") {
        setQuery(message.pattern);
        return;
      }

      if (message.type === "trainEpoch") {
        pauseTraining(false);
        await trainSingleEpoch();
        return;
      }

      if (message.type === "startTraining") {
        pausePlayback(false);
        pauseTraining(false);
        trainingTimer = self.setInterval(() => {
          void trainSingleEpoch().then((shouldContinue) => {
            if (!shouldContinue) {
              pauseTraining(false);
            }
          });
        }, Math.max(16, message.intervalMs));
        return;
      }

      if (message.type === "pauseTraining") {
        pauseTraining(true);
        return;
      }

      if (message.type === "resetTraining") {
        if (!config) {
          throw new Error("RBM worker is not initialized.");
        }
        await initializeWorker(datasetName, config);
        return;
      }

      if (message.type === "startPlayback") {
        pausePlayback(false);
        let stepCount = 0;
        playbackTimer = self.setInterval(() => {
          stepPlayback();
          stepCount += 1;
          if (!currentSnapshot || currentSnapshot.converged || stepCount >= message.maxSteps) {
            pausePlayback(true);
          }
        }, Math.max(40, message.intervalMs));
        return;
      }

      if (message.type === "pausePlayback") {
        pausePlayback(true);
        return;
      }

      if (message.type === "stepPlayback") {
        stepPlayback();
        return;
      }

      if (message.type === "resetPlayback") {
        resetPlayback();
        return;
      }
    } catch (error) {
      pauseTraining(false);
      pausePlayback(false);
      post({
        type: "error",
        message: error instanceof Error ? error.message : "Unknown RBM worker error",
      });
    }
  })();
};
