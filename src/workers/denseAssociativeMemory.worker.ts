/// <reference lib="webworker" />

import { grayscaleToFloat32, loadDatasetArchive, selectMemorySamples, type DatasetName } from "../data/datasetArchives";
import {
  buildDenseAssociativeFeatureMaps,
  buildDenseAssociativeFeatureStrength,
  createBlankDenseAssociativePattern,
  type DenseAssociativeActivation,
  type DenseAssociativeEpochMetrics,
  type DenseAssociativeMemoryModel,
  type DenseAssociativeMemoryTrainingConfig,
  type DenseAssociativeSnapshot,
} from "../core/denseAssociativeMemory";
import type {
  DenseAssociativeWorkerRequest,
  DenseAssociativeWorkerResponse,
} from "../core/denseAssociativeMemoryWorkerProtocol";

type DenseAssociativeWasmModule = typeof import("../../wasm-core/pkg/hopfield_energy_wasm.js");
type DenseAssociativeMemoryCore = InstanceType<DenseAssociativeWasmModule["DenseAssociativeMemoryCore"]>;

const TOLERANCE = 1e-3;

let datasetName: DatasetName = "mnist";
let config: DenseAssociativeMemoryTrainingConfig | null = null;
let trainingSamples: Float32Array[] = [];
let referencePatterns: Float32Array[] = [];
let currentQuery = createBlankDenseAssociativePattern();
let currentSnapshot: DenseAssociativeSnapshot | null = null;
let currentEpoch = 0;
let trainingTimer: number | null = null;
let playbackTimer: number | null = null;
let trainingInFlight = false;

let wasmModulePromise: Promise<DenseAssociativeWasmModule> | null = null;
let core: DenseAssociativeMemoryCore | null = null;
const wasmBinaryUrl = new URL("../../wasm-core/pkg/hopfield_energy_wasm_bg.wasm", import.meta.url);

function post(message: DenseAssociativeWorkerResponse): void {
  self.postMessage(message);
}

function cloneSnapshot(snapshot: DenseAssociativeSnapshot): DenseAssociativeSnapshot {
  return {
    ...snapshot,
    visible: snapshot.visible.slice(),
    reconstruction: snapshot.reconstruction.slice(),
    hiddenActivations: snapshot.hiddenActivations.slice(),
    hiddenScores: snapshot.hiddenScores.slice(),
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

async function loadWasmModule(): Promise<DenseAssociativeWasmModule> {
  if (!wasmModulePromise) {
    wasmModulePromise = import("../../wasm-core/pkg/hopfield_energy_wasm.js").then(async (module) => {
      const wasm = module as DenseAssociativeWasmModule;
      await wasm.default(wasmBinaryUrl);
      return wasm;
    });
  }
  return wasmModulePromise;
}

function parseActivation(module: DenseAssociativeWasmModule, activation: DenseAssociativeActivation) {
  if (activation === "signed-power") {
    return module.DenseAssociativeActivationKind.SignedPower;
  }
  if (activation === "softmax") {
    return module.DenseAssociativeActivationKind.Softmax;
  }
  return module.DenseAssociativeActivationKind.ReluPower;
}

function flattenPatterns(patterns: Float32Array[]): number[] {
  return patterns.flatMap((pattern) => Array.from(pattern));
}

function createModelFromCore(stateConfig: DenseAssociativeMemoryTrainingConfig, stateCore: DenseAssociativeMemoryCore): DenseAssociativeMemoryModel {
  return {
    weights: stateCore.weights(),
    visibleUnits: stateCore.visible_units(),
    hiddenUnits: stateCore.hidden_units(),
    activation: stateConfig.activation,
    sharpness: stateConfig.sharpness,
  };
}

function createSnapshotFromCore(stateCore: DenseAssociativeMemoryCore, step: number): DenseAssociativeSnapshot {
  return {
    visible: stateCore.visible(),
    reconstruction: stateCore.reconstruction(),
    hiddenActivations: stateCore.hidden_activations(),
    hiddenScores: stateCore.hidden_scores(),
    energy: stateCore.energy(),
    reconstructionError: stateCore.reconstruction_error(),
    step,
    converged: stateCore.converged(),
    matchedPatternIndex: stateCore.matched_pattern_index(),
    topHiddenIndex: stateCore.top_hidden_index(),
    topHiddenActivation: stateCore.top_hidden_activation(),
    hiddenEntropy: stateCore.hidden_entropy(),
  };
}

function buildFeatureMapsForModel(model: DenseAssociativeMemoryModel) {
  return buildDenseAssociativeFeatureMaps(model, buildDenseAssociativeFeatureStrength(model));
}

async function initializeWorker(nextDatasetName: DatasetName, nextConfig: DenseAssociativeMemoryTrainingConfig): Promise<void> {
  pauseTraining(false);
  pausePlayback(false);

  datasetName = nextDatasetName;
  config = { ...nextConfig };

  const archive = await loadDatasetArchive(datasetName);
  trainingSamples = archive.samples.map((sample) => grayscaleToFloat32(sample.pattern));
  referencePatterns = selectMemorySamples(archive).map((sample) => grayscaleToFloat32(sample.pattern));
  currentEpoch = 0;
  currentQuery = createBlankDenseAssociativePattern();

  const module = await loadWasmModule();
  core?.free();
  core = new module.DenseAssociativeMemoryCore(
    currentQuery.length,
    nextConfig.hiddenUnits,
    parseActivation(module, nextConfig.activation),
    nextConfig.sharpness,
    Float32Array.from(flattenPatterns(trainingSamples)),
    trainingSamples.length,
    Float32Array.from(flattenPatterns(referencePatterns)),
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
    featureMaps: buildFeatureMapsForModel(model),
    epoch: currentEpoch,
    trainingError: 0,
  });
}

function ensureState(): {
  core: DenseAssociativeMemoryCore;
  config: DenseAssociativeMemoryTrainingConfig;
  snapshot: DenseAssociativeSnapshot;
} {
  if (!core || !config || !currentSnapshot) {
    throw new Error("Dense Associative Memory worker received a command before initialization completed.");
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
      state.config.momentum ?? 0.65,
      state.config.weightDecay ?? 0.0004,
    );
    currentEpoch = state.core.epoch();
      state.core.set_query(currentQuery.slice());
    currentSnapshot = createSnapshotFromCore(state.core, 0);
    const model = createModelFromCore(state.config, state.core);
    const metrics: DenseAssociativeEpochMetrics = {
      epoch: metricsVector[0] ?? currentEpoch,
      reconstructionError: metricsVector[1] ?? state.core.reconstruction_error(),
      contrastiveGap: metricsVector[2] ?? 0,
      hiddenActivation: metricsVector[3] ?? 0,
      winnerShare: metricsVector[4] ?? 0,
      weightMeanAbs: metricsVector[5] ?? 0,
      energy: metricsVector[6] ?? state.core.energy(),
    };

    post({
      type: "trainingEpoch",
      backend: "wasm-core",
      model,
      snapshot: cloneSnapshot(currentSnapshot),
      featureMaps: buildFeatureMapsForModel(model),
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
  currentQuery = pattern.slice();
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

self.onmessage = (event: MessageEvent<DenseAssociativeWorkerRequest>) => {
  const message = event.data;

  void (async () => {
    try {
      if (message.type === "initialize") {
        await initializeWorker(message.datasetName, {
          hiddenUnits: message.hiddenUnits,
          epochs: message.epochs,
          learningRate: message.learningRate,
          batchSize: message.batchSize,
          sharpness: message.sharpness,
          activation: message.activation,
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
          throw new Error("Dense Associative Memory worker is not initialized.");
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
      }
    } catch (error) {
      pauseTraining(false);
      pausePlayback(false);
      post({
        type: "error",
        message: error instanceof Error ? error.message : "Unknown Dense Associative Memory worker error",
      });
    }
  })();
};
