/// <reference lib="webworker" />

import { grayscaleToFloat32, loadDatasetArchive, selectMemorySamples, type DatasetName } from "../data/datasetArchives";
import type { DenseSnapshot } from "../core/denseHopfield";
import type {
  DenseHopfieldMemoryView,
  DenseHopfieldWorkerRequest,
  DenseHopfieldWorkerResponse,
} from "../core/denseHopfieldWorkerProtocol";

type DenseHopfieldWasmModule = typeof import("../../wasm-core/pkg/hopfield_energy_wasm.js");
type DenseHopfieldCore = InstanceType<DenseHopfieldWasmModule["DenseHopfieldCore"]>;

const TOLERANCE = 1e-4;

let datasetName: DatasetName = "mnist";
let currentQuery = new Float32Array(784);
let currentSnapshot: DenseSnapshot | null = null;
let currentMemorySet: DenseHopfieldMemoryView | null = null;
let playbackTimer: number | null = null;

let wasmModulePromise: Promise<DenseHopfieldWasmModule> | null = null;
let core: DenseHopfieldCore | null = null;
const wasmBinaryUrl = new URL("../../wasm-core/pkg/hopfield_energy_wasm_bg.wasm", import.meta.url);

function post(message: DenseHopfieldWorkerResponse): void {
  self.postMessage(message);
}

function pausePlayback(postMessage = false): void {
  if (playbackTimer !== null) {
    self.clearInterval(playbackTimer);
    playbackTimer = null;
  }
  if (postMessage) {
    post({ type: "paused" });
  }
}

async function loadWasmModule(): Promise<DenseHopfieldWasmModule> {
  if (!wasmModulePromise) {
    wasmModulePromise = import("../../wasm-core/pkg/hopfield_energy_wasm.js").then(async (module) => {
      const wasm = module as DenseHopfieldWasmModule;
      await wasm.default(wasmBinaryUrl);
      return wasm;
    });
  }
  return wasmModulePromise;
}

function flattenPatterns(patterns: Float32Array[]): Float32Array {
  return Float32Array.from(patterns.flatMap((pattern) => Array.from(pattern)));
}

function ensureCore(): DenseHopfieldCore {
  if (!core) {
    throw new Error("Dense Hopfield worker received a command before initialization completed.");
  }
  return core;
}

function createSnapshot(stateCore: DenseHopfieldCore): DenseSnapshot {
  return {
    state: stateCore.state(),
    energy: stateCore.energy(),
    step: stateCore.step_index(),
    delta: stateCore.delta(),
    matchedPatternIndex: stateCore.matched_pattern_index(),
    converged: stateCore.converged(),
    attention: stateCore.attention(),
    entropy: stateCore.entropy(),
    topAttention: stateCore.top_attention(),
  };
}

function cloneMemorySet(memorySet: DenseHopfieldMemoryView): DenseHopfieldMemoryView {
  return {
    labels: [...memorySet.labels],
    patterns: memorySet.patterns.map((pattern) => pattern.slice()),
    similarityMatrix: memorySet.similarityMatrix.slice(),
    maxSimilarityAbs: memorySet.maxSimilarityAbs,
    description: memorySet.description,
  };
}

function cloneSnapshot(snapshot: DenseSnapshot): DenseSnapshot {
  return {
    ...snapshot,
    state: snapshot.state.slice(),
    attention: snapshot.attention.slice(),
  };
}

async function initializeWorker(nextDatasetName: DatasetName, beta: number): Promise<void> {
  pausePlayback(false);
  datasetName = nextDatasetName;

  const archive = await loadDatasetArchive(datasetName);
  const memorySamples = selectMemorySamples(archive);
  const labels = memorySamples.map((sample) => sample.labelName);
  const patterns = memorySamples.map((sample) => grayscaleToFloat32(sample.pattern));
  const description = `Real grayscale ${archive.name} exemplars loaded from the bundled binary archive.`;

  const module = await loadWasmModule();
  core?.free();
  core = new module.DenseHopfieldCore(flattenPatterns(patterns), patterns.length, patterns[0]?.length ?? currentQuery.length, beta);
  currentQuery = new Float32Array(patterns[0]?.length ?? currentQuery.length);
  core.set_state(Float32Array.from(currentQuery));
  currentSnapshot = createSnapshot(core);
  currentMemorySet = {
    labels,
    patterns,
    similarityMatrix: core.similarity_matrix(),
    maxSimilarityAbs: core.max_similarity_abs(),
    description,
  };

  post({
    type: "initialized",
    backend: "wasm-core",
    memorySet: cloneMemorySet(currentMemorySet),
    snapshot: cloneSnapshot(currentSnapshot),
  });
}

function setQuery(pattern: Float32Array): void {
  const activeCore = ensureCore();
  pausePlayback(false);
  currentQuery = pattern.slice();
  activeCore.set_state(Float32Array.from(currentQuery));
  currentSnapshot = createSnapshot(activeCore);
  post({ type: "snapshot", snapshot: cloneSnapshot(currentSnapshot) });
}

function setBeta(beta: number): void {
  const activeCore = ensureCore();
  pausePlayback(false);
  activeCore.set_beta(beta);
  currentSnapshot = createSnapshot(activeCore);
  post({ type: "snapshot", snapshot: cloneSnapshot(currentSnapshot) });
}

function resetState(): void {
  const activeCore = ensureCore();
  pausePlayback(false);
  activeCore.set_state(Float32Array.from(currentQuery));
  currentSnapshot = createSnapshot(activeCore);
  post({ type: "snapshot", snapshot: cloneSnapshot(currentSnapshot) });
}

function stepState(): void {
  const activeCore = ensureCore();
  pausePlayback(false);
  activeCore.step(TOLERANCE);
  currentSnapshot = createSnapshot(activeCore);
  post({ type: "snapshot", snapshot: cloneSnapshot(currentSnapshot) });
}

self.onmessage = (event: MessageEvent<DenseHopfieldWorkerRequest>) => {
  const message = event.data;

  void (async () => {
    try {
      if (message.type === "initialize") {
        await initializeWorker(message.datasetName, message.beta);
        return;
      }

      if (message.type === "setQuery") {
        setQuery(message.pattern);
        return;
      }

      if (message.type === "setBeta") {
        setBeta(message.beta);
        return;
      }

      if (message.type === "step") {
        stepState();
        return;
      }

      if (message.type === "play") {
        pausePlayback(false);
        let stepCount = 0;
        playbackTimer = self.setInterval(() => {
          stepState();
          stepCount += 1;
          if (!currentSnapshot || currentSnapshot.converged || stepCount >= message.maxSteps) {
            pausePlayback(true);
          }
        }, Math.max(40, message.intervalMs));
        return;
      }

      if (message.type === "pause") {
        pausePlayback(true);
        return;
      }

      if (message.type === "reset") {
        resetState();
      }
    } catch (error) {
      pausePlayback(false);
      post({
        type: "error",
        message: error instanceof Error ? error.message : "Unknown Dense Hopfield worker error",
      });
    }
  })();
};
