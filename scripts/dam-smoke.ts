import { grayscaleToFloat32, selectMemorySamples, type DatasetName } from "../src/data/datasetArchives";
import { loadDatasetArchiveFromNodeFs } from "./shared/datasetArchivesNode";

type DamWasmModule = typeof import("../wasm-core/pkg-node/hopfield_energy_wasm.js");

interface SmokeOptions {
  dataset: DatasetName;
  hiddenUnits: number;
  epochs: number;
  learningRate: number;
  batchSize: number;
  sharpness: number;
  activation: "relu-power" | "signed-power" | "softmax";
  momentum: number;
  weightDecay: number;
  retrievalSteps: number;
}

function parseArgs(argv: string[]): SmokeOptions {
  const values = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith("--")) {
      continue;
    }
    const key = token.slice(2);
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) {
      values.set(key, "true");
      continue;
    }
    values.set(key, value);
    index += 1;
  }

  return {
    dataset: (values.get("dataset") ?? "mnist") as DatasetName,
    hiddenUnits: Number(values.get("hidden-units") ?? 96),
    epochs: Number(values.get("epochs") ?? 8),
    learningRate: Number(values.get("learning-rate") ?? 0.035),
    batchSize: Number(values.get("batch-size") ?? 25),
    sharpness: Number(values.get("sharpness") ?? 8),
    activation: (values.get("activation") ?? "relu-power") as SmokeOptions["activation"],
    momentum: Number(values.get("momentum") ?? 0.65),
    weightDecay: Number(values.get("weight-decay") ?? 0.0004),
    retrievalSteps: Number(values.get("retrieval-steps") ?? 10),
  };
}

function flattenPatterns(patterns: Float32Array[]): Float32Array {
  return Float32Array.from(patterns.flatMap((pattern) => Array.from(pattern)));
}

function applyDenseAssociativeNoise(pattern: Float32Array, corruptionPercent: number, obfuscationPercent: number): Float32Array {
  const next = pattern.slice();
  const total = next.length;
  const corruptionCount = Math.round((total * corruptionPercent) / 100);
  const obfuscationCount = Math.round((total * obfuscationPercent) / 100);

  const corruptionIndices = shuffleIndices(total);
  for (let index = 0; index < corruptionCount; index += 1) {
    const target = corruptionIndices[index]!;
    next[target] = 1 - next[target]!;
  }

  const obfuscationIndices = shuffleIndices(total);
  for (let index = 0; index < obfuscationCount; index += 1) {
    next[obfuscationIndices[index]!] = 0;
  }

  return next;
}

function shuffleIndices(size: number): Uint16Array {
  const indices = new Uint16Array(size);
  for (let index = 0; index < size; index += 1) {
    indices[index] = index;
  }
  for (let index = size - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    const current = indices[index]!;
    indices[index] = indices[swapIndex]!;
    indices[swapIndex] = current;
  }
  return indices;
}

function parseActivation(module: DamWasmModule, activation: SmokeOptions["activation"]) {
  if (activation === "signed-power") {
    return module.DenseAssociativeActivationKind.SignedPower;
  }
  if (activation === "softmax") {
    return module.DenseAssociativeActivationKind.Softmax;
  }
  return module.DenseAssociativeActivationKind.ReluPower;
}

function evaluateReconstruction(
  core: InstanceType<DamWasmModule["DenseAssociativeMemoryCore"]>,
  samples: Float32Array[],
  limit = 128,
) {
  const count = Math.min(limit, samples.length);
  let totalError = 0;
  let totalEnergy = 0;
  for (let index = 0; index < count; index += 1) {
    core.set_query(samples[index]!.slice());
    totalError += core.reconstruction_error();
    totalEnergy += core.energy();
  }
  return {
    meanError: totalError / Math.max(count, 1),
    meanEnergy: totalEnergy / Math.max(count, 1),
  };
}

function evaluateRetrieval(
  core: InstanceType<DamWasmModule["DenseAssociativeMemoryCore"]>,
  references: Float32Array[],
  steps: number,
) {
  let exactMatches = 0;
  let totalError = 0;

  for (let index = 0; index < references.length; index += 1) {
    core.set_query(applyDenseAssociativeNoise(references[index]!, 20, 20));
    for (let step = 0; step < steps; step += 1) {
      if (core.converged()) {
        break;
      }
      core.step(1e-3);
    }
    totalError += core.reconstruction_error();
    if (core.matched_pattern_index() === index) {
      exactMatches += 1;
    }
  }

  return {
    exactMatches,
    total: references.length,
    meanError: totalError / Math.max(references.length, 1),
  };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const wasm = (await import("../wasm-core/pkg-node/hopfield_energy_wasm.js")) as DamWasmModule;
  const archive = await loadDatasetArchiveFromNodeFs(options.dataset);
  const samples = archive.samples.map((sample) => grayscaleToFloat32(sample.pattern));
  const references = selectMemorySamples(archive).map((sample) => grayscaleToFloat32(sample.pattern));
  const visibleUnits = samples[0]?.length ?? 784;

  const core = new wasm.DenseAssociativeMemoryCore(
    visibleUnits,
    options.hiddenUnits,
    parseActivation(wasm, options.activation),
    options.sharpness,
    flattenPatterns(samples),
    samples.length,
    flattenPatterns(references),
    references.length,
  );

  const before = evaluateReconstruction(core, samples);
  const epochMetrics = [];
  for (let epoch = 0; epoch < options.epochs; epoch += 1) {
    const metrics = core.train_epoch(options.learningRate, options.batchSize, options.momentum, options.weightDecay);
    epochMetrics.push({
      epoch: metrics[0] ?? core.epoch(),
      reconstructionError: metrics[1] ?? core.reconstruction_error(),
      contrastiveGap: metrics[2] ?? 0,
      hiddenActivation: metrics[3] ?? 0,
      winnerShare: metrics[4] ?? 0,
      weightMeanAbs: metrics[5] ?? 0,
      energy: metrics[6] ?? core.energy(),
    });
  }
  const after = evaluateReconstruction(core, samples);
  const retrieval = evaluateRetrieval(core, references, options.retrievalSteps);

  console.log(
    JSON.stringify(
      {
        backend: "wasm-core",
        dataset: archive.name,
        sampleCount: samples.length,
        referenceCount: references.length,
        config: options,
        reconstruction: {
          before,
          after,
          improvement: before.meanError - after.meanError,
        },
        finalEpoch: epochMetrics[epochMetrics.length - 1] ?? null,
        retrieval,
      },
      null,
      2,
    ),
  );
}

void main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
