import { grayscaleToFloat32, selectMemorySamples, type DatasetName } from "../src/data/datasetArchives";
import { loadDatasetArchiveFromNodeFs } from "./shared/datasetArchivesNode";

type RbmWasmModule = typeof import("../wasm-core/pkg-node/hopfield_energy_wasm.js");

interface SmokeOptions {
  dataset: DatasetName;
  hiddenUnits: number;
  epochs: number;
  learningRate: number;
  batchSize: number;
  cdSteps: number;
  momentum: number;
  weightDecay: number;
  visibleModel: "bernoulli" | "gaussian";
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
    hiddenUnits: Number(values.get("hidden-units") ?? 64),
    epochs: Number(values.get("epochs") ?? 5),
    learningRate: Number(values.get("learning-rate") ?? 0.05),
    batchSize: Number(values.get("batch-size") ?? 25),
    cdSteps: Number(values.get("cd-steps") ?? 3),
    momentum: Number(values.get("momentum") ?? 0.72),
    weightDecay: Number(values.get("weight-decay") ?? 0.00015),
    visibleModel: (values.get("visible-model") ?? "bernoulli") as SmokeOptions["visibleModel"],
    retrievalSteps: Number(values.get("retrieval-steps") ?? 8),
  };
}

function flattenPatterns(patterns: Float32Array[]): Float32Array {
  return Float32Array.from(patterns.flatMap((pattern) => Array.from(pattern)));
}

function quantizeVisiblePattern(pattern: Float32Array, visibleModel: SmokeOptions["visibleModel"]): Float32Array {
  if (visibleModel === "gaussian") {
    return pattern.slice();
  }
  const next = new Float32Array(pattern.length);
  for (let index = 0; index < pattern.length; index += 1) {
    next[index] = pattern[index]! >= 0.5 ? 1 : 0;
  }
  return next;
}

function applyVisibleNoise(
  pattern: Float32Array,
  corruptionPercent: number,
  obfuscationPercent: number,
  visibleModel: SmokeOptions["visibleModel"],
): Float32Array {
  const next = pattern.slice();
  const total = next.length;
  const corruptionCount = Math.round((total * corruptionPercent) / 100);
  const obfuscationCount = Math.round((total * obfuscationPercent) / 100);

  const corruptionIndices = shuffleIndices(total);
  for (let index = 0; index < corruptionCount; index += 1) {
    const target = corruptionIndices[index]!;
    if (visibleModel === "gaussian") {
      next[target] = 1 - next[target]!;
    } else {
      next[target] = next[target]! > 0.5 ? 0 : 1;
    }
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

function parseVisibleModel(module: RbmWasmModule, visibleModel: SmokeOptions["visibleModel"]) {
  return visibleModel === "gaussian" ? module.RbmVisibleModelKind.Gaussian : module.RbmVisibleModelKind.Bernoulli;
}

function evaluateTrainingSet(
  core: InstanceType<RbmWasmModule["RbmCore"]>,
  samples: Float32Array[],
  limit = 128,
) {
  const count = Math.min(limit, samples.length);
  let totalError = 0;
  let totalEnergy = 0;
  for (let index = 0; index < count; index += 1) {
    core.set_query(samples[index]!.slice());
    totalError += core.reconstruction_error();
    totalEnergy += core.free_energy();
  }
  return {
    meanError: totalError / Math.max(count, 1),
    meanFreeEnergy: totalEnergy / Math.max(count, 1),
  };
}

function evaluateRetrieval(
  core: InstanceType<RbmWasmModule["RbmCore"]>,
  references: Float32Array[],
  visibleModel: SmokeOptions["visibleModel"],
  steps: number,
) {
  let exactMatches = 0;
  let totalError = 0;

  for (let index = 0; index < references.length; index += 1) {
    core.set_query(applyVisibleNoise(references[index]!, 20, 20, visibleModel));
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
  const wasm = (await import("../wasm-core/pkg-node/hopfield_energy_wasm.js")) as RbmWasmModule;
  const archive = await loadDatasetArchiveFromNodeFs(options.dataset);
  const samples = archive.samples.map((sample) => quantizeVisiblePattern(grayscaleToFloat32(sample.pattern), options.visibleModel));
  const references = selectMemorySamples(archive).map((sample) =>
    quantizeVisiblePattern(grayscaleToFloat32(sample.pattern), options.visibleModel),
  );
  const visibleUnits = samples[0]?.length ?? 784;

  const core = new wasm.RbmCore(
    visibleUnits,
    options.hiddenUnits,
    parseVisibleModel(wasm, options.visibleModel),
    flattenPatterns(samples),
    samples.length,
    flattenPatterns(references),
    references.length,
  );

  const before = evaluateTrainingSet(core, samples);
  const epochMetrics = [];
  for (let epoch = 0; epoch < options.epochs; epoch += 1) {
    const metrics = core.train_epoch(options.learningRate, options.batchSize, options.cdSteps, options.momentum, options.weightDecay);
    epochMetrics.push({
      epoch: metrics[0] ?? core.epoch(),
      reconstructionError: metrics[1] ?? core.reconstruction_error(),
      contrastiveGap: metrics[2] ?? 0,
      freeEnergy: metrics[3] ?? core.free_energy(),
      hiddenActivation: metrics[4] ?? 0,
      weightMeanAbs: metrics[5] ?? 0,
    });
  }
  const after = evaluateTrainingSet(core, samples);
  const retrieval = evaluateRetrieval(core, references, options.visibleModel, options.retrievalSteps);

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
