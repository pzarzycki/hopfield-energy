import { DenseAssociativeMemoryWasmSession } from "../shared/denseAssociativeMemoryWasmSession";

interface SmokeSummary {
  config: {
    visibleUnits: number;
    hiddenUnits: number;
    activation: string;
    sharpness: number;
  };
  backend: string;
  topology: { layers: number; connections: number };
  initial: Record<string, number>;
  afterTrain: Record<string, number>;
  afterStep: Record<string, number>;
}

function makeToySamples(): number[] {
  return [
    1, 0, 1, 0,
    0, 1, 0, 1,
    1, 1, 0, 0,
    0, 0, 1, 1,
  ];
}

function parseArgs(argv: string[]) {
  const values = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith("--")) {
      continue;
    }
    const key = token.slice(2);
    const next = argv[index + 1];
    if (!next || next.startsWith("--")) {
      values.set(key, "true");
      continue;
    }
    values.set(key, next);
    index += 1;
  }

  return {
    visibleUnits: Number(values.get("visible-units") ?? 4),
    hiddenUnits: Number(values.get("hidden-units") ?? 3),
    sharpness: Number(values.get("sharpness") ?? 4),
    activation: values.get("activation") ?? "relu-power",
    learningRate: Number(values.get("learning-rate") ?? 0.05),
    batchSize: Number(values.get("batch-size") ?? 1),
    momentum: Number(values.get("momentum") ?? 0.65),
    weightDecay: Number(values.get("weight-decay") ?? 0.0004),
    tolerance: Number(values.get("tolerance") ?? 0.001),
  };
}

const options = parseArgs(process.argv.slice(2));
const trainingPatterns = [
  Float32Array.from([1, 0, 1, 0]),
  Float32Array.from([0, 1, 0, 1]),
  Float32Array.from([1, 1, 0, 0]),
  Float32Array.from([0, 0, 1, 1]),
];

const session = new DenseAssociativeMemoryWasmSession();
await session.initialize({
  family: "dense-associative-memory",
  architecture: {
    visibleUnits: options.visibleUnits,
    hiddenUnits: options.hiddenUnits,
    activation: options.activation,
    sharpness: options.sharpness,
  },
  training: {
    learningRate: options.learningRate,
    batchSize: options.batchSize,
    momentum: options.momentum,
    weightDecay: options.weightDecay,
  },
  dataset: {
    patterns: trainingPatterns,
    referencePatterns: trainingPatterns,
  },
});

await session.setQuery(Float32Array.from([1, 0, 1, 0]));
const initial = await session.inspect();
const afterTrainTrace = await session.trainEpoch();
const afterStep = await session.step();
const topology = session.getTopology();
const backendInfo = session.getBackendInfo();

function metricMap(snapshot: { metrics: Array<{ id: string; value: number }> }) {
  return Object.fromEntries(snapshot.metrics.map((metric) => [metric.id, metric.value]));
}

const summary: SmokeSummary = {
  config: {
    visibleUnits: options.visibleUnits,
    hiddenUnits: options.hiddenUnits,
    activation: options.activation,
    sharpness: options.sharpness,
  },
  backend: backendInfo.kind,
  topology: {
    layers: topology.layers.length,
    connections: topology.connections.length,
  },
  initial: metricMap(initial),
  afterTrain: metricMap(afterTrainTrace.final),
  afterStep: metricMap(afterStep),
};

console.log(JSON.stringify(summary, null, 2));
