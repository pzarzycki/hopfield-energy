import type {
  BackendInfo,
  ConnectionSnapshot,
  ModelInitialization,
  ModelSession,
  ModelSnapshot,
  ModelTopology,
  ModelTrace,
  ReconstructionOptions,
} from "../../src/core/api/modelApi";

type DenseAssociativeWasmModule = typeof import("../../wasm-core/pkg-node/hopfield_energy_wasm.js");

interface DenseAssociativeWasmInit {
  visibleUnits: number;
  hiddenUnits: number;
  activation: "relu-power" | "signed-power" | "softmax";
  sharpness: number;
  trainingPatterns: Float32Array[];
  referencePatterns: Float32Array[];
}

const backendInfo: BackendInfo = {
  kind: "wasm-core",
  version: "0.1.0",
  capabilities: ["train-epoch", "inspect", "step", "reconstruct", "weights", "layer-state", "metrics"],
};

function flattenPatterns(patterns: Float32Array[], visibleUnits: number): number[] {
  return patterns.flatMap((pattern) => Array.from(pattern.slice(0, visibleUnits)));
}

function parseActivation(module: DenseAssociativeWasmModule, activation: DenseAssociativeWasmInit["activation"]) {
  if (activation === "signed-power") {
    return module.DenseAssociativeActivationKind.SignedPower;
  }
  if (activation === "softmax") {
    return module.DenseAssociativeActivationKind.Softmax;
  }
  return module.DenseAssociativeActivationKind.ReluPower;
}

function createTopology(visibleUnits: number, hiddenUnits: number): ModelTopology {
  return {
    family: "dense-associative-memory",
    title: "Dense Associative Memory",
    layers: [
      {
        id: "visible",
        label: "Visible",
        role: "visible",
        units: visibleUnits,
        stateKinds: ["activation"],
      },
      {
        id: "hidden",
        label: "Hidden",
        role: "hidden",
        units: hiddenUnits,
        stateKinds: ["activation"],
      },
    ],
    connections: [
      {
        id: "weights",
        from: "visible",
        to: "hidden",
        kind: "weights",
        rows: hiddenUnits,
        cols: visibleUnits,
      },
    ],
  };
}

export class DenseAssociativeMemoryWasmSession implements ModelSession {
  private wasm: DenseAssociativeWasmModule | null = null;
  private core: InstanceType<DenseAssociativeWasmModule["DenseAssociativeMemoryCore"]> | null = null;
  private topology: ModelTopology | null = null;
  private config: DenseAssociativeWasmInit | null = null;

  async initialize(init: ModelInitialization): Promise<void> {
    if (init.family !== "dense-associative-memory") {
      throw new Error(`DenseAssociativeMemoryWasmSession cannot initialize family ${init.family}.`);
    }
    if (!init.dataset?.patterns?.length) {
      throw new Error("Dense Associative Memory initialization requires dataset patterns.");
    }

    const visibleUnits = Number(init.architecture.visibleUnits);
    const hiddenUnits = Number(init.architecture.hiddenUnits);
    const activation = String(init.architecture.activation ?? "relu-power") as DenseAssociativeWasmInit["activation"];
    const sharpness = Number(init.architecture.sharpness ?? 8);
    const referencePatterns = init.dataset.referencePatterns ?? init.dataset.patterns;

    this.wasm = (await import("../../wasm-core/pkg-node/hopfield_energy_wasm.js")) as DenseAssociativeWasmModule;
    this.config = {
      visibleUnits,
      hiddenUnits,
      activation,
      sharpness,
      trainingPatterns: init.dataset.patterns,
      referencePatterns,
    };
    this.topology = createTopology(visibleUnits, hiddenUnits);
    this.core = new this.wasm.DenseAssociativeMemoryCore(
      visibleUnits,
      hiddenUnits,
      parseActivation(this.wasm, activation),
      sharpness,
      flattenPatterns(init.dataset.patterns, visibleUnits),
      init.dataset.patterns.length,
      flattenPatterns(referencePatterns, visibleUnits),
      referencePatterns.length,
    );
  }

  getTopology(): ModelTopology {
    if (!this.topology) {
      throw new Error("Dense Associative Memory Wasm session is not initialized.");
    }
    return this.topology;
  }

  getBackendInfo(): BackendInfo {
    return backendInfo;
  }

  async setQuery(pattern: Float32Array): Promise<void> {
    this.requireCore().set_query(Array.from(pattern));
  }

  async trainEpoch(): Promise<ModelTrace> {
    const core = this.requireCore();
    const init = this.requireConfig();
    const metrics = core.train_epoch(
      Number(init.trainingPatterns.length > 0 ? 0.035 : 0.035),
      25,
      0.65,
      0.0004,
    );
    const snapshot = this.captureSnapshot("training-negative");
    snapshot.metrics.push(
      { id: "epoch", label: "Epoch", value: metrics[0] ?? 0 },
      { id: "reconstruction-error", label: "Reconstruction Error", value: metrics[1] ?? 0 },
      { id: "contrastive-gap", label: "Contrastive Gap", value: metrics[2] ?? 0 },
      { id: "hidden-activation", label: "Hidden Activation", value: metrics[3] ?? 0 },
      { id: "winner-share", label: "Winner Share", value: metrics[4] ?? 0 },
      { id: "weight-mean-abs", label: "Weight Mean Abs", value: metrics[5] ?? 0 },
      { id: "energy", label: "Energy", value: metrics[6] ?? 0 },
    );
    return {
      topology: this.getTopology(),
      snapshots: [snapshot],
      final: snapshot,
    };
  }

  async reconstruct(options: ReconstructionOptions): Promise<ModelTrace> {
    const snapshots: ModelSnapshot[] = [this.captureSnapshot("reconstruction")];
    for (let index = 0; index < options.maxSteps; index += 1) {
      const previous = snapshots[snapshots.length - 1];
      if (previous.converged) {
        break;
      }
      await this.stepInternal(options.tolerance);
      snapshots.push(this.captureSnapshot("reconstruction"));
      if (snapshots[snapshots.length - 1]?.converged) {
        break;
      }
    }
    return {
      topology: this.getTopology(),
      snapshots,
      final: snapshots[snapshots.length - 1]!,
    };
  }

  async generate(options: { maxSteps: number; temperature?: number }): Promise<ModelTrace> {
    return this.reconstruct({ maxSteps: options.maxSteps, tolerance: 1e-3 });
  }

  async step(): Promise<ModelSnapshot> {
    await this.stepInternal(1e-3);
    return this.captureSnapshot("reconstruction");
  }

  async reset(): Promise<ModelSnapshot> {
    this.requireCore().inspect();
    return this.captureSnapshot("idle");
  }

  async inspect(): Promise<ModelSnapshot> {
    this.requireCore().inspect();
    return this.captureSnapshot("idle");
  }

  async dispose(): Promise<void> {
    this.core = null;
    this.wasm = null;
    this.topology = null;
    this.config = null;
  }

  private async stepInternal(tolerance: number): Promise<void> {
    this.requireCore().step(tolerance);
  }

  private captureSnapshot(phase: ModelSnapshot["phase"]): ModelSnapshot {
    const core = this.requireCore();
    const topology = this.getTopology();
    const connections: ConnectionSnapshot[] = [
      {
        connectionId: "weights",
        values: Float32Array.from(core.weights()),
        rows: topology.connections[0]!.rows,
        cols: topology.connections[0]!.cols,
      },
    ];

    return {
      phase,
      step: core.step_index(),
      converged: core.converged(),
      matchedPatternIndex: core.matched_pattern_index(),
      visiblePattern: Float32Array.from(core.visible()),
      outputPattern: Float32Array.from(core.reconstruction()),
      layers: [
        {
          layerId: "visible",
          activations: Float32Array.from(core.visible()),
        },
        {
          layerId: "hidden",
          activations: Float32Array.from(core.hidden_activations()),
        },
      ],
      connections,
      metrics: [
        { id: "energy", label: "Energy", value: core.energy() },
        {
          id: "reconstruction-error",
          label: "Reconstruction Error",
          value: core.reconstruction_error(),
        },
        {
          id: "top-hidden-index",
          label: "Top Hidden Index",
          value: core.top_hidden_index(),
        },
        {
          id: "top-hidden-activation",
          label: "Top Hidden Activation",
          value: core.top_hidden_activation(),
        },
        {
          id: "hidden-entropy",
          label: "Hidden Entropy",
          value: core.hidden_entropy(),
        },
        {
          id: "epoch",
          label: "Epoch",
          value: core.epoch(),
        },
      ],
    };
  }

  private requireCore() {
    if (!this.core) {
      throw new Error("Dense Associative Memory Wasm session is not initialized.");
    }
    return this.core;
  }

  private requireConfig() {
    if (!this.config) {
      throw new Error("Dense Associative Memory Wasm session is not initialized.");
    }
    return this.config;
  }
}
