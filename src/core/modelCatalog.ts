import type { ModelFamily } from "./api/modelApi";

export type ModelPrimaryTask =
  | "pattern-reconstruction"
  | "associative-retrieval"
  | "generation"
  | "unsupervised-feature-learning"
  | "classification-probe";

export interface ModelCatalogEntry {
  family: ModelFamily;
  path: string;
  label: string;
  status: "implemented" | "placeholder";
  primaryTasks: ModelPrimaryTask[];
  summary: string;
}

export const MODEL_CATALOG: readonly ModelCatalogEntry[] = [
  {
    family: "hopfield",
    path: "/networks/hopfield-network",
    label: "Hopfield Network",
    status: "implemented",
    primaryTasks: ["associative-retrieval", "pattern-reconstruction"],
    summary: "Classical energy-based associative memory over binary visible states.",
  },
  {
    family: "dense-hopfield",
    path: "/networks/dense-hopfield-network",
    label: "Dense Hopfield Network",
    status: "implemented",
    primaryTasks: ["associative-retrieval", "pattern-reconstruction"],
    summary: "Modern attention-like Hopfield retrieval over continuous visible states.",
  },
  {
    family: "dense-associative-memory",
    path: "/networks/dense-associative-memory",
    label: "Dense Associative Memory",
    status: "implemented",
    primaryTasks: ["associative-retrieval", "pattern-reconstruction"],
    summary: "Krotov-style bipartite memory with sharp hidden competition and iterative recall.",
  },
  {
    family: "boltzmann-machine",
    path: "/networks/boltzmann-machine",
    label: "Boltzmann Machine",
    status: "placeholder",
    primaryTasks: ["generation", "pattern-reconstruction", "unsupervised-feature-learning"],
    summary: "Fully connected stochastic energy model for sampling and representation learning.",
  },
  {
    family: "rbm",
    path: "/networks/restricted-boltzmann-machine",
    label: "Restricted Boltzmann Machine",
    status: "implemented",
    primaryTasks: ["generation", "pattern-reconstruction", "unsupervised-feature-learning"],
    summary: "Bipartite stochastic latent-variable model for reconstruction and generative sampling.",
  },
] as const;

export function formatPrimaryTasks(tasks: readonly ModelPrimaryTask[]): string {
  return tasks
    .map((task) => {
      switch (task) {
        case "pattern-reconstruction":
          return "pattern reconstruction";
        case "associative-retrieval":
          return "associative retrieval";
        case "generation":
          return "generation";
        case "unsupervised-feature-learning":
          return "unsupervised feature learning";
        case "classification-probe":
          return "classification probe";
      }
    })
    .join(", ");
}

export function getModelCatalogEntry(family: ModelFamily): ModelCatalogEntry {
  const entry = MODEL_CATALOG.find((candidate) => candidate.family === family);
  if (!entry) {
    throw new Error(`Unknown model catalog family: ${family}`);
  }
  return entry;
}
