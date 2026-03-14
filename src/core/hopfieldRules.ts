export type LearningRule = "hebbian" | "pseudoinverse" | "storkey" | "krauth-mezard" | "unlearning";
export type ConvergenceRule = "async-random" | "synchronous" | "stochastic";

export interface HebbianLearningConfig {
  rule: "hebbian";
}

export interface PseudoinverseLearningConfig {
  rule: "pseudoinverse";
}

export interface StorkeyLearningConfig {
  rule: "storkey";
}

export interface KrauthMezardLearningConfig {
  rule: "krauth-mezard";
  kappa: number;
  epsilon: number;
  maxEpochs: number;
}

export interface UnlearningLearningConfig {
  rule: "unlearning";
  epsilon: number;
  steps: number;
}

export type LearningRuleConfig =
  | HebbianLearningConfig
  | PseudoinverseLearningConfig
  | StorkeyLearningConfig
  | KrauthMezardLearningConfig
  | UnlearningLearningConfig;

export interface AsyncRandomConvergenceConfig {
  rule: "async-random";
}

export interface SynchronousConvergenceConfig {
  rule: "synchronous";
}

export interface StochasticConvergenceConfig {
  rule: "stochastic";
  temperature: number;
}

export type ConvergenceRuleConfig =
  | AsyncRandomConvergenceConfig
  | SynchronousConvergenceConfig
  | StochasticConvergenceConfig;

export interface RuleOption<T extends string> {
  value: T;
  label: string;
  kind: "one-step" | "iterative";
}

export const LEARNING_RULE_OPTIONS: RuleOption<LearningRule>[] = [
  { value: "hebbian", label: "Hebbian", kind: "one-step" },
  { value: "pseudoinverse", label: "Pseudoinverse", kind: "one-step" },
  { value: "storkey", label: "Storkey", kind: "iterative" },
  { value: "krauth-mezard", label: "Krauth-Mezard", kind: "iterative" },
  { value: "unlearning", label: "Unlearning", kind: "iterative" },
];

export const CONVERGENCE_RULE_OPTIONS: RuleOption<ConvergenceRule>[] = [
  { value: "async-random", label: "Asynchronous", kind: "iterative" },
  { value: "synchronous", label: "Synchronous", kind: "iterative" },
  { value: "stochastic", label: "Stochastic", kind: "iterative" },
];

export function createDefaultLearningRuleConfig(rule: LearningRule): LearningRuleConfig {
  switch (rule) {
    case "hebbian":
      return { rule };
    case "pseudoinverse":
      return { rule };
    case "storkey":
      return { rule };
    case "krauth-mezard":
      return {
        rule,
        kappa: 0,
        epsilon: 1,
        maxEpochs: 40,
      };
    case "unlearning":
      return {
        rule,
        epsilon: 0.2,
        steps: 24,
      };
  }

  throw new Error(`Unsupported learning rule: ${rule}`);
}

export function createDefaultConvergenceRuleConfig(rule: ConvergenceRule): ConvergenceRuleConfig {
  switch (rule) {
    case "async-random":
      return { rule };
    case "synchronous":
      return { rule };
    case "stochastic":
      return {
        rule,
        temperature: 0.8,
      };
  }

  throw new Error(`Unsupported convergence rule: ${rule}`);
}

export function getLearningRuleLabel(rule: LearningRule): string {
  return LEARNING_RULE_OPTIONS.find((option) => option.value === rule)?.label ?? rule;
}

export function getConvergenceRuleLabel(rule: ConvergenceRule): string {
  return CONVERGENCE_RULE_OPTIONS.find((option) => option.value === rule)?.label ?? rule;
}
