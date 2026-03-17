import { type ReactNode, useEffect, useRef, useState } from "react";
import { CircleHelp, Pause, Play, RotateCcw, SkipForward, X } from "lucide-react";
import { BlockMath, InlineMath } from "react-katex";

import type { HopfieldSnapshot } from "../core/hopfield";
import type { ConvergenceRule, ConvergenceRuleConfig, LearningRule, LearningRuleConfig } from "../core/hopfieldRules";
import {
  CONVERGENCE_RULE_OPTIONS,
  LEARNING_RULE_OPTIONS,
  createDefaultConvergenceRuleConfig,
  createDefaultLearningRuleConfig,
  getConvergenceRuleLabel,
  getLearningRuleLabel,
} from "../core/hopfieldRules";
import {
  clonePattern,
  createBlankPattern,
  getDefaultPatternSetId,
  loadPatternSetById,
  PATTERN_SET_OPTIONS,
  PATTERN_SIDE,
  type PatternSetDefinition,
} from "../core/patternSets";
import { formatPrimaryTasks, getModelCatalogEntry } from "../core/modelCatalog";
import type { WorkerRequest, WorkerResponse } from "../core/workerProtocol";
import { EnergyPlot } from "../features/hopfield/EnergyPlot";
import { ValueGridHeatmap, WeightHeatmap } from "../features/hopfield/HeatmapCanvas";
import { PatternCanvas } from "../features/hopfield/PatternCanvas";
import { PatternGallery } from "../features/hopfield/PatternGallery";
import { DatasetDialog } from "../features/common/DatasetDialog";

type HelpPanelKey = "dataset" | "learning" | "convergence" | null;
interface HelpContent {
  title: string;
  summary: string;
  formula: string;
  params: string[];
  symbols: string[];
  explanation: string[];
  notes: string[];
}

function createSnapshot(state: Int8Array): HopfieldSnapshot {
  return {
    state,
    energy: 0,
    step: 0,
    changedCount: 0,
    matchedPatternIndex: -1,
    converged: false,
  };
}

function renderTextWithMath(text: string): ReactNode {
  const segments = text.split(/(\$[^$]+\$)/g).filter(Boolean);
  return segments.map((segment, index) => {
    if (segment.startsWith("$") && segment.endsWith("$")) {
      return <InlineMath key={`${segment}-${index}`}>{segment.slice(1, -1)}</InlineMath>;
    }
    return <span key={`${segment}-${index}`}>{segment}</span>;
  });
}

function getLearningHelpContent(config: LearningRuleConfig): HelpContent {
  switch (config.rule) {
    case "hebbian":
      return {
        title: "Hebbian Learning",
        summary: "Single-pass outer-product learning over all stored patterns.",
        formula: String.raw`W = \frac{1}{N}\sum_{\mu}\xi^{\mu}(\xi^{\mu})^{T}, \qquad \operatorname{diag}(W)=0`,
        params: ["No tunable parameters."],
        symbols: [
          "$W$: the learned symmetric weight matrix of the Hopfield network.",
          "$N$: the total number of neurons in the network.",
          "$\\sum_{\\mu}$: sum over every stored pattern index $\\mu$.",
          "$\\xi^{\\mu}$: stored pattern number $\\mu$, written as a bipolar vector.",
          "$(\\xi^{\\mu})^T$: transpose of that pattern vector, used to form the outer product.",
          "$\\operatorname{diag}(W)=0$: all self-connections on the diagonal are forced to zero.",
        ],
        explanation: [
          "Hebbian learning adds one outer product for each stored memory, then normalizes the result by the network size. The outer product reinforces pairs of neurons that tend to be active together and weakens pairs that usually disagree.",
          "Because the rule is a single closed-form pass, it is the simplest and fastest storage rule in the UI. The tradeoff is limited storage capacity and higher sensitivity to correlated memories.",
        ],
        notes: [
          "Capacity is about $0.138N$ stored patterns.",
          String.raw`Patterns should be bipolar in $\{-1, +1\}$.`,
          "Fastest learning rule in this UI.",
        ],
      };
    case "pseudoinverse":
      return {
        title: "Pseudoinverse Learning",
        summary: "Projection-style learning using the Moore-Penrose pseudoinverse.",
        formula: String.raw`W = X\,\operatorname{pinv}(X), \qquad \operatorname{diag}(W)=0`,
        params: ["No tunable parameters."],
        symbols: [
          "$W$: the learned symmetric weight matrix.",
          "$X$: the pattern matrix whose columns are the stored memories.",
          "$\\operatorname{pinv}(X)$: the Moore-Penrose pseudoinverse of $X$.",
          "$\\operatorname{diag}(W)=0$: diagonal entries are cleared so neurons do not feed back into themselves.",
          "$P$: number of stored patterns.",
          "$N$: number of neurons.",
        ],
        explanation: [
          "This rule builds a projection operator that tries to map the state space back onto the subspace spanned by the stored patterns. When the patterns are sufficiently independent, recall can be much cleaner than with plain Hebbian storage.",
          "The cost is that the full memory set has to be processed together. Adding or removing one memory means recomputing the whole matrix rather than updating incrementally.",
        ],
        notes: [
          String.raw`Best when $P \ll N$ and the stored patterns are linearly independent.`,
          "Can approach $N$ exact memories under favorable conditions.",
          "Not incremental: changing the memory set requires full recomputation.",
        ],
      };
    case "storkey":
      return {
        title: "Storkey Learning",
        summary: "Incremental local learning that subtracts interference through local fields.",
        formula: String.raw`W^{\nu}=W^{\nu-1}+\frac{1}{N}\xi^{\nu}(\xi^{\nu})^{T}-\frac{1}{N}\xi^{\nu}(h^{\nu})^{T}-\frac{1}{N}h^{\nu}(\xi^{\nu})^{T}`,
        params: ["No tunable parameters."],
        symbols: [
          "$W^{\\nu-1}$: the weight matrix before storing pattern $\\nu$.",
          "$W^{\\nu}$: the updated weight matrix after storing pattern $\\nu$.",
          "$N$: the number of neurons.",
          "$\\xi^{\\nu}$: the current stored pattern being added.",
          "$h^{\\nu}$: the local-field vector induced by the previous weights on pattern $\\nu$.",
          "$(^T)$: transpose, used to form outer products between vectors.",
        ],
        explanation: [
          "Storkey starts with the Hebbian outer-product term but then subtracts two correction terms based on the local field. Those corrections reduce the amount of crosstalk that a newly added memory introduces into already stored structure.",
          "Because the update is local and incremental, the rule can store memories one by one while usually handling correlated memories more gracefully than plain Hebbian learning.",
        ],
        notes: [
          String.raw`Local field: $h_i^{\nu}=\sum_{k\neq i}W_{ik}^{\nu-1}\xi_k^{\nu}$.`,
          "Incremental: memories can be added one by one.",
          "Usually handles correlated memories better than plain Hebbian learning.",
        ],
      };
    case "krauth-mezard":
      return {
        title: "Krauth-Mezard Learning",
        summary: "Perceptron-style stability optimization with iterative row updates.",
        formula: String.raw`\text{stability}_i^{\mu}=\xi_i^{\mu}\sum_{j\neq i}W_{ij}\xi_j^{\mu}, \qquad
\text{if } \text{stability}_i^{\mu}\le \kappa,\;
W_{ij}\leftarrow W_{ij}+\frac{\epsilon}{N}\xi_i^{\mu}\xi_j^{\mu}`,
        params: [
          `Current $\\kappa$ = ${learningConfigToNumber(config, "kappa").toFixed(2)}`,
          `Current $\\epsilon$ = ${learningConfigToNumber(config, "epsilon").toFixed(2)}`,
          `Current max epochs = ${learningConfigToNumber(config, "maxEpochs").toFixed(0)}`,
        ],
        symbols: [
          "$\\mathrm{stability}_i^{\\mu}$: how strongly neuron $i$ supports pattern $\\mu$ under the current weights.",
          "$\\xi_i^{\\mu}$: the target value of neuron $i$ inside stored pattern $\\mu$.",
          "$W_{ij}$: connection from neuron $j$ to neuron $i$.",
          "$\\kappa$: the requested stability margin. Larger values demand more separation from the decision boundary.",
          "$\\epsilon$: the learning-rate scale applied when a constraint is violated.",
          "$N$: the number of neurons, used to normalize the update size.",
        ],
        explanation: [
          "This rule treats each neuron row like a perceptron and pushes the stored patterns away from the decision boundary until the requested stability margin is achieved. If a pattern is not stable enough, the corresponding row receives a corrective update.",
          "It is slower than the closed-form rules because it iterates over patterns, neurons, and epochs. The payoff is higher storage capacity and more explicit control over the margin through $\\kappa$.",
        ],
        notes: [
          "Best capacity among the implemented learning rules.",
          "Slowest rule in this UI because it iterates over neurons and memories repeatedly.",
          "The matrix is symmetrized after each epoch.",
        ],
      };
    case "unlearning":
      return {
        title: "Unlearning",
        summary: "Hebbian initialization followed by anti-Hebbian suppression of spurious attractors.",
        formula: String.raw`W \leftarrow W - \frac{\epsilon}{N}s^{*}(s^{*})^{T}`,
        params: [
          `Current $\\epsilon$ = ${learningConfigToNumber(config, "epsilon").toFixed(2)}`,
          `Current unlearning steps = ${learningConfigToNumber(config, "steps").toFixed(0)}`,
        ],
        symbols: [
          "$W$: the current weight matrix being refined.",
          "$\\leftarrow$: assignment, meaning the matrix is replaced by the new value.",
          "$\\epsilon$: the unlearning rate controlling how strong each anti-Hebbian update is.",
          "$N$: the number of neurons, used for normalization.",
          "$s^{*}$: a converged attractor reached from a random initial condition.",
          "$(s^{*})^T$: transpose of that attractor vector, producing an outer product with $s^{*}$.",
        ],
        explanation: [
          "Unlearning starts from a standard Hebbian matrix and then repeatedly searches for attractors reached from random initial states. When the network settles into one of those states, the rule slightly weakens that attractor through an anti-Hebbian outer product.",
          "The intent is to reduce the strength of spurious basins without erasing the intended memories. That makes this rule slower, because training includes full retrieval runs inside the learning loop.",
        ],
        notes: [
          "Biologically motivated by the Hopfield-Feinstein-Palmer unlearning idea.",
          "Training runs retrieval internally, so it is much slower than one-shot rules.",
          "Useful for reducing spurious basins beyond plain Hebbian storage.",
        ],
      };
  }
}

function getConvergenceHelpContent(config: ConvergenceRuleConfig, speed: number, maxPlaybackSteps: number): HelpContent {
  switch (config.rule) {
    case "async-random":
      return {
        title: "Asynchronous Convergence",
        summary: "Original Hopfield retrieval with one neuron update at a time.",
        formula: String.raw`s_i \leftarrow \operatorname{sign}\!\left(\sum_j W_{ij}s_j\right)`,
        params: [`Playback speed = ${speedLabelValue(speed)}`, `Playback sweeps = ${maxPlaybackSteps}`],
        symbols: [
          "$s_i$: current state of neuron $i$ after the update.",
          "$\\leftarrow$: assignment, meaning neuron $i$ is replaced by the new value.",
          "$\\operatorname{sign}(\\cdot)$: sign function returning the bipolar state $-1$ or $+1$.",
          "$\\sum_j$: sum over all presynaptic neurons $j$.",
          "$W_{ij}$: connection weight from neuron $j$ to neuron $i$.",
          "$s_j$: current state of neuron $j$ before neuron $i$ is updated.",
        ],
        explanation: [
          "Asynchronous retrieval updates one neuron at a time using the current network state. Because each update immediately changes the state seen by subsequent neurons, the retrieval trajectory follows the original Hopfield descent dynamics.",
          "In this UI, one visible step corresponds to one random sweep through the neurons. That makes the chart readable while preserving the monotonic energy behavior of asynchronous updates.",
        ],
        notes: [
          "Each visible step in this UI is one random sweep over all $784$ neurons.",
          String.raw`Energy $E=-0.5\, s^{T}Ws$ is guaranteed to be non-increasing.`,
          "Safest default when you want stable convergence behavior.",
        ],
      };
    case "synchronous":
      return {
        title: "Synchronous Convergence",
        summary: "All neurons update at once on each retrieval iteration.",
        formula: String.raw`s \leftarrow \operatorname{sign}(Ws)`,
        params: [`Playback speed = ${speedLabelValue(speed)}`, `Playback sweeps = ${maxPlaybackSteps}`],
        symbols: [
          "$s$: the full network state vector.",
          "$W$: the learned Hopfield weight matrix.",
          "$Ws$: matrix-vector product giving the local-field values for all neurons at once.",
          "$\\operatorname{sign}(\\cdot)$: applied elementwise to convert those fields back to bipolar neuron states.",
          "$\\leftarrow$: assignment, replacing the whole state in one step.",
        ],
        explanation: [
          "Synchronous retrieval computes all local fields from the old state and then updates every neuron simultaneously. That makes each iteration conceptually simple and fast to evaluate.",
          "The tradeoff is dynamical: because no neuron sees the newly updated state of another neuron during the same step, the network can bounce between states instead of settling into a fixed point.",
        ],
        notes: [
          "Fast per iteration.",
          "Can enter period-2 oscillations instead of reaching a fixed point.",
          "Useful to contrast with asynchronous retrieval dynamics.",
        ],
      };
    case "stochastic":
      return {
        title: "Stochastic Convergence",
        summary: "Boltzmann-style retrieval with probabilistic flips controlled by temperature.",
        formula: String.raw`P(s_i=+1)=\frac{1}{1+\exp(-2h_i/T)}, \qquad h_i=\sum_j W_{ij}s_j`,
        params: [
          `Current temperature $T$ = ${config.temperature.toFixed(2)}`,
          `Playback speed = ${speedLabelValue(speed)}`,
          `Playback sweeps = ${maxPlaybackSteps}`,
        ],
        symbols: [
          "$P(s_i=+1)$: probability that neuron $i$ becomes active.",
          "$s_i$: state of neuron $i$ after the stochastic update.",
          "$h_i$: local field acting on neuron $i$.",
          "$W_{ij}$: connection weight from neuron $j$ to neuron $i$.",
          "$s_j$: current state of neuron $j$.",
          "$T$: temperature controlling how random the update is.",
        ],
        explanation: [
          "Instead of taking a hard sign, stochastic retrieval converts the local field into a probability through a logistic function. Large positive fields strongly favor $+1$, large negative fields favor $-1$, and ambiguous fields become genuinely probabilistic.",
          "Temperature sets how sharp that decision is. Low $T$ behaves almost deterministically, while higher $T$ adds enough randomness to escape shallow or spurious attractors.",
        ],
        notes: [
          "At low temperature it approaches deterministic asynchronous updates.",
          "At higher temperature it can escape shallow spurious attractors.",
          "Each visible step in this UI performs one stochastic sweep over the neurons.",
        ],
      };
  }
}

function learningConfigToNumber<T extends "kappa" | "epsilon" | "maxEpochs" | "steps">(
  config: LearningRuleConfig,
  key: T,
): number {
  return key in config ? Number(config[key as keyof LearningRuleConfig]) : 0;
}

function speedLabelValue(speed: number): string {
  return `${speed} sweeps/s`;
}

function shuffleIndices(size: number): Uint16Array {
  const indices = new Uint16Array(size);
  for (let index = 0; index < size; index += 1) {
    indices[index] = index;
  }

  for (let index = size - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    const current = indices[index];
    indices[index] = indices[swapIndex];
    indices[swapIndex] = current;
  }

  return indices;
}

function applyPatternNoise(pattern: Int8Array, corruptionPercent: number, obfuscationPercent: number): Int8Array {
  const next = pattern.slice();
  const total = next.length;

  const corruptionCount = Math.round((total * corruptionPercent) / 100);
  const obfuscationCount = Math.round((total * obfuscationPercent) / 100);

  if (corruptionCount > 0) {
    const corruptionIndices = shuffleIndices(total);
    for (let index = 0; index < corruptionCount; index += 1) {
      const target = corruptionIndices[index];
      next[target] = next[target] === 1 ? -1 : 1;
    }
  }

  if (obfuscationCount > 0) {
    const obfuscationIndices = shuffleIndices(total);
    for (let index = 0; index < obfuscationCount; index += 1) {
      next[obfuscationIndices[index]] = 0;
    }
  }

  return next;
}

function HelpDialog({
  content,
  onClose,
}: {
  content: HelpContent;
  onClose: () => void;
}) {
  return (
    <div className="modal-backdrop" onClick={onClose} role="presentation">
      <section
        className="modal-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="help-dialog-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-header">
          <div>
            <h2 id="help-dialog-title">{content.title}</h2>
            <p>{content.summary}</p>
          </div>
          <button type="button" className="modal-close-btn" onClick={onClose} aria-label="Close help dialog">
            <X size={16} />
          </button>
        </div>
        <div className="modal-body">
          <section className="modal-section">
            <span className="modal-section-label">Math</span>
            <div className="formula-block">
              <BlockMath math={content.formula} />
            </div>
          </section>
          <section className="modal-section">
            <span className="modal-section-label">Parameters</span>
            <ul className="modal-list">
              {content.params.map((item) => (
                <li key={item}>{renderTextWithMath(item)}</li>
              ))}
            </ul>
          </section>
          <section className="modal-section">
            <span className="modal-section-label">Symbol guide</span>
            <ul className="modal-list modal-list--symbols">
              {content.symbols.map((item) => (
                <li key={item}>{renderTextWithMath(item)}</li>
              ))}
            </ul>
          </section>
          <section className="modal-section modal-section--notes">
            <span className="modal-section-label">Interpretation</span>
            <div className="modal-prose">
              {content.explanation.map((item) => (
                <p key={item}>{renderTextWithMath(item)}</p>
              ))}
            </div>
          </section>
          <section className="modal-section">
            <span className="modal-section-label">Notes</span>
            <ul className="modal-list modal-list--notes">
              {content.notes.map((item) => (
                <li key={item}>{renderTextWithMath(item)}</li>
              ))}
            </ul>
          </section>
        </div>
      </section>
    </div>
  );
}

export default function HopfieldNetworkPage() {
  const modelEntry = getModelCatalogEntry("hopfield");
  const defaultPatternSetId = getDefaultPatternSetId();
  const [patternSetId, setPatternSetId] = useState(defaultPatternSetId);
  const [patternSet, setPatternSet] = useState<PatternSetDefinition>({
    id: defaultPatternSetId,
    name: PATTERN_SET_OPTIONS[0].name,
    description: "Loading pattern set...",
    labels: [],
    patterns: [],
  });
  const [learningConfig, setLearningConfig] = useState<LearningRuleConfig>(() => createDefaultLearningRuleConfig("hebbian"));
  const [convergenceConfig, setConvergenceConfig] = useState<ConvergenceRuleConfig>(() =>
    createDefaultConvergenceRuleConfig("async-random"),
  );
  const [queryPattern, setQueryPattern] = useState<Int8Array>(() => createBlankPattern());
  const [snapshot, setSnapshot] = useState<HopfieldSnapshot>(() => createSnapshot(createBlankPattern()));
  const [weights, setWeights] = useState<Float32Array>(() => new Float32Array(PATTERN_SIDE * PATTERN_SIDE * PATTERN_SIDE * PATTERN_SIDE));
  const [maxWeightAbs, setMaxWeightAbs] = useState(1);
  const [energyHistory, setEnergyHistory] = useState<number[]>([]);
  const [speed, setSpeed] = useState(8);
  const [maxPlaybackSteps, setMaxPlaybackSteps] = useState(80);
  const [corruptionLevel, setCorruptionLevel] = useState(0);
  const [obfuscationLevel, setObfuscationLevel] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [isPatternSetLoading, setIsPatternSetLoading] = useState(true);
  const [workerError, setWorkerError] = useState<string | null>(null);
  const [openHelpPanel, setOpenHelpPanel] = useState<HelpPanelKey>(null);
  const isFirstLearningRenderRef = useRef(true);

  const hasAppliedQueryRef = useRef(false);
  const patternSetIdRef = useRef(patternSetId);
  const workerRef = useRef<Worker | null>(null);

  useEffect(() => {
    const worker = new Worker(new URL("../workers/hopfield.worker.ts", import.meta.url), { type: "module" });
    workerRef.current = worker;

    const handleMessage = (event: MessageEvent<WorkerResponse>) => {
      const message = event.data;

      if (message.type === "ready") {
        setWeights(message.weights);
        setMaxWeightAbs(message.maxWeightAbs);
        setSnapshot(message.snapshot);
        setEnergyHistory([message.snapshot.energy]);
        setIsReady(true);
        setIsPlaying(false);
        setWorkerError(null);
        return;
      }

      if (message.type === "snapshot") {
        setSnapshot(message.snapshot);
        setEnergyHistory((previous) => {
          if (message.snapshot.step === 0) {
            return [message.snapshot.energy];
          }
          return [...previous, message.snapshot.energy];
        });
        if (message.snapshot.converged) {
          setIsPlaying(false);
        }
        return;
      }

      if (message.type === "paused") {
        setIsPlaying(false);
        return;
      }

      if (message.type === "error") {
        setWorkerError(message.message);
        setIsPlaying(false);
      }
    };

    worker.addEventListener("message", handleMessage);
    return () => {
      worker.removeEventListener("message", handleMessage);
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!openHelpPanel) {
      return;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpenHelpPanel(null);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [openHelpPanel]);

  useEffect(() => {
    patternSetIdRef.current = patternSetId;
  }, [patternSetId]);

  useEffect(() => {
    const worker = workerRef.current;
    if (!worker) {
      return;
    }

    let cancelled = false;
    setQueryPattern(createBlankPattern());
    setEnergyHistory([]);
    setSnapshot(createSnapshot(createBlankPattern()));
    setIsReady(false);
    setIsPatternSetLoading(true);
    setWorkerError(null);
    hasAppliedQueryRef.current = false;

    void loadPatternSetById(patternSetId)
      .then((nextPatternSet) => {
        if (cancelled) {
          return;
        }

        setPatternSet(nextPatternSet);
        worker.postMessage({
          type: "initialize",
          patternSetId,
          learningConfig,
        } satisfies WorkerRequest);
      })
      .catch((error: unknown) => {
        if (cancelled) {
          return;
        }
        setWorkerError(error instanceof Error ? error.message : "Failed to load pattern set.");
      })
      .finally(() => {
        if (!cancelled) {
          setIsPatternSetLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [patternSetId]);

  useEffect(() => {
    const worker = workerRef.current;
    if (!worker) {
      return;
    }

    if (isFirstLearningRenderRef.current) {
      isFirstLearningRenderRef.current = false;
      return;
    }

    setEnergyHistory([]);
    setSnapshot(createSnapshot(createBlankPattern()));
    setIsReady(false);
    setWorkerError(null);
    hasAppliedQueryRef.current = false;

    worker.postMessage({
      type: "initialize",
      patternSetId: patternSetIdRef.current,
      learningConfig,
    } satisfies WorkerRequest);
  }, [learningConfig]);

  function setQueryOnWorker(): void {
    const worker = workerRef.current;
    if (!worker) {
      return;
    }
    hasAppliedQueryRef.current = true;
    const outgoing = queryPattern.slice();
    worker.postMessage({ type: "setQuery", pattern: outgoing } satisfies WorkerRequest, [outgoing.buffer]);
  }

  function applyQuery(): void {
    if (!isReady) {
      return;
    }

    setQueryOnWorker();
    setIsPlaying(true);
    workerRef.current?.postMessage({
      type: "play",
      convergenceConfig,
      intervalMs: Math.max(40, Math.round(1000 / speed)),
      maxSteps: Math.max(1, maxPlaybackSteps),
    } satisfies WorkerRequest);
  }

  function handlePlay(): void {
    const worker = workerRef.current;
    if (!worker) {
      return;
    }
    if (!hasAppliedQueryRef.current) {
      setQueryOnWorker();
    }
    setIsPlaying(true);
    worker.postMessage({
      type: "play",
      convergenceConfig,
      intervalMs: Math.max(40, Math.round(1000 / speed)),
      maxSteps: Math.max(1, maxPlaybackSteps),
    } satisfies WorkerRequest);
  }

  function handlePause(): void {
    workerRef.current?.postMessage({ type: "pause" } satisfies WorkerRequest);
  }

  function handleStep(): void {
    if (!hasAppliedQueryRef.current) {
      applyQuery();
      return;
    }
    workerRef.current?.postMessage({
      type: "step",
      convergenceConfig,
    } satisfies WorkerRequest);
  }

  function handleReset(): void {
    workerRef.current?.postMessage({ type: "reset" } satisfies WorkerRequest);
  }

  function handleClear(): void {
    const worker = workerRef.current;
    if (!worker) {
      return;
    }
    const blank = createBlankPattern();
    setQueryPattern(blank);
    hasAppliedQueryRef.current = false;
    const outgoing = blank.slice();
    worker.postMessage({ type: "setQuery", pattern: outgoing } satisfies WorkerRequest, [outgoing.buffer]);
  }

  function handleLoadPattern(index: number): void {
    const next = applyPatternNoise(clonePattern(patternSet.patterns[index]), corruptionLevel, obfuscationLevel);
    setQueryPattern(next);
    hasAppliedQueryRef.current = false;
  }

  function handlePatternChange(nextPattern: Int8Array): void {
    setQueryPattern(nextPattern);
    hasAppliedQueryRef.current = false;
  }

  function handleLearningRuleChange(rule: LearningRule): void {
    setLearningConfig(createDefaultLearningRuleConfig(rule));
  }

  function renderLearningFields() {
    if (learningConfig.rule === "krauth-mezard") {
      return (
        <div className="rule-config-grid">
          <label className="field compact-field">
            <span>Kappa</span>
            <input
              type="number"
              min="0"
              max="2"
              step="0.1"
              value={learningConfig.kappa}
              onChange={(event) =>
                setLearningConfig({
                  ...learningConfig,
                  kappa: Number(event.target.value),
                })
              }
            />
          </label>
          <label className="field compact-field">
            <span>Epsilon</span>
            <input
              type="number"
              min="0.1"
              max="2"
              step="0.1"
              value={learningConfig.epsilon}
              onChange={(event) =>
                setLearningConfig({
                  ...learningConfig,
                  epsilon: Number(event.target.value),
                })
              }
            />
          </label>
          <label className="field compact-field">
            <span>Max epochs</span>
            <input
              type="number"
              min="1"
              max="200"
              step="1"
              value={learningConfig.maxEpochs}
              onChange={(event) =>
                setLearningConfig({
                  ...learningConfig,
                  maxEpochs: Number(event.target.value),
                })
              }
            />
          </label>
        </div>
      );
    }

    if (learningConfig.rule === "unlearning") {
      return (
        <div className="rule-config-grid">
          <label className="field compact-field">
            <span>Epsilon</span>
            <input
              type="number"
              min="0.05"
              max="1"
              step="0.05"
              value={learningConfig.epsilon}
              onChange={(event) =>
                setLearningConfig({
                  ...learningConfig,
                  epsilon: Number(event.target.value),
                })
              }
            />
          </label>
          <label className="field compact-field">
            <span>Unlearning steps</span>
            <input
              type="number"
              min="1"
              max="120"
              step="1"
              value={learningConfig.steps}
              onChange={(event) =>
                setLearningConfig({
                  ...learningConfig,
                  steps: Number(event.target.value),
                })
              }
            />
          </label>
        </div>
      );
    }

    return null;
  }

  function renderConvergenceFields() {
    return (
      <div className="rule-config-grid">
        {convergenceConfig.rule === "stochastic" ? (
          <label className="field compact-field">
            <span>Temperature</span>
            <input
              type="number"
              min="0.05"
              max="3"
              step="0.05"
              value={convergenceConfig.temperature}
              onChange={(event) =>
                setConvergenceConfig({
                  ...convergenceConfig,
                  temperature: Number(event.target.value),
                })
              }
            />
          </label>
        ) : null}
        <label className="field compact-field">
          <span>Speed</span>
          <input type="range" min="1" max="20" value={speed} onChange={(event) => setSpeed(Number(event.target.value))} />
          <strong className="range-value">{speed} sweeps/s</strong>
        </label>
        <label className="field compact-field">
          <span>Sweeps</span>
          <input
            type="number"
            min="1"
            max="240"
            step="1"
            value={maxPlaybackSteps}
            onChange={(event) => setMaxPlaybackSteps(Number(event.target.value))}
          />
        </label>
      </div>
    );
  }

  const learningHelpContent = getLearningHelpContent(learningConfig);
  const convergenceHelpContent = getConvergenceHelpContent(convergenceConfig, speed, maxPlaybackSteps);

  return (
    <div className="page-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Wasm-backed worker runtime</p>
          <h1>Hopfield Network</h1>
          <p className="hero-copy">
            Typed-array Hopfield core, worker-driven convergence, editable 28x28 input, live neuron-state heatmap,
            and a full 784x784 connection matrix rendered in the client.
          </p>
          <p className="hero-task">Primary task: {formatPrimaryTasks(modelEntry.primaryTasks)}</p>
        </div>
        <div className="hero-stats">
          <div className="stat-card">
            <span>Neurons</span>
            <strong>784</strong>
          </div>
          <div className="stat-card">
            <span>Weights</span>
            <strong>{weights.length.toLocaleString()}</strong>
          </div>
          <div className="stat-card">
            <span>Status</span>
            <strong>{snapshot.converged ? "stable" : isPlaying ? "running" : "ready"}</strong>
          </div>
        </div>
      </header>

      {workerError ? <div className="error-banner">{workerError}</div> : null}

      <section className="panel architecture-bar">
        <div className="control-strip-group">
          <div className="control-strip-header">
            <span className="control-strip-title">Dataset</span>
            <button
              type="button"
              className="help-btn"
              aria-expanded={openHelpPanel === "dataset"}
              onClick={() => setOpenHelpPanel(openHelpPanel === "dataset" ? null : "dataset")}
              title="Dataset help"
            >
              <CircleHelp size={15} />
            </button>
          </div>
          <label className="field compact-field">
            <span>Dataset</span>
            <select value={patternSetId} onChange={(event) => setPatternSetId(event.target.value)}>
              {PATTERN_SET_OPTIONS.map((entry) => (
                <option key={entry.id} value={entry.id}>
                  {entry.name}
                </option>
              ))}
            </select>
          </label>
          <p className="control-strip-note">{isPatternSetLoading ? "Loading selected pattern set..." : "Binary patterns: active = +1, inactive = -1."}</p>
        </div>

        <div className="control-strip-group">
          <div className="control-strip-header">
            <span className="control-strip-title">Learning control</span>
            <button
              type="button"
              className="help-btn"
              aria-expanded={openHelpPanel === "learning"}
              onClick={() => setOpenHelpPanel(openHelpPanel === "learning" ? null : "learning")}
              title="Learning rule help"
            >
              <CircleHelp size={15} />
            </button>
          </div>
          <label className="field compact-field">
            <span>Learning rule</span>
            <select value={learningConfig.rule} onChange={(event) => handleLearningRuleChange(event.target.value as LearningRule)}>
              {LEARNING_RULE_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          {renderLearningFields()}
        </div>

        <div className="control-strip-group control-strip-group--wide">
          <div className="control-strip-header">
            <span className="control-strip-title">Convergence control</span>
            <button
              type="button"
              className="help-btn"
              aria-expanded={openHelpPanel === "convergence"}
              onClick={() => setOpenHelpPanel(openHelpPanel === "convergence" ? null : "convergence")}
              title="Convergence rule help"
            >
              <CircleHelp size={15} />
            </button>
          </div>
          <label className="field compact-field">
            <span>Convergence rule</span>
            <select
              value={convergenceConfig.rule}
              onChange={(event) => setConvergenceConfig(createDefaultConvergenceRuleConfig(event.target.value as ConvergenceRule))}
            >
              {CONVERGENCE_RULE_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <div className="convergence-strip">
            <div className="convergence-strip-main">
              <div className="convergence-fields">{renderConvergenceFields()}</div>
              <div className="control-actions">
                  <button type="button" className="icon-btn primary" onClick={applyQuery} disabled={!isReady || isPatternSetLoading} title="Apply query">
                  <SkipForward size={14} />
                  <span>Apply</span>
                </button>
                {isPlaying ? (
                  <button type="button" className="icon-btn" onClick={handlePause} title="Pause">
                    <Pause size={14} />
                    <span>Pause</span>
                  </button>
                ) : (
                  <button type="button" className="icon-btn" onClick={handlePlay} disabled={!isReady || isPatternSetLoading} title="Play">
                    <Play size={14} />
                    <span>Play</span>
                  </button>
                )}
                <button type="button" className="icon-btn" onClick={handleStep} disabled={!isReady || isPlaying || isPatternSetLoading} title="Step">
                  <SkipForward size={14} />
                  <span>Step</span>
                </button>
                <button type="button" className="icon-btn" onClick={handleReset} disabled={!isReady || isPatternSetLoading} title="Reset">
                  <RotateCcw size={14} />
                  <span>Reset</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </section>

      <div className="dashboard-grid dashboard-grid--two-column">
        <div className="main-column">
          <section className="center-row">
            <section className="panel input-panel">
              <div className="panel-header">
                <h3>Query</h3>
                <p>Pick a stored pattern, degrade it if needed, then edit directly on the 28x28 lattice before running retrieval.</p>
              </div>
              <div className="query-workbench">
                <div className="query-toolbar">
                  <div className="field compact-field">
                    <span>Examples</span>
                    <div className="pattern-picker pattern-picker--compact">
                      {patternSet.labels.map((label, index) => (
                        <button key={label} type="button" onClick={() => handleLoadPattern(index)} title={`Load ${label}`}>
                          {label}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="query-toolbar--row">
                    <label className="field compact-field">
                      <span>Corruption</span>
                      <input type="range" min="0" max="100" value={corruptionLevel} onChange={(event) => setCorruptionLevel(Number(event.target.value))} title="Randomly flip bipolar values before loading into the query editor" />
                      <strong className="range-value">{corruptionLevel}%</strong>
                    </label>
                    <label className="field compact-field">
                      <span>Obfuscation</span>
                      <input type="range" min="0" max="100" value={obfuscationLevel} onChange={(event) => setObfuscationLevel(Number(event.target.value))} title="Set part of the loaded pattern to zero before retrieval" />
                      <strong className="range-value">{obfuscationLevel}%</strong>
                    </label>
                  </div>
                </div>
                <div className="input-grid">
                <PatternCanvas pattern={queryPattern} onChange={handlePatternChange} />
                <div className="query-actions">
                  <button type="button" onClick={handleClear} title="Clear the query editor">
                    clear
                  </button>
                </div>
                </div>
              </div>
            </section>

            <div className="hopfield-connectome">
              <WeightHeatmap
                title="Connectome heatmap"
                data={weights}
                side={PATTERN_SIDE * PATTERN_SIDE}
                maxAbs={maxWeightAbs}
                caption={`${getLearningRuleLabel(learningConfig.rule)} weight matrix across all 784 neurons.`}
              />
            </div>
          </section>
        </div>

        <div className="right-column">
          <ValueGridHeatmap
            title="Current neuron state"
            data={snapshot.state}
            side={PATTERN_SIDE}
            maxAbs={1}
            caption={`Step ${snapshot.step} • ${snapshot.changedCount} neuron flips in last iteration`}
          />
          <EnergyPlot values={energyHistory} />
          <section className="panel">
            <div className="panel-header">
              <h3>Run state</h3>
              <p>Live summary of the current retrieval trajectory.</p>
            </div>
            <dl className="run-stats">
              <div>
                <dt>Matched memory</dt>
                <dd>{snapshot.matchedPatternIndex >= 0 ? patternSet.labels[snapshot.matchedPatternIndex] : "n/a"}</dd>
              </div>
              <div>
                <dt>Energy</dt>
                <dd>{snapshot.energy.toFixed(3)}</dd>
              </div>
              <div>
                <dt>Learning rule</dt>
                <dd>{getLearningRuleLabel(learningConfig.rule)}</dd>
              </div>
              <div>
                <dt>Convergence</dt>
                <dd>{getConvergenceRuleLabel(convergenceConfig.rule)}</dd>
              </div>
              <div>
                <dt>Step</dt>
                <dd>{snapshot.step}</dd>
              </div>
              <div>
                <dt>Changed neurons</dt>
                <dd>{snapshot.changedCount}</dd>
              </div>
              <div>
                <dt>Converged</dt>
                <dd>{snapshot.converged ? "yes" : "no"}</dd>
              </div>
            </dl>
          </section>
        </div>
      </div>

      {openHelpPanel === "dataset" ? (
        <DatasetDialog
          title="Dataset"
          summary={patternSet.description}
          onClose={() => setOpenHelpPanel(null)}
        >
          <PatternGallery patternSet={patternSet} matchedIndex={snapshot.matchedPatternIndex} />
        </DatasetDialog>
      ) : null}

      {openHelpPanel === "learning" ? (
        <HelpDialog
          content={learningHelpContent}
          onClose={() => setOpenHelpPanel(null)}
        />
      ) : null}

      {openHelpPanel === "convergence" ? (
        <HelpDialog
          content={convergenceHelpContent}
          onClose={() => setOpenHelpPanel(null)}
        />
      ) : null}
    </div>
  );
}
