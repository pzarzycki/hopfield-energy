import { useEffect, useMemo, useRef, useState } from "react";
import { CircleHelp, Pause, Play, RotateCcw, SkipForward, X } from "lucide-react";
import { BlockMath, InlineMath } from "react-katex";

import { PATTERN_SIDE, PATTERN_SIZE } from "../core/patternSets";
import { formatPrimaryTasks, getModelCatalogEntry } from "../core/modelCatalog";
import {
  applyDenseAssociativeNoise,
  buildDenseAssociativeHiddenGrid,
  createBlankDenseAssociativePattern,
  getDenseAssociativeHiddenGridSide,
  type DenseAssociativeActivation,
  type DenseAssociativeEpochMetrics,
  type DenseAssociativeFeatureMap,
  type DenseAssociativeMemoryModel,
  type DenseAssociativeSnapshot,
} from "../core/denseAssociativeMemory";
import type {
  DenseAssociativeWorkerRequest,
  DenseAssociativeWorkerResponse,
} from "../core/denseAssociativeMemoryWorkerProtocol";
import { grayscaleToFloat32, loadDatasetArchive, selectMemorySamples, summarizeDatasetArchive, type DatasetName } from "../data/datasetArchives";
import { GrayscaleCanvas } from "../features/denseHopfield/GrayscaleCanvas";
import { MemoryGallery } from "../features/denseHopfield/MemoryGallery";
import { FeatureGallery } from "../features/rbm/FeatureGallery";
import { EnergyPlot } from "../features/hopfield/EnergyPlot";
import { GrayscaleHeatmap, MatrixHeatmap, ValueGridHeatmap } from "../features/hopfield/HeatmapCanvas";
import { DatasetDialog } from "../features/common/DatasetDialog";

interface DenseAssociativeDatasetBundle {
  memoryLabels: string[];
  memoryPatterns: Float32Array[];
  trainingSampleCount: number;
  description: string;
  datasetFacts: string[];
}

function renderTextWithMath(text: string) {
  const segments = text.split(/(\$[^$]+\$)/g).filter(Boolean);
  return segments.map((segment, index) => {
    if (segment.startsWith("$") && segment.endsWith("$")) {
      return <InlineMath key={`${segment}-${index}`}>{segment.slice(1, -1)}</InlineMath>;
    }
    return <span key={`${segment}-${index}`}>{segment}</span>;
  });
}

async function loadDenseAssociativeDataset(name: DatasetName): Promise<DenseAssociativeDatasetBundle> {
  const archive = await loadDatasetArchive(name);
  const memorySamples = selectMemorySamples(archive);
  const stats = summarizeDatasetArchive(archive);
  const perClassLabel =
    stats.minSamplesPerClass === stats.maxSamplesPerClass
      ? `${stats.minSamplesPerClass} per class`
      : `${stats.minSamplesPerClass}-${stats.maxSamplesPerClass} per class`;
  return {
    memoryLabels: memorySamples.map((sample) => sample.labelName),
    memoryPatterns: memorySamples.map((sample) => grayscaleToFloat32(sample.pattern)),
    trainingSampleCount: archive.samples.length,
    description: `Original grayscale ${archive.name} exemplars loaded from the bundled binary archive. The DAM UI stays in [0,1] grayscale; the Wasm core scores and trains in a centered contrast space.`,
    datasetFacts: [`${stats.sampleCount} samples`, `${stats.classCount} classes`, perClassLabel],
  };
}

function getWeightMaxAbs(model: DenseAssociativeMemoryModel | null): number {
  if (!model) {
    return 1;
  }
  let maxAbs = 0;
  for (let index = 0; index < model.weights.length; index += 1) {
    maxAbs = Math.max(maxAbs, Math.abs(model.weights[index]));
  }
  return maxAbs || 1;
}

function getActivationSummary(activation: DenseAssociativeActivation): string {
  if (activation === "relu-power") {
    return "Rectified power is the default Krotov-style choice: the single strongest positively aligned hidden slot survives, and higher exponent sharpens competition further.";
  }
  if (activation === "signed-power") {
    return "Signed power keeps the same sparse winner path but preserves sign, so the winning hidden slot can support or suppress reconstruction directions.";
  }
  return "Softmax replaces sparse polynomial competition with normalized attention over learned prototype slots. It is available for comparison, not the default DAM path.";
}

function getSharpnessLabel(activation: DenseAssociativeActivation, sharpness: number): string {
  return activation === "softmax" ? `beta = ${sharpness}` : `n = ${sharpness}`;
}

function HelpDialog({ activation, sharpness, onClose }: { activation: DenseAssociativeActivation; sharpness: number; onClose: () => void }) {
  const formula =
    activation === "softmax"
      ? String.raw`h = \mathrm{softmax}(\beta W^\top x), \qquad x_{t+1}=Wh, \qquad \Delta W \propto x_{\text{data}} h_{\text{pos}}^\top - x_{\text{recon}} h_{\text{neg}}^\top`
      : String.raw`h_k = F((W^\top x)_k), \qquad x_{t+1}=Wh, \qquad \Delta W \propto x_{\text{data}} h_{\text{pos}}^\top - x_{\text{recon}} h_{\text{neg}}^\top`;

  return (
    <div className="modal-backdrop" onClick={onClose} role="presentation">
      <section
        className="modal-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="dam-help-dialog-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-header">
          <div>
            <h2 id="dam-help-dialog-title">Dense Associative Memory</h2>
            <p>Trainable bipartite associative memory with Krotov-style hidden competition.</p>
          </div>
          <button type="button" className="modal-close-btn" onClick={onClose} aria-label="Close help dialog">
            <X size={16} />
          </button>
        </div>
        <div className="modal-body">
          <section className="modal-section">
            <span className="modal-section-label">Math</span>
            <div className="formula-block">
              <BlockMath math={formula} />
            </div>
          </section>
          <section className="modal-section">
            <span className="modal-section-label">Interpretation</span>
            <div className="modal-prose">
              <p>{renderTextWithMath("The visible pattern $x$ drives hidden prototype slots through a sharp nonlinearity. Higher sharpness makes the hidden layer more winner-take-all.")}</p>
              <p>{renderTextWithMath("The UI stays in grayscale $[0,1]$, but the Wasm core scores and trains in a contrast-centered space before mapping reconstructions back to grayscale.")}</p>
              <p>{renderTextWithMath("Each hidden column of $W$ acts like a learned prototype. Retrieval alternates between hidden activation and visible reconstruction until the state stabilizes.")}</p>
              <p>{renderTextWithMath(`Current sharpness is ${getSharpnessLabel(activation, sharpness)}.`)}</p>
            </div>
          </section>
        </div>
      </section>
    </div>
  );
}

export default function DenseAssociativeMemoryPage() {
  const modelEntry = getModelCatalogEntry("dense-associative-memory");
  const [datasetName, setDatasetName] = useState<DatasetName>("mnist");
  const [hiddenUnits, setHiddenUnits] = useState(96);
  const [epochs, setEpochs] = useState(10);
  const [learningRate, setLearningRate] = useState(0.035);
  const [batchSize, setBatchSize] = useState(25);
  const [sharpness, setSharpness] = useState(8);
  const [activation, setActivation] = useState<DenseAssociativeActivation>("relu-power");
  const [momentum, setMomentum] = useState(0.65);
  const [weightDecay, setWeightDecay] = useState(0.0004);
  const [brushValue, setBrushValue] = useState(0.7);
  const [datasetBundle, setDatasetBundle] = useState<DenseAssociativeDatasetBundle | null>(null);
  const [model, setModel] = useState<DenseAssociativeMemoryModel | null>(null);
  const [featureMaps, setFeatureMaps] = useState<DenseAssociativeFeatureMap[]>([]);
  const [queryPattern, setQueryPattern] = useState<Float32Array>(() => createBlankDenseAssociativePattern());
  const [snapshot, setSnapshot] = useState<DenseAssociativeSnapshot | null>(null);
  const [energyHistory, setEnergyHistory] = useState<number[]>([]);
  const [trainingHistory, setTrainingHistory] = useState<DenseAssociativeEpochMetrics[]>([]);
  const [trainingError, setTrainingError] = useState(0);
  const [speed, setSpeed] = useState(8);
  const [maxPlaybackSteps, setMaxPlaybackSteps] = useState(36);
  const [corruptionLevel, setCorruptionLevel] = useState(0);
  const [obfuscationLevel, setObfuscationLevel] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [isTrainingConfigured, setIsTrainingConfigured] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [workerError, setWorkerError] = useState<string | null>(null);
  const [showHelp, setShowHelp] = useState(false);
  const [showDatasetHelp, setShowDatasetHelp] = useState(false);

  const hasAppliedQueryRef = useRef(false);
  const workerRef = useRef<Worker | null>(null);

  const displayedPatterns = useMemo(() => datasetBundle?.memoryPatterns.map((pattern) => pattern.slice()) ?? [], [datasetBundle]);
  const hiddenGrid = useMemo(() => (snapshot ? buildDenseAssociativeHiddenGrid(snapshot.hiddenActivations) : null), [snapshot]);
  const hiddenGridSide = useMemo(() => getDenseAssociativeHiddenGridSide(hiddenUnits), [hiddenUnits]);
  const labels = datasetBundle?.memoryLabels ?? [];
  const weightMaxAbs = useMemo(() => getWeightMaxAbs(model), [model]);
  const activeHiddenIndex = snapshot?.topHiddenIndex ?? -1;
  const trainerEpoch = currentEpoch;
  const trainingProgress = epochs > 0 ? Math.min(100, Math.round((trainerEpoch / epochs) * 100)) : 0;
  const latestTrainingMetrics = trainingHistory[trainingHistory.length - 1] ?? null;

  useEffect(() => {
    const worker = new Worker(new URL("../workers/denseAssociativeMemory.worker.ts", import.meta.url), { type: "module" });
    workerRef.current = worker;
    let cancelled = false;
    const blank = createBlankDenseAssociativePattern();

    const handleMessage = (event: MessageEvent<DenseAssociativeWorkerResponse>) => {
      const message = event.data;

      if (message.type === "initialized") {
        setModel(message.model);
        setFeatureMaps(message.featureMaps);
        setTrainingError(message.trainingError);
        setCurrentEpoch(message.epoch);
        setSnapshot(message.snapshot);
        setEnergyHistory([message.snapshot.energy]);
        setTrainingHistory([]);
        setIsReady(message.epoch > 0);
        setIsTrainingConfigured(true);
        setIsTraining(false);
        setIsPlaying(false);
        setWorkerError(null);
        return;
      }

      if (message.type === "trainingEpoch") {
        setModel(message.model);
        setFeatureMaps(message.featureMaps);
        setTrainingError(message.trainingError);
        setCurrentEpoch(message.epoch);
        setSnapshot(message.snapshot);
        setEnergyHistory([message.snapshot.energy]);
        setIsReady(message.epoch > 0);
        setTrainingHistory((previous) => {
          if (previous.length > 0 && previous[previous.length - 1]?.epoch === message.metrics.epoch) {
            return [...previous.slice(0, -1), message.metrics];
          }
          return [...previous, message.metrics];
        });
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

      if (message.type === "trainingPaused") {
        setIsTraining(false);
        return;
      }

      if (message.type === "playbackPaused") {
        setIsPlaying(false);
        return;
      }

      if (message.type === "error") {
        setWorkerError(message.message);
        setIsTraining(false);
        setIsPlaying(false);
      }
    };

    worker.addEventListener("message", handleMessage);
    setIsReady(false);
    setIsTraining(false);
    setIsTrainingConfigured(false);
    setIsPlaying(false);
    setWorkerError(null);
    setModel(null);
    setFeatureMaps([]);
    setDatasetBundle(null);
    setQueryPattern(blank);
    setSnapshot(null);
    setEnergyHistory([]);
    setTrainingHistory([]);
    setTrainingError(0);
    setCurrentEpoch(0);
    hasAppliedQueryRef.current = false;

    loadDenseAssociativeDataset(datasetName)
      .then((bundle) => {
        if (cancelled) {
          return;
        }
        setDatasetBundle(bundle);
        worker.postMessage({
          type: "initialize",
          datasetName,
          hiddenUnits,
          epochs,
          learningRate,
          batchSize,
          sharpness,
          activation,
          momentum,
          weightDecay,
        } satisfies DenseAssociativeWorkerRequest);
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setWorkerError(error instanceof Error ? error.message : "Failed to load the bundled dataset source.");
        }
      });

    return () => {
      cancelled = true;
      worker.removeEventListener("message", handleMessage);
      worker.terminate();
      if (workerRef.current === worker) {
        workerRef.current = null;
      }
    };
  }, [activation, batchSize, datasetName, epochs, hiddenUnits, learningRate, momentum, sharpness, weightDecay]);

  function setQueryOnWorker(nextPattern: Float32Array = queryPattern): Float32Array {
    const worker = workerRef.current;
    const normalized = nextPattern.slice();
    setQueryPattern(normalized);
    if (worker) {
      const outgoing = normalized.slice();
      worker.postMessage({ type: "setQuery", pattern: outgoing } satisfies DenseAssociativeWorkerRequest, [outgoing.buffer]);
    }
    hasAppliedQueryRef.current = true;
    return normalized;
  }

  function applyQuery(): void {
    const worker = workerRef.current;
    if (!worker || !isReady) {
      return;
    }
    setQueryOnWorker();
    setIsPlaying(true);
    worker.postMessage({
      type: "startPlayback",
      intervalMs: Math.max(40, Math.round(1000 / speed)),
      maxSteps: Math.max(1, maxPlaybackSteps),
    } satisfies DenseAssociativeWorkerRequest);
  }

  function handlePlay(): void {
    const worker = workerRef.current;
    if (!worker || !isReady) {
      return;
    }
    if (!hasAppliedQueryRef.current) {
      setQueryOnWorker();
    }
    setIsPlaying(true);
    worker.postMessage({
      type: "startPlayback",
      intervalMs: Math.max(40, Math.round(1000 / speed)),
      maxSteps: Math.max(1, maxPlaybackSteps),
    } satisfies DenseAssociativeWorkerRequest);
  }

  function handlePause(): void {
    workerRef.current?.postMessage({ type: "pausePlayback" } satisfies DenseAssociativeWorkerRequest);
  }

  function handleStep(): void {
    const worker = workerRef.current;
    if (!worker || !isReady) {
      return;
    }
    setIsPlaying(false);
    if (!hasAppliedQueryRef.current) {
      setQueryOnWorker();
    }
    worker.postMessage({ type: "stepPlayback" } satisfies DenseAssociativeWorkerRequest);
  }

  function handleReset(): void {
    setIsPlaying(false);
    workerRef.current?.postMessage({ type: "resetPlayback" } satisfies DenseAssociativeWorkerRequest);
  }

  function handleClear(): void {
    const blank = createBlankDenseAssociativePattern();
    setIsPlaying(false);
    hasAppliedQueryRef.current = false;
    setQueryOnWorker(blank);
    hasAppliedQueryRef.current = false;
  }

  function handlePatternChange(nextPattern: Float32Array): void {
    setIsPlaying(false);
    setQueryPattern(nextPattern);
    hasAppliedQueryRef.current = false;
  }

  function handleLoadPattern(index: number): void {
    if (displayedPatterns.length === 0) {
      return;
    }
    const next = applyDenseAssociativeNoise(displayedPatterns[index], corruptionLevel, obfuscationLevel);
    setIsPlaying(false);
    setQueryPattern(next);
    hasAppliedQueryRef.current = false;
  }

  function handleStartTraining(): void {
    if (!isTrainingConfigured || currentEpoch >= epochs) {
      return;
    }
    setIsTraining(true);
    workerRef.current?.postMessage({ type: "startTraining", intervalMs: 120 } satisfies DenseAssociativeWorkerRequest);
  }

  function handlePauseTraining(): void {
    workerRef.current?.postMessage({ type: "pauseTraining" } satisfies DenseAssociativeWorkerRequest);
  }

  function handleStepTraining(): void {
    setIsTraining(false);
    workerRef.current?.postMessage({ type: "trainEpoch" } satisfies DenseAssociativeWorkerRequest);
  }

  function handleResetTraining(): void {
    setIsTraining(false);
    workerRef.current?.postMessage({ type: "resetTraining" } satisfies DenseAssociativeWorkerRequest);
  }

  const reconstruction = snapshot?.reconstruction ?? createBlankDenseAssociativePattern();
  const hiddenSummary = snapshot?.hiddenActivations ?? new Float32Array(hiddenUnits);
  const trainingErrorSeries = trainingHistory.map((entry) => entry.reconstructionError);
  const contrastiveGapSeries = trainingHistory.map((entry) => entry.contrastiveGap);
  const hiddenActivationSeries = trainingHistory.map((entry) => entry.hiddenActivation);
  const winnerShareSeries = trainingHistory.map((entry) => entry.winnerShare);
  const energyValue = snapshot?.energy ?? 0;
  const [activePhase, setActivePhase] = useState<"training" | "inference">("training");
  const trainingPhaseStatus = isTraining ? "running" : isReady ? "trained" : isTrainingConfigured ? "required" : "loading";
  const inferencePhaseStatus = isReady ? "available" : isTrainingConfigured ? "locked" : "loading";

  return (
    <div className="page-shell rbm-page">
      <header className="hero">
        <div>
          <p className="eyebrow">Wasm-backed worker runtime</p>
          <h1>Dense Associative Memory</h1>
          <p className="hero-copy">
            Trainable Krotov-style bipartite associative memory over grayscale MNIST or Fashion-MNIST, with configurable hidden competition,
            iterative retrieval, learned prototype slots, and a live hidden-to-visible matrix.
          </p>
          <p className="hero-task">Primary task: {formatPrimaryTasks(modelEntry.primaryTasks)}</p>
        </div>
        <div className="hero-stats">
          <div className="stat-card"><span>Visible units</span><strong>{PATTERN_SIZE}</strong></div>
          <div className="stat-card"><span>Hidden units</span><strong>{hiddenUnits}</strong></div>
          <div className="stat-card"><span>Status</span><strong>{isTraining ? "training" : isPlaying ? "running" : isReady ? "ready" : isTrainingConfigured ? "configured" : "loading"}</strong></div>
        </div>
      </header>

      {workerError ? <div className="error-banner">{workerError}</div> : null}

      <section className="panel architecture-bar">
        <div className="control-strip-group">
          <div className="control-strip-header">
            <span className="control-strip-title">Dataset</span>
            <button type="button" className="help-btn" aria-expanded={showDatasetHelp} onClick={() => setShowDatasetHelp((current) => !current)} title="Dataset help">
              <CircleHelp size={15} />
            </button>
          </div>
          <label className="field compact-field"><span>Dataset</span><select value={datasetName} onChange={(event) => setDatasetName(event.target.value as DatasetName)}><option value="mnist">MNIST</option><option value="fashion-mnist">Fashion-MNIST</option></select></label>
          <p className="control-strip-note">{datasetBundle?.description ?? "Loading original dataset exemplars..."}</p>
        </div>

        <div className="control-strip-group">
          <div className="control-strip-header">
            <span className="control-strip-title">Architecture</span>
            <button type="button" className="help-btn" aria-expanded={showHelp} onClick={() => setShowHelp((current) => !current)} title="Dense Associative Memory help">
              <CircleHelp size={15} />
            </button>
          </div>
          <div className="rule-config-grid rbm-architecture-grid">
            <label className="field compact-field"><span>Nonlinearity</span><select value={activation} onChange={(event) => setActivation(event.target.value as DenseAssociativeActivation)}><option value="relu-power">ReLU power</option><option value="signed-power">Signed power</option><option value="softmax">Softmax</option></select></label>
            <label className="field compact-field"><span>Visible units</span><input type="text" value={PATTERN_SIZE} readOnly /></label>
            <label className="field compact-field"><span>Hidden units</span><input type="number" min="16" max="256" step="8" value={hiddenUnits} onChange={(event) => setHiddenUnits(Number(event.target.value))} /></label>
          </div>
          <p className="control-strip-note">{getActivationSummary(activation)}</p>
        </div>

      </section>

      <section className="phase-tabs" aria-label="Dense Associative Memory phase">
        <div className="phase-tabs__list" role="tablist" aria-label="Dense Associative Memory phase">
          <button type="button" className={activePhase === "training" ? "is-active" : ""} onClick={() => setActivePhase("training")} role="tab" aria-selected={activePhase === "training"}>
            Training
            <span className={`phase-tab-indicator phase-tab-indicator--${trainingPhaseStatus}`}>{trainingPhaseStatus}</span>
          </button>
          <button type="button" className={activePhase === "inference" ? "is-active" : ""} onClick={() => setActivePhase("inference")} role="tab" aria-selected={activePhase === "inference"}>
            Inference
            <span className={`phase-tab-indicator phase-tab-indicator--${inferencePhaseStatus}`}>{inferencePhaseStatus}</span>
          </button>
        </div>
      </section>

      <div className="rbm-stage-grid">
        <div className="rbm-stage-main">
          {activePhase === "training" ? (
          <section className="panel">
            <div className="panel-header"><h3>Training process</h3><p>Train the prototype matrix first, then inspect how sharply the hidden layer specializes and how well reconstructions improve over epochs.</p></div>
            <div className="rbm-training-top">
              <section className="rbm-training-setup">
                <div className="panel-header rbm-subpanel-header"><h4>Training parameters</h4><p>These controls set the contrastive training dynamics for the bipartite memory.</p></div>
                <div className="rule-config-grid rbm-rule-grid">
                  <label className="field compact-field"><span>Epochs</span><input type="number" min="1" max="30" step="1" value={epochs} onChange={(event) => setEpochs(Number(event.target.value))} /></label>
                  <label className="field compact-field"><span>Learn rate</span><input type="number" min="0.005" max="0.2" step="0.005" value={learningRate} onChange={(event) => setLearningRate(Number(event.target.value))} /></label>
                  <label className="field compact-field"><span>Batch size</span><input type="number" min="5" max="200" step="5" value={batchSize} onChange={(event) => setBatchSize(Number(event.target.value))} /></label>
                  <label className="field compact-field"><span>{activation === "softmax" ? "Beta" : "Power"}</span><input type="number" min="2" max="24" step="1" value={sharpness} onChange={(event) => setSharpness(Number(event.target.value))} /></label>
                  <label className="field compact-field"><span>Momentum</span><input type="number" min="0" max="0.95" step="0.01" value={momentum} onChange={(event) => setMomentum(Number(event.target.value))} /></label>
                  <label className="field compact-field"><span>Weight decay</span><input type="number" min="0" max="0.005" step="0.00005" value={weightDecay} onChange={(event) => setWeightDecay(Number(event.target.value))} /></label>
                </div>
                <p className="control-strip-note">Training uses a positive data phase and a negative reconstruction phase over {datasetBundle?.trainingSampleCount ?? "the bundled"} grayscale samples. Sharpness is {getSharpnessLabel(activation, sharpness)}.</p>
              </section>

              <section className="rbm-training-execution">
                <div className="panel-header rbm-subpanel-header"><h4>Training execution</h4><p>Run continuously, step one epoch at a time, or reset the trainer to compare another nonlinearity or hidden size.</p></div>
                <div className="control-actions rbm-training-actions">
                  {isTraining ? (
                    <button type="button" className="icon-btn" onClick={handlePauseTraining} disabled={!isTrainingConfigured}><Pause size={14} /><span>Pause training</span></button>
                  ) : (
                    <button type="button" className="icon-btn primary" onClick={handleStartTraining} disabled={!isTrainingConfigured || trainerEpoch >= epochs}><Play size={14} /><span>Start training</span></button>
                  )}
                  <button type="button" className="icon-btn" onClick={handleStepTraining} disabled={!isTrainingConfigured || isTraining || trainerEpoch >= epochs}><SkipForward size={14} /><span>Step epoch</span></button>
                  <button type="button" className="icon-btn" onClick={handleResetTraining} disabled={!isTrainingConfigured}><RotateCcw size={14} /><span>Reset trainer</span></button>
                </div>
                <div className="rbm-training-progress">
                  <div className="rbm-training-progress-copy"><strong>Epoch {trainerEpoch} / {epochs}</strong><span>{isTraining ? "Training is running. Watch reconstruction, competition, and energy metrics update after each epoch." : "Use Step epoch if you want to inspect prototype formation slowly."}</span></div>
                  <div className="rbm-training-progress-bar" aria-hidden="true"><div className="rbm-training-progress-fill" style={{ width: `${trainingProgress}%` }} /></div>
                </div>
              </section>
            </div>
            <div className="rbm-training-metrics">
              <div className="rbm-training-stat"><span>Training samples</span><strong>{datasetBundle?.trainingSampleCount ?? "n/a"}</strong></div>
              <div className="rbm-training-stat"><span>Current epoch</span><strong>{trainerEpoch}</strong></div>
              <div className="rbm-training-stat"><span>Final recon. error</span><strong>{latestTrainingMetrics ? latestTrainingMetrics.reconstructionError.toFixed(4) : "n/a"}</strong></div>
              <div className="rbm-training-stat"><span>Contrastive gap</span><strong>{latestTrainingMetrics ? latestTrainingMetrics.contrastiveGap.toFixed(4) : "n/a"}</strong></div>
              <div className="rbm-training-stat"><span>Mean hidden act.</span><strong>{latestTrainingMetrics ? latestTrainingMetrics.hiddenActivation.toFixed(3) : "n/a"}</strong></div>
              <div className="rbm-training-stat"><span>Winner share</span><strong>{latestTrainingMetrics ? `${(latestTrainingMetrics.winnerShare * 100).toFixed(1)}%` : "n/a"}</strong></div>
              <div className="rbm-training-stat"><span>Mean |W|</span><strong>{latestTrainingMetrics ? latestTrainingMetrics.weightMeanAbs.toFixed(4) : "n/a"}</strong></div>
              <div className="rbm-training-stat"><span>Avg energy</span><strong>{latestTrainingMetrics ? latestTrainingMetrics.energy.toFixed(3) : "n/a"}</strong></div>
              <div className="rbm-training-stat"><span>Train error</span><strong>{trainingError.toFixed(4)}</strong></div>
            </div>
            <div className="rbm-training-plots">
              <EnergyPlot values={trainingErrorSeries} title="Reconstruction Error" caption="Average visible reconstruction error after each epoch." xLabel="epoch" yLabel="error" width={320} height={140} />
              <EnergyPlot values={contrastiveGapSeries} title="Contrastive Gap" caption="Mean data-vs-reconstruction mismatch." xLabel="epoch" yLabel="gap" width={320} height={140} />
              <EnergyPlot values={hiddenActivationSeries} title="Hidden Activity" caption="Average hidden activation magnitude." xLabel="epoch" yLabel="activity" width={320} height={140} />
              <EnergyPlot values={winnerShareSeries} title="Winner Share" caption="How much one hidden slot dominates the activation mass." xLabel="epoch" yLabel="share" width={320} height={140} />
            </div>
          </section>
          ) : null}

          {activePhase === "inference" ? (
          <section className="panel">
            <div className="panel-header"><h3>Query and retrieval playback</h3><p>After training, apply a visible query and inspect how repeated hidden-prototype updates sharpen or stabilize the reconstruction.</p></div>
            <div className="convergence-strip">
              <div className="convergence-strip-main">
                <div className="convergence-fields">
                  <div className="rule-config-grid rbm-playback-grid">
                    <label className="field compact-field"><span>Playback speed</span><input type="range" min="1" max="20" value={speed} onChange={(event) => setSpeed(Number(event.target.value))} /><strong className="range-value">{speed} steps/s</strong></label>
                    <label className="field compact-field"><span>Steps</span><input type="number" min="1" max="120" step="1" value={maxPlaybackSteps} onChange={(event) => setMaxPlaybackSteps(Number(event.target.value))} /></label>
                  </div>
                  <p className="control-strip-note rbm-gibbs-note">Each playback step re-encodes through the hidden prototype layer and reconstructs back to visible space until the trajectory settles.</p>
                </div>
                <div className="control-actions">
                  <button type="button" className="icon-btn primary" onClick={applyQuery} disabled={!isReady || isTraining}><SkipForward size={14} /><span>Apply</span></button>
                  {isPlaying ? (
                    <button type="button" className="icon-btn" onClick={handlePause}><Pause size={14} /><span>Pause</span></button>
                  ) : (
                    <button type="button" className="icon-btn" onClick={handlePlay} disabled={!isReady || isTraining}><Play size={14} /><span>Play</span></button>
                  )}
                  <button type="button" className="icon-btn" onClick={handleStep} disabled={!isReady || isPlaying || isTraining}><SkipForward size={14} /><span>Step</span></button>
                  <button type="button" className="icon-btn" onClick={handleReset} disabled={!isReady || isTraining}><RotateCcw size={14} /><span>Reset</span></button>
                </div>
              </div>
            </div>
          </section>
          ) : null}

          {activePhase === "inference" ? (
          <section className="rbm-canvas-row">
            <section className="panel input-panel">
              <div className="panel-header"><h3>Query</h3><p>Pick a stored exemplar, degrade it if needed, then edit the visible query directly before associative playback.</p></div>
              <div className="query-workbench">
                <div className="query-toolbar">
                  <div className="field compact-field">
                    <span>Examples</span>
                    <div className="pattern-picker pattern-picker--compact">
                      {labels.map((label, index) => (
                        <button key={`${label}-${index}`} type="button" onClick={() => handleLoadPattern(index)} title={`Load ${label}`}>
                          {label}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="query-toolbar--row">
                    <label className="field compact-field">
                      <span>Corruption</span>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={corruptionLevel}
                        onChange={(event) => setCorruptionLevel(Number(event.target.value))}
                        title="Invert intensity on a portion of the selected exemplar before loading the query"
                      />
                      <strong className="range-value">{corruptionLevel}%</strong>
                    </label>
                    <label className="field compact-field">
                      <span>Obfuscation</span>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={obfuscationLevel}
                        onChange={(event) => setObfuscationLevel(Number(event.target.value))}
                        title="Set part of the selected exemplar to zero before playback"
                      />
                      <strong className="range-value">{obfuscationLevel}%</strong>
                    </label>
                  </div>
                </div>
                <div className="input-grid">
                  <GrayscaleCanvas pattern={queryPattern} onChange={handlePatternChange} paintValue={brushValue} />
                  <div className="query-actions query-actions--with-brush">
                    <button type="button" onClick={handleClear} title="Clear the query editor">clear</button>
                    <label className="field compact-field gaussian-brush-field">
                      <span>Brush</span>
                      <input type="range" min="0.1" max="1" step="0.05" value={brushValue} onChange={(event) => setBrushValue(Number(event.target.value))} title="Painting intensity used in the query editor" />
                      <strong className="range-value">{brushValue.toFixed(2)}</strong>
                    </label>
                  </div>
                </div>
              </div>
            </section>

            <GrayscaleHeatmap
              title="Current reconstruction"
              data={reconstruction}
              side={PATTERN_SIDE}
              scale={9}
              autoContrast
              caption={snapshot ? `Step ${snapshot.step} • mean absolute reconstruction error ${snapshot.reconstructionError.toFixed(4)}` : "Current visible reconstruction."}
            />

            {hiddenGrid ? (
              <ValueGridHeatmap
                title="Hidden activations"
                data={hiddenGrid}
                side={hiddenGridSide}
                maxAbs={Math.max(snapshot?.topHiddenActivation ?? 1, 1)}
                scale={18}
                caption="Magnitude and sign of the hidden prototype activations."
                xLabel="hidden x"
                yLabel="hidden y"
              />
            ) : null}
          </section>
          ) : null}

          {activePhase === "training" && model ? (
            <div className="rbm-matrix-scroll">
              <MatrixHeatmap
                title="Hidden-to-visible prototypes"
                data={model.weights}
                rows={model.hiddenUnits}
                columns={model.visibleUnits}
                maxAbs={weightMaxAbs}
                caption="Each row is one learned prototype slot over visible pixels."
                xLabel="visible pixel i"
                yLabel="hidden unit k"
              />
            </div>
          ) : null}
        </div>

        <div className="rbm-stage-side">
          {activePhase === "training" ? (
            <section className="panel">
              <div className="panel-header"><h3>Training summary</h3><p>Compact summary of the latest epoch and hidden-competition quality.</p></div>
              <dl className="run-stats">
                <div><dt>Nonlinearity</dt><dd>{activation === "relu-power" ? "ReLU power" : activation === "signed-power" ? "Signed power" : "Softmax"}</dd></div>
                <div><dt>Sharpness</dt><dd>{getSharpnessLabel(activation, sharpness)}</dd></div>
                <div><dt>Training samples</dt><dd>{datasetBundle?.trainingSampleCount ?? "n/a"}</dd></div>
                <div><dt>Current epoch</dt><dd>{trainerEpoch}</dd></div>
                <div><dt>Recon. error</dt><dd>{latestTrainingMetrics ? latestTrainingMetrics.reconstructionError.toFixed(4) : "n/a"}</dd></div>
                <div><dt>Contrastive gap</dt><dd>{latestTrainingMetrics ? latestTrainingMetrics.contrastiveGap.toFixed(4) : "n/a"}</dd></div>
                <div><dt>Winner share</dt><dd>{latestTrainingMetrics ? `${(latestTrainingMetrics.winnerShare * 100).toFixed(1)}%` : "n/a"}</dd></div>
                <div><dt>Mean |W|</dt><dd>{latestTrainingMetrics ? latestTrainingMetrics.weightMeanAbs.toFixed(4) : "n/a"}</dd></div>
              </dl>
            </section>
          ) : null}
          {activePhase === "inference" ? (
          <section className="panel">
            <div className="panel-header"><h3>Run state</h3><p>Live summary of the current retrieval trajectory and hidden competition.</p></div>
            <dl className="run-stats">
              <div><dt>Matched exemplar</dt><dd>{snapshot && snapshot.matchedPatternIndex >= 0 ? labels[snapshot.matchedPatternIndex] : "n/a"}</dd></div>
              <div><dt>Nonlinearity</dt><dd>{activation === "relu-power" ? "ReLU power" : activation === "signed-power" ? "Signed power" : "Softmax"}</dd></div>
              <div><dt>Sharpness</dt><dd>{getSharpnessLabel(activation, sharpness)}</dd></div>
              <div><dt>Energy</dt><dd>{energyValue.toFixed(3)}</dd></div>
              <div><dt>Recon. error</dt><dd>{snapshot ? snapshot.reconstructionError.toFixed(4) : "n/a"}</dd></div>
              <div><dt>Train error</dt><dd>{trainingError.toFixed(4)}</dd></div>
              <div><dt>Top hidden</dt><dd>{snapshot && snapshot.topHiddenIndex >= 0 ? `h${snapshot.topHiddenIndex + 1}` : "n/a"}</dd></div>
              <div><dt>Top activation</dt><dd>{snapshot ? snapshot.topHiddenActivation.toFixed(3) : "n/a"}</dd></div>
              <div><dt>Hidden entropy</dt><dd>{snapshot ? snapshot.hiddenEntropy.toFixed(3) : "n/a"}</dd></div>
              <div><dt>Current step</dt><dd>{snapshot ? snapshot.step : 0}</dd></div>
              <div><dt>Stable</dt><dd>{snapshot?.converged ? "yes" : "no"}</dd></div>
              <div><dt>Mean |h|</dt><dd>{hiddenSummary.length > 0 ? (hiddenSummary.reduce((total, value) => total + Math.abs(value), 0) / hiddenSummary.length).toFixed(3) : "n/a"}</dd></div>
            </dl>
          </section>
          ) : null}
          {activePhase === "inference" ? (
          <EnergyPlot values={energyHistory} title="Associative Energy" caption="Compact retrieval trace over the current query." xLabel="step" yLabel="energy" width={320} height={140} />
          ) : null}
        </div>
      </div>

      {activePhase === "training" ? (
      <div className="rbm-gallery-grid">
        {featureMaps.length > 0 ? <FeatureGallery features={featureMaps} activeHiddenIndex={activeHiddenIndex} /> : null}
      </div>
      ) : null}

      {showDatasetHelp && datasetBundle ? (
        <DatasetDialog title="Dataset" summary={datasetBundle.description} facts={datasetBundle.datasetFacts} onClose={() => setShowDatasetHelp(false)}>
          <MemoryGallery labels={datasetBundle.memoryLabels} patterns={displayedPatterns} matchedIndex={snapshot?.matchedPatternIndex ?? -1} />
        </DatasetDialog>
      ) : null}

      {showHelp ? <HelpDialog activation={activation} sharpness={sharpness} onClose={() => setShowHelp(false)} /> : null}
    </div>
  );
}
