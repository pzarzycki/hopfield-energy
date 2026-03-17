import { useEffect, useMemo, useRef, useState } from "react";
import { CircleHelp, Pause, Play, RotateCcw, SkipForward, X } from "lucide-react";
import { BlockMath, InlineMath } from "react-katex";

import { PATTERN_SIDE, PATTERN_SIZE } from "../core/patternSets";
import { formatPrimaryTasks, getModelCatalogEntry } from "../core/modelCatalog";
import {
  applyVisibleNoise,
  buildHiddenGrid,
  createBlankVisiblePattern,
  getHiddenGridSide,
  quantizeVisiblePattern,
  type RBMEpochMetrics,
  type RBMFeatureMap,
  type RBMModel,
  type RBMSnapshot,
  type RBMVisibleModel,
} from "../core/rbm";
import type { RBMBackendKind, RBMWorkerRequest, RBMWorkerResponse } from "../core/rbmWorkerProtocol";
import { GrayscaleCanvas } from "../features/denseHopfield/GrayscaleCanvas";
import { MemoryGallery } from "../features/denseHopfield/MemoryGallery";
import { EnergyPlot } from "../features/hopfield/EnergyPlot";
import { GrayscaleHeatmap, MatrixHeatmap, ValueGridHeatmap } from "../features/hopfield/HeatmapCanvas";
import { FeatureGallery } from "../features/rbm/FeatureGallery";
import { DatasetDialog } from "../features/common/DatasetDialog";
import { grayscaleToFloat32, loadDatasetArchive, selectMemorySamples, summarizeDatasetArchive, type DatasetName } from "../data/datasetArchives";

interface RBMDatasetBundle {
  memoryLabels: string[];
  memoryPatterns: Float32Array[];
  trainingPatterns: Float32Array[];
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

async function loadRBMDataset(name: DatasetName): Promise<RBMDatasetBundle> {
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
    trainingPatterns: archive.samples.map((sample) => grayscaleToFloat32(sample.pattern)),
    trainingSampleCount: archive.samples.length,
    description: `Original grayscale ${archive.name} exemplars loaded from the bundled binary archive.`,
    datasetFacts: [`${stats.sampleCount} samples`, `${stats.classCount} classes`, perClassLabel],
  };
}

function getWeightMaxAbs(model: RBMModel | null): number {
  if (!model) {
    return 1;
  }

  let maxAbs = 0;
  for (let index = 0; index < model.weights.length; index += 1) {
    maxAbs = Math.max(maxAbs, Math.abs(model.weights[index]));
  }
  return maxAbs || 1;
}

function getVariantSummary(visibleModel: RBMVisibleModel): string {
  return visibleModel === "bernoulli"
    ? "Bernoulli visible units threshold the source images to binary before training and querying."
    : "Gaussian visible units keep the source images continuous in [0, 1] and reconstruct grayscale values directly.";
}

function HelpDialog({ visibleModel, onClose }: { visibleModel: RBMVisibleModel; onClose: () => void }) {
  const formula =
    visibleModel === "bernoulli"
      ? String.raw`p(h_j=1 \mid v)=\sigma(c_j + W_j^\top v), \qquad
p(v_i=1 \mid h)=\sigma(b_i + W_i h), \qquad
F(v) = -b^\top v - \sum_j \log(1 + e^{c_j + W_j^\top v})`
      : String.raw`p(h_j=1 \mid v)=\sigma(c_j + W_j^\top v), \qquad
\hat{v}_i=b_i + W_i h, \qquad
F(v)=\frac{1}{2}\lVert v-b \rVert^2 - \sum_j \log(1 + e^{c_j + W_j^\top v})`;

  return (
    <div className="modal-backdrop" onClick={onClose} role="presentation">
      <section
        className="modal-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="rbm-help-dialog-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-header">
          <div>
            <h2 id="rbm-help-dialog-title">Restricted Boltzmann Machine</h2>
            <p>
              {visibleModel === "bernoulli"
                ? "Bernoulli-Bernoulli RBM over on-the-fly binarized source images."
                : "Gaussian-Bernoulli RBM over continuous grayscale source images."}
            </p>
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
              <p>
                {visibleModel === "bernoulli"
                  ? renderTextWithMath("The visible layer is a binary image $v \\in \\{0,1\\}^{784}$. The app derives that binary view by thresholding the original grayscale exemplar before training or querying the model.")
                  : renderTextWithMath("The visible layer is continuous $v \\in [0,1]^{784}$. The app keeps the original grayscale exemplar values during training, querying, and reconstruction.")}
              </p>
              <p>{renderTextWithMath("A Gibbs step alternates between hidden-unit sampling and visible reconstruction. The connectivity matrix below shows the learned hidden-to-visible weights $W$.")}</p>
              <p>{renderTextWithMath("Training uses contrastive divergence: move weights toward data-driven correlations and away from short-run model reconstructions.")}</p>
            </div>
          </section>
          <section className="modal-section">
            <span className="modal-section-label">Notes</span>
            <ul className="modal-list modal-list--notes">
              <li>{renderTextWithMath("The app now uses one dataset source only: the bundled original grayscale exemplars.")}</li>
              <li>{renderTextWithMath("Bernoulli mode binarizes those exemplars dynamically. Gaussian mode keeps them continuous, so the stored memories and query editor remain grayscale.")}</li>
              <li>{renderTextWithMath("The connectivity matrix is rectangular: rows are hidden units, columns are visible pixels.")}</li>
            </ul>
          </section>
        </div>
      </section>
    </div>
  );
}

export default function RestrictedBoltzmannMachinePage() {
  const modelEntry = getModelCatalogEntry("rbm");
  const [datasetName, setDatasetName] = useState<DatasetName>("mnist");
  const [visibleModel, setVisibleModel] = useState<RBMVisibleModel>("bernoulli");
  const [hiddenUnits, setHiddenUnits] = useState(64);
  const [epochs, setEpochs] = useState(8);
  const [learningRate, setLearningRate] = useState(0.05);
  const [batchSize, setBatchSize] = useState(25);
  const [cdSteps, setCdSteps] = useState(3);
  const [momentum, setMomentum] = useState(0.72);
  const [weightDecay, setWeightDecay] = useState(0.00015);
  const [brushValue, setBrushValue] = useState(0.7);
  const [datasetBundle, setDatasetBundle] = useState<RBMDatasetBundle | null>(null);
  const [model, setModel] = useState<RBMModel | null>(null);
  const [featureMaps, setFeatureMaps] = useState<RBMFeatureMap[]>([]);
  const [queryPattern, setQueryPattern] = useState<Float32Array>(() => createBlankVisiblePattern());
  const [snapshot, setSnapshot] = useState<RBMSnapshot | null>(null);
  const [energyHistory, setEnergyHistory] = useState<number[]>([]);
  const [trainingHistory, setTrainingHistory] = useState<RBMEpochMetrics[]>([]);
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
  const [backendKind, setBackendKind] = useState<RBMBackendKind>("wasm-core");
  const [workerError, setWorkerError] = useState<string | null>(null);
  const [showHelp, setShowHelp] = useState(false);
  const [showDatasetHelp, setShowDatasetHelp] = useState(false);

  const hasAppliedQueryRef = useRef(false);
  const workerRef = useRef<Worker | null>(null);

  const referencePatterns = useMemo(
    () => datasetBundle?.memoryPatterns.map((pattern) => quantizeVisiblePattern(pattern, visibleModel)) ?? [],
    [datasetBundle, visibleModel],
  );
  const displayedPatterns = useMemo(() => referencePatterns.map((pattern) => pattern.slice()), [referencePatterns]);
  const hiddenGrid = useMemo(() => (snapshot ? buildHiddenGrid(snapshot.hiddenProbabilities) : null), [snapshot]);
  const hiddenGridSide = useMemo(() => getHiddenGridSide(hiddenUnits), [hiddenUnits]);
  const labels = datasetBundle?.memoryLabels ?? [];
  const weightMaxAbs = useMemo(() => getWeightMaxAbs(model), [model]);
  const activeHiddenIndex = useMemo(() => {
    if (!snapshot || snapshot.hiddenProbabilities.length === 0) {
      return -1;
    }
    let bestIndex = 0;
    for (let index = 1; index < snapshot.hiddenProbabilities.length; index += 1) {
      if (snapshot.hiddenProbabilities[index] > snapshot.hiddenProbabilities[bestIndex]) {
        bestIndex = index;
      }
    }
    return bestIndex;
  }, [snapshot]);

  useEffect(() => {
    const nextBrushValue = visibleModel === "gaussian" ? 0.7 : 1;
    setBrushValue(nextBrushValue);
  }, [visibleModel]);

  useEffect(() => {
    const worker = new Worker(new URL("../workers/rbm.worker.ts", import.meta.url), { type: "module" });
    workerRef.current = worker;
    let cancelled = false;
    const blank = createBlankVisiblePattern();

    const handleMessage = (event: MessageEvent<RBMWorkerResponse>) => {
      const message = event.data;

      if (message.type === "initialized") {
        setBackendKind(message.backend);
        setModel(message.model);
        setFeatureMaps(message.featureMaps);
        setTrainingError(message.trainingError);
        setCurrentEpoch(message.epoch);
        setSnapshot(message.snapshot);
        setEnergyHistory([message.snapshot.freeEnergy]);
        setTrainingHistory([]);
        setIsReady(message.epoch > 0);
        setIsTrainingConfigured(true);
        setIsTraining(false);
        setIsPlaying(false);
        setWorkerError(null);
        return;
      }

      if (message.type === "trainingEpoch") {
        setBackendKind(message.backend);
        setModel(message.model);
        setFeatureMaps(message.featureMaps);
        setTrainingError(message.trainingError);
        setCurrentEpoch(message.epoch);
        setSnapshot(message.snapshot);
        setEnergyHistory([message.snapshot.freeEnergy]);
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
            return [message.snapshot.freeEnergy];
          }
          return [...previous, message.snapshot.freeEnergy];
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
    setBackendKind("wasm-core");
    hasAppliedQueryRef.current = false;

    loadRBMDataset(datasetName)
      .then((bundle) => {
        if (cancelled) {
          return;
        }
        setDatasetBundle(bundle);
        worker.postMessage({
          type: "initialize",
          datasetName,
          visibleModel,
          hiddenUnits,
          epochs,
          learningRate,
          batchSize,
          cdSteps,
          momentum,
          weightDecay,
        } satisfies RBMWorkerRequest);
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
  }, [batchSize, cdSteps, datasetName, epochs, hiddenUnits, learningRate, momentum, visibleModel, weightDecay]);

  function setQueryOnWorker(nextPattern: Float32Array = queryPattern): Float32Array {
    const worker = workerRef.current;
    const normalized = quantizeVisiblePattern(nextPattern, visibleModel);
    setQueryPattern(normalized);
    if (worker) {
      const outgoing = normalized.slice();
      worker.postMessage({ type: "setQuery", pattern: outgoing } satisfies RBMWorkerRequest, [outgoing.buffer]);
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
    } satisfies RBMWorkerRequest);
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
    } satisfies RBMWorkerRequest);
  }

  function handlePause(): void {
    workerRef.current?.postMessage({ type: "pausePlayback" } satisfies RBMWorkerRequest);
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
    worker.postMessage({ type: "stepPlayback" } satisfies RBMWorkerRequest);
  }

  function handleReset(): void {
    setIsPlaying(false);
    workerRef.current?.postMessage({ type: "resetPlayback" } satisfies RBMWorkerRequest);
  }

  function handleClear(): void {
    const blank = createBlankVisiblePattern();
    setIsPlaying(false);
    hasAppliedQueryRef.current = false;
    setQueryOnWorker(blank);
    hasAppliedQueryRef.current = false;
  }

  function handlePatternChange(nextPattern: Float32Array): void {
    const normalized = quantizeVisiblePattern(nextPattern, visibleModel);
    setIsPlaying(false);
    setQueryPattern(normalized);
    hasAppliedQueryRef.current = false;
  }

  function handleLoadPattern(index: number): void {
    if (referencePatterns.length === 0) {
      return;
    }

    const next = applyVisibleNoise(referencePatterns[index], corruptionLevel, obfuscationLevel, visibleModel);
    setIsPlaying(false);
    setQueryPattern(next);
    hasAppliedQueryRef.current = false;
  }

  const reconstruction = snapshot?.reconstruction ?? createBlankVisiblePattern();
  const hiddenSummary = snapshot?.hiddenProbabilities ?? new Float32Array(hiddenUnits);
  const latestTrainingMetrics = trainingHistory[trainingHistory.length - 1] ?? null;
  const trainingErrorSeries = trainingHistory.map((entry) => entry.reconstructionError);
  const contrastiveGapSeries = trainingHistory.map((entry) => entry.contrastiveGap);
  const hiddenActivationSeries = trainingHistory.map((entry) => entry.hiddenActivation);
  const weightMeanAbsSeries = trainingHistory.map((entry) => entry.weightMeanAbs);
  const trainerEpoch = currentEpoch;
  const trainingProgress = epochs > 0 ? Math.min(100, Math.round((trainerEpoch / epochs) * 100)) : 0;

  function handleStartTraining(): void {
    if (!isTrainingConfigured || currentEpoch >= epochs) {
      return;
    }
    setIsTraining(true);
    workerRef.current?.postMessage({
      type: "startTraining",
      intervalMs: 120,
    } satisfies RBMWorkerRequest);
  }

  function handlePauseTraining(): void {
    workerRef.current?.postMessage({ type: "pauseTraining" } satisfies RBMWorkerRequest);
  }

  function handleStepTraining(): void {
    setIsTraining(false);
    workerRef.current?.postMessage({ type: "trainEpoch" } satisfies RBMWorkerRequest);
  }

  function handleResetTraining(): void {
    setIsTraining(false);
    workerRef.current?.postMessage({ type: "resetTraining" } satisfies RBMWorkerRequest);
  }

  return (
    <div className="page-shell rbm-page">
      <header className="hero">
        <div>
          <p className="eyebrow">Wasm-backed worker runtime</p>
          <h1>Restricted Boltzmann Machine</h1>
          <p className="hero-copy">
            Switchable Bernoulli and Gaussian visible-unit RBMs over one shared grayscale MNIST or Fashion-MNIST source,
            with Gibbs reconstruction, learned hidden features, and a live hidden-to-visible connectivity matrix.
          </p>
          <p className="hero-task">Primary task: {formatPrimaryTasks(modelEntry.primaryTasks)}</p>
        </div>
        <div className="hero-stats">
          <div className="stat-card">
            <span>Visible units</span>
            <strong>{PATTERN_SIZE}</strong>
          </div>
          <div className="stat-card">
            <span>Hidden units</span>
            <strong>{hiddenUnits}</strong>
          </div>
          <div className="stat-card">
            <span>Status</span>
            <strong>{isTraining ? "training" : isPlaying ? "running" : isReady ? "ready" : isTrainingConfigured ? "configured" : "loading"}</strong>
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
              aria-expanded={showDatasetHelp}
              onClick={() => setShowDatasetHelp((current) => !current)}
              title="Dataset help"
            >
              <CircleHelp size={15} />
            </button>
          </div>
          <label className="field compact-field">
            <span>Dataset</span>
            <select value={datasetName} onChange={(event) => setDatasetName(event.target.value as DatasetName)}>
              <option value="mnist">MNIST</option>
              <option value="fashion-mnist">Fashion-MNIST</option>
            </select>
          </label>
          <p className="control-strip-note">
            {datasetBundle?.description ?? "Loading original dataset exemplars..."}
          </p>
        </div>

        <div className="control-strip-group">
          <div className="control-strip-header">
            <span className="control-strip-title">Architecture</span>
            <button
              type="button"
              className="help-btn"
              aria-expanded={showHelp}
              onClick={() => setShowHelp((current) => !current)}
              title="Restricted Boltzmann Machine help"
            >
              <CircleHelp size={15} />
            </button>
          </div>
          <div className="rule-config-grid rbm-architecture-grid">
            <label className="field compact-field">
              <span>Visible model</span>
              <select value={visibleModel} onChange={(event) => setVisibleModel(event.target.value as RBMVisibleModel)}>
                <option value="bernoulli">Bernoulli</option>
                <option value="gaussian">Gaussian</option>
              </select>
            </label>
            <label className="field compact-field">
              <span>Visible units</span>
              <input type="text" value={PATTERN_SIZE} readOnly />
            </label>
            <label className="field compact-field">
              <span>Hidden units</span>
              <input type="number" min="16" max="196" step="4" value={hiddenUnits} onChange={(event) => setHiddenUnits(Number(event.target.value))} />
            </label>
          </div>
          <p className="control-strip-note">
            {getVariantSummary(visibleModel)}
          </p>
        </div>

      </section>

      <div className="rbm-stage-grid">
        <div className="rbm-stage-main">
          <section className="panel">
            <div className="panel-header">
              <h3>Training process</h3>
              <p>Start here. These controls and plots tell you whether the RBM is actually learning useful structure from the training subset.</p>
            </div>
            <div className="rbm-training-top">
              <section className="rbm-training-setup">
                <div className="panel-header rbm-subpanel-header">
                  <h4>Training parameters</h4>
                  <p>These values control how the fixed architecture is optimized over the training subset.</p>
                </div>
                <div className="rule-config-grid rbm-rule-grid">
                  <label className="field compact-field">
                    <span>Epochs</span>
                    <input type="number" min="1" max="24" step="1" value={epochs} onChange={(event) => setEpochs(Number(event.target.value))} />
                  </label>
                  <label className="field compact-field">
                    <span>Learn rate</span>
                    <input
                      type="number"
                      min="0.005"
                      max="0.2"
                      step="0.005"
                      value={learningRate}
                      onChange={(event) => setLearningRate(Number(event.target.value))}
                    />
                  </label>
                  <label className="field compact-field">
                    <span>Batch size</span>
                    <input type="number" min="5" max="200" step="5" value={batchSize} onChange={(event) => setBatchSize(Number(event.target.value))} />
                  </label>
                  <label className="field compact-field">
                    <span>CD steps</span>
                    <input type="number" min="1" max="8" step="1" value={cdSteps} onChange={(event) => setCdSteps(Number(event.target.value))} />
                  </label>
                  <label className="field compact-field">
                    <span>Momentum</span>
                    <input type="number" min="0" max="0.95" step="0.01" value={momentum} onChange={(event) => setMomentum(Number(event.target.value))} />
                  </label>
                  <label className="field compact-field">
                    <span>Weight decay</span>
                    <input
                      type="number"
                      min="0"
                      max="0.005"
                      step="0.00005"
                      value={weightDecay}
                      onChange={(event) => setWeightDecay(Number(event.target.value))}
                    />
                  </label>
                </div>
                <p className="control-strip-note">
                  Training uses minibatched CD-{cdSteps} with momentum and weight decay over{" "}
                  {datasetBundle?.trainingSampleCount ?? "the bundled"} balanced samples.
                </p>
              </section>

              <section className="rbm-training-execution">
                <div className="panel-header rbm-subpanel-header">
                  <h4>Training execution</h4>
                  <p>Run the trainer continuously, advance one epoch at a time, or reset and compare another configuration.</p>
                </div>
                <div className="control-actions rbm-training-actions">
                  {isTraining ? (
                    <button type="button" className="icon-btn" onClick={handlePauseTraining} disabled={!isTrainingConfigured}>
                      <Pause size={14} />
                      <span>Pause training</span>
                    </button>
                  ) : (
                    <button type="button" className="icon-btn primary" onClick={handleStartTraining} disabled={!isTrainingConfigured || trainerEpoch >= epochs}>
                      <Play size={14} />
                      <span>Start training</span>
                    </button>
                  )}
                  <button type="button" className="icon-btn" onClick={handleStepTraining} disabled={!isTrainingConfigured || isTraining || trainerEpoch >= epochs}>
                    <SkipForward size={14} />
                    <span>Step epoch</span>
                  </button>
                  <button type="button" className="icon-btn" onClick={handleResetTraining} disabled={!isTrainingConfigured}>
                    <RotateCcw size={14} />
                    <span>Reset trainer</span>
                  </button>
                </div>
                <div className="rbm-training-progress">
                  <div className="rbm-training-progress-copy">
                    <strong>Epoch {trainerEpoch} / {epochs}</strong>
                    <span>{isTraining ? "Training is running. Watch the metrics and plots update after each epoch." : "Use Step epoch for a slow, inspectable training walkthrough."}</span>
                  </div>
                  <div className="rbm-training-progress-bar" aria-hidden="true">
                    <div className="rbm-training-progress-fill" style={{ width: `${trainingProgress}%` }} />
                  </div>
                </div>
              </section>
            </div>
            <div className="rbm-training-metrics">
              <div className="rbm-training-stat">
                <span>Training samples</span>
                <strong>{datasetBundle?.trainingSampleCount ?? "n/a"}</strong>
              </div>
              <div className="rbm-training-stat">
                <span>Current epoch</span>
                <strong>{trainerEpoch}</strong>
              </div>
              <div className="rbm-training-stat">
                <span>Backend</span>
                <strong>{backendKind}</strong>
              </div>
              <div className="rbm-training-stat">
                <span>Epochs</span>
                <strong>{trainingHistory.length || epochs}</strong>
              </div>
              <div className="rbm-training-stat">
                <span>Final recon. error</span>
                <strong>{latestTrainingMetrics ? latestTrainingMetrics.reconstructionError.toFixed(4) : "n/a"}</strong>
              </div>
              <div className="rbm-training-stat">
                <span>Contrastive gap</span>
                <strong>{latestTrainingMetrics ? latestTrainingMetrics.contrastiveGap.toFixed(4) : "n/a"}</strong>
              </div>
              <div className="rbm-training-stat">
                <span>Mean hidden act.</span>
                <strong>{latestTrainingMetrics ? latestTrainingMetrics.hiddenActivation.toFixed(3) : "n/a"}</strong>
              </div>
              <div className="rbm-training-stat">
                <span>Mean |W|</span>
                <strong>{latestTrainingMetrics ? latestTrainingMetrics.weightMeanAbs.toFixed(4) : "n/a"}</strong>
              </div>
              <div className="rbm-training-stat">
                <span>Avg free energy</span>
                <strong>{latestTrainingMetrics ? latestTrainingMetrics.freeEnergy.toFixed(3) : "n/a"}</strong>
              </div>
              <div className="rbm-training-stat">
                <span>Train error</span>
                <strong>{trainingError.toFixed(4)}</strong>
              </div>
            </div>
            <div className="rbm-training-plots">
              <EnergyPlot
                values={trainingErrorSeries}
                title="Reconstruction Error"
                caption="Average visible reconstruction error after each training epoch."
                xLabel="epoch"
                yLabel="error"
                width={320}
                height={140}
              />
              <EnergyPlot
                values={contrastiveGapSeries}
                title="Contrastive Gap"
                caption="Mean data-vs-reconstruction deviation accumulated during contrastive divergence."
                xLabel="epoch"
                yLabel="gap"
                width={320}
                height={140}
              />
              <EnergyPlot
                values={hiddenActivationSeries}
                title="Hidden Activity"
                caption="Average hidden probability. Low values mean dead units; high values mean saturation."
                xLabel="epoch"
                yLabel="activity"
                width={320}
                height={140}
              />
              <EnergyPlot
                values={weightMeanAbsSeries}
                title="Weight Magnitude"
                caption="Mean absolute hidden-to-visible weight size over training."
                xLabel="epoch"
                yLabel="|W|"
                width={320}
                height={140}
              />
            </div>
          </section>

          <section className="panel">
            <div className="panel-header">
              <h3>Query and Gibbs playback</h3>
              <p>After training, apply a visible query and inspect how the Gibbs chain reconstructs or drifts over successive hidden-visible updates.</p>
            </div>
            <div className="convergence-strip">
              <div className="convergence-strip-main">
                <div className="convergence-fields">
                  <div className="rule-config-grid rbm-playback-grid">
                    <label className="field compact-field">
                      <span>Speed</span>
                      <input type="range" min="1" max="20" value={speed} onChange={(event) => setSpeed(Number(event.target.value))} />
                      <strong className="range-value">{speed} steps/s</strong>
                    </label>
                    <label className="field compact-field">
                      <span>Steps</span>
                      <input type="number" min="1" max="120" step="1" value={maxPlaybackSteps} onChange={(event) => setMaxPlaybackSteps(Number(event.target.value))} />
                    </label>
                  </div>
                  <p className="control-strip-note rbm-gibbs-note">
                    Apply initializes the visible query, then each Gibbs step alternates hidden sampling and visible reconstruction. Lower speed makes the trajectory readable; higher step count lets the chain mix longer before stopping.
                  </p>
                </div>
                <div className="control-actions">
                  <button type="button" className="icon-btn primary" onClick={applyQuery} disabled={!isReady || isTraining} title="Apply query">
                    <SkipForward size={14} />
                    <span>Apply</span>
                  </button>
                  {isPlaying ? (
                    <button type="button" className="icon-btn" onClick={handlePause} title="Pause">
                      <Pause size={14} />
                      <span>Pause</span>
                    </button>
                  ) : (
                    <button type="button" className="icon-btn" onClick={handlePlay} disabled={!isReady || isTraining} title="Play">
                      <Play size={14} />
                      <span>Play</span>
                    </button>
                  )}
                  <button type="button" className="icon-btn" onClick={handleStep} disabled={!isReady || isPlaying || isTraining} title="Step">
                    <SkipForward size={14} />
                    <span>Step</span>
                  </button>
                  <button type="button" className="icon-btn" onClick={handleReset} disabled={!isReady || isTraining} title="Reset">
                    <RotateCcw size={14} />
                    <span>Reset</span>
                  </button>
                </div>
              </div>
            </div>
          </section>

          <section className="rbm-canvas-row">
            <section className="panel input-panel">
              <div className="panel-header">
                <h3>Query</h3>
                <p>
                  {visibleModel === "bernoulli"
                    ? "Pick a stored exemplar, degrade it if needed, then edit the binary visible query before Gibbs playback."
                    : "Pick a stored exemplar, degrade it if needed, then edit the grayscale visible query before Gibbs playback."}
                </p>
              </div>
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
                        title={visibleModel === "bernoulli" ? "Flip a percentage of the selected pixels before loading the query" : "Invert intensity on a percentage of the selected pixels before loading the query"}
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
                  <GrayscaleCanvas pattern={queryPattern} onChange={handlePatternChange} paintValue={visibleModel === "gaussian" ? brushValue : 1} />
                  <div className="query-actions">
                    <button type="button" onClick={handleClear} title="Clear the query editor">
                      clear
                    </button>
                  </div>
                  {visibleModel === "gaussian" ? (
                    <label className="field compact-field gaussian-brush-field">
                      <span>Brush</span>
                      <input
                        type="range"
                        min="0.1"
                        max="1"
                        step="0.05"
                        value={brushValue}
                        onChange={(event) => setBrushValue(Number(event.target.value))}
                        title="Painting intensity used in the query editor"
                      />
                      <strong className="range-value">{brushValue.toFixed(2)}</strong>
                    </label>
                  ) : null}
                </div>
              </div>
            </section>

            <GrayscaleHeatmap
              title="Current reconstruction"
              data={reconstruction}
              side={PATTERN_SIDE}
              scale={9}
              caption={
                snapshot
                  ? `Step ${snapshot.step} • mean absolute reconstruction error ${snapshot.reconstructionError.toFixed(4)}`
                  : "The reconstruction produced by the current hidden sample."
              }
            />

            {hiddenGrid ? (
              <ValueGridHeatmap
                title="Hidden activations"
                data={hiddenGrid}
                side={hiddenGridSide}
                maxAbs={1}
                scale={18}
                caption="Sigmoid probabilities for the hidden layer on the current visible pattern."
                xLabel="hidden x"
                yLabel="hidden y"
              />
            ) : null}
          </section>

          {model ? (
            <div className="rbm-matrix-scroll">
              <MatrixHeatmap
                title="Hidden-to-visible weights"
                data={model.weights}
                rows={model.hiddenUnits}
                columns={model.visibleUnits}
                maxAbs={weightMaxAbs}
                caption="Every bipartite RBM weight is shown below. Rows are hidden units and columns are visible pixels."
                xLabel="visible pixel i"
                yLabel="hidden unit j"
              />
            </div>
          ) : null}
        </div>

        <div className="rbm-stage-side">
          <section className="panel">
            <div className="panel-header">
              <h3>Run state</h3>
              <p>Live summary of the current RBM reconstruction trajectory, visible-unit model, and training fit.</p>
            </div>
            <dl className="run-stats">
              <div>
                <dt>Matched exemplar</dt>
                <dd>{snapshot && snapshot.matchedPatternIndex >= 0 ? labels[snapshot.matchedPatternIndex] : "n/a"}</dd>
              </div>
              <div>
                <dt>Visible model</dt>
                <dd>{visibleModel === "bernoulli" ? "Bernoulli" : "Gaussian"}</dd>
              </div>
              <div>
                <dt>Free energy</dt>
                <dd>{snapshot ? snapshot.freeEnergy.toFixed(3) : "n/a"}</dd>
              </div>
              <div>
                <dt>Recon. error</dt>
                <dd>{snapshot ? snapshot.reconstructionError.toFixed(4) : "n/a"}</dd>
              </div>
              <div>
                <dt>Train error</dt>
                <dd>{trainingError.toFixed(4)}</dd>
              </div>
              <div>
                <dt>Current step</dt>
                <dd>{snapshot ? snapshot.step : 0}</dd>
              </div>
              <div>
                <dt>Active hidden mean</dt>
                <dd>
                  {hiddenSummary.length > 0
                    ? (hiddenSummary.reduce((total, value) => total + value, 0) / hiddenSummary.length).toFixed(3)
                    : "n/a"}
                </dd>
              </div>
            </dl>
          </section>
          <EnergyPlot
            values={energyHistory}
            title="Free Energy"
            caption="Compact Gibbs playback trace over the current query."
            xLabel="step"
            yLabel="energy"
            width={320}
            height={140}
          />
        </div>
      </div>

      <div className="rbm-gallery-grid">
        {featureMaps.length > 0 ? <FeatureGallery features={featureMaps} activeHiddenIndex={activeHiddenIndex} /> : null}
      </div>

      {showDatasetHelp && datasetBundle ? (
        <DatasetDialog title="Dataset" summary={datasetBundle.description} facts={datasetBundle.datasetFacts} onClose={() => setShowDatasetHelp(false)}>
          <MemoryGallery labels={datasetBundle.memoryLabels} patterns={displayedPatterns} matchedIndex={snapshot?.matchedPatternIndex ?? -1} />
        </DatasetDialog>
      ) : null}

      {showHelp ? <HelpDialog visibleModel={visibleModel} onClose={() => setShowHelp(false)} /> : null}
    </div>
  );
}
