import { useEffect, useMemo, useRef, useState } from "react";
import { CircleHelp, Pause, Play, RotateCcw, SkipForward, X } from "lucide-react";
import { BlockMath, InlineMath } from "react-katex";

import { createDenseSnapshot, type DenseSnapshot } from "../core/denseHopfield";
import type { DenseHopfieldMemoryView, DenseHopfieldWorkerRequest, DenseHopfieldWorkerResponse } from "../core/denseHopfieldWorkerProtocol";
import { formatPrimaryTasks, getModelCatalogEntry } from "../core/modelCatalog";
import { PATTERN_SIDE, PATTERN_SIZE } from "../core/patternSets";
import { loadDatasetArchive, summarizeDatasetArchive, type DatasetName } from "../data/datasetArchives";
import { EnergyPlot } from "../features/hopfield/EnergyPlot";
import { GrayscaleHeatmap, WeightHeatmap } from "../features/hopfield/HeatmapCanvas";
import { GrayscaleCanvas } from "../features/denseHopfield/GrayscaleCanvas";
import { MemoryGallery } from "../features/denseHopfield/MemoryGallery";
import { AttentionPanel } from "../features/denseHopfield/AttentionPanel";
import { DatasetDialog } from "../features/common/DatasetDialog";

function renderTextWithMath(text: string) {
  const segments = text.split(/(\$[^$]+\$)/g).filter(Boolean);
  return segments.map((segment, index) => {
    if (segment.startsWith("$") && segment.endsWith("$")) {
      return <InlineMath key={`${segment}-${index}`}>{segment.slice(1, -1)}</InlineMath>;
    }
    return <span key={`${segment}-${index}`}>{segment}</span>;
  });
}

function createBlankPattern(): Float32Array {
  return new Float32Array(PATTERN_SIZE);
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

function applyPatternNoise(pattern: Float32Array, corruptionPercent: number, obfuscationPercent: number): Float32Array {
  const next = pattern.slice();
  const total = next.length;
  const corruptionCount = Math.round((total * corruptionPercent) / 100);
  const obfuscationCount = Math.round((total * obfuscationPercent) / 100);

  if (corruptionCount > 0) {
    const corruptionIndices = shuffleIndices(total);
    for (let index = 0; index < corruptionCount; index += 1) {
      const target = corruptionIndices[index];
      next[target] = next[target] > 0.5 ? 0 : 1;
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

function HelpDialog({ onClose }: { onClose: () => void }) {
  const formula = String.raw`x' = X^{T}\,\mathrm{softmax}\!\left(\beta Xx\right), \qquad
E(x)=\frac{1}{2}\lVert x \rVert^2 - \frac{1}{\beta}\log\sum_i \exp(\beta x_i^{T}x)`;

  return (
    <div className="modal-backdrop" onClick={onClose} role="presentation">
      <section
        className="modal-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="dense-help-dialog-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-header">
          <div>
            <h2 id="dense-help-dialog-title">Dense Hopfield Retrieval</h2>
            <p>Continuous-state retrieval implemented as attention over stored memories.</p>
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
              <p>{renderTextWithMath("The query $x$ is compared against every stored memory in $X$, scaled by inverse temperature $\\beta$, and normalized with a softmax.")}</p>
              <p>{renderTextWithMath("The next state is a weighted average of memories, so retrieval stays continuous and can express mixtures before it settles on a dominant attractor.")}</p>
              <p>{renderTextWithMath("Higher $\\beta$ makes attention sharper. Lower $\\beta$ spreads probability mass across multiple memories and produces softer reconstructions.")}</p>
            </div>
          </section>
          <section className="modal-section">
            <span className="modal-section-label">Notes</span>
            <ul className="modal-list modal-list--notes">
              <li>{renderTextWithMath("This page uses real grayscale MNIST and Fashion-MNIST exemplars as the memory source.")}</li>
              <li>{renderTextWithMath("Each stored memory is one exemplar per class, so the page emphasizes associative retrieval dynamics rather than large-capacity storage.")}</li>
              <li>{renderTextWithMath("The reconstruction remains grayscale because the stored memories themselves are continuous intensity images in $[0,1]$.")}</li>
            </ul>
          </section>
        </div>
      </section>
    </div>
  );
}

export default function DenseHopfieldNetworkPage() {
  const modelEntry = getModelCatalogEntry("dense-hopfield");
  const [datasetName, setDatasetName] = useState<DatasetName>("mnist");
  const [memorySet, setMemorySet] = useState<DenseHopfieldMemoryView | null>(null);
  const [queryPattern, setQueryPattern] = useState<Float32Array>(() => createBlankPattern());
  const [snapshot, setSnapshot] = useState<DenseSnapshot>(() => createDenseSnapshot(createBlankPattern()));
  const [energyHistory, setEnergyHistory] = useState<number[]>([]);
  const [beta, setBeta] = useState(8);
  const [speed, setSpeed] = useState(8);
  const [maxPlaybackSteps, setMaxPlaybackSteps] = useState(40);
  const [corruptionLevel, setCorruptionLevel] = useState(0);
  const [obfuscationLevel, setObfuscationLevel] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [showHelp, setShowHelp] = useState(false);
  const [showDatasetHelp, setShowDatasetHelp] = useState(false);
  const [datasetFacts, setDatasetFacts] = useState<string[]>([]);

  const hasAppliedQueryRef = useRef(false);
  const workerRef = useRef<Worker | null>(null);

  useEffect(() => {
    const worker = new Worker(new URL("../workers/denseHopfield.worker.ts", import.meta.url), { type: "module" });
    workerRef.current = worker;

    const handleMessage = (event: MessageEvent<DenseHopfieldWorkerResponse>) => {
      const message = event.data;

      if (message.type === "initialized") {
        setMemorySet(message.memorySet);
        setSnapshot(message.snapshot);
        setEnergyHistory([message.snapshot.energy]);
        setIsReady(true);
        setIsPlaying(false);
        setErrorMessage(null);
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
        setErrorMessage(message.message);
        setIsPlaying(false);
      }
    };

    worker.addEventListener("message", handleMessage);
    setIsReady(false);
    setIsPlaying(false);
    setErrorMessage(null);
    setMemorySet(null);
    setQueryPattern(createBlankPattern());
    setSnapshot(createDenseSnapshot(createBlankPattern()));
    setEnergyHistory([]);
    hasAppliedQueryRef.current = false;
    worker.postMessage({ type: "initialize", datasetName, beta } satisfies DenseHopfieldWorkerRequest);

    return () => {
      worker.removeEventListener("message", handleMessage);
      worker.terminate();
      if (workerRef.current === worker) {
        workerRef.current = null;
      }
    };
  }, [datasetName]);

  useEffect(() => {
    let cancelled = false;

    void loadDatasetArchive(datasetName)
      .then((archive) => {
        if (cancelled) {
          return;
        }
        const stats = summarizeDatasetArchive(archive);
        const perClassLabel =
          stats.minSamplesPerClass === stats.maxSamplesPerClass
            ? `${stats.minSamplesPerClass} per class`
            : `${stats.minSamplesPerClass}-${stats.maxSamplesPerClass} per class`;
        setDatasetFacts([`${stats.sampleCount} samples`, `${stats.classCount} classes`, perClassLabel]);
      })
      .catch(() => {
        if (!cancelled) {
          setDatasetFacts([]);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [datasetName]);

  useEffect(() => {
    if (!isReady) {
      return;
    }
    setIsPlaying(false);
    hasAppliedQueryRef.current = false;
    workerRef.current?.postMessage({ type: "setBeta", beta } satisfies DenseHopfieldWorkerRequest);
  }, [beta]);

  const affinityMaxAbs = useMemo(() => memorySet?.maxSimilarityAbs ?? 1, [memorySet]);

  function setQueryOnWorker(nextState: Float32Array = queryPattern): Float32Array {
    const worker = workerRef.current;
    const normalized = nextState.slice();
    setQueryPattern(normalized);
    if (worker) {
      const outgoing = normalized.slice();
      worker.postMessage({ type: "setQuery", pattern: outgoing } satisfies DenseHopfieldWorkerRequest, [outgoing.buffer]);
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
      type: "play",
      intervalMs: Math.max(40, Math.round(1000 / speed)),
      maxSteps: Math.max(1, maxPlaybackSteps),
    } satisfies DenseHopfieldWorkerRequest);
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
      type: "play",
      intervalMs: Math.max(40, Math.round(1000 / speed)),
      maxSteps: Math.max(1, maxPlaybackSteps),
    } satisfies DenseHopfieldWorkerRequest);
  }

  function handlePause(): void {
    workerRef.current?.postMessage({ type: "pause" } satisfies DenseHopfieldWorkerRequest);
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
    worker.postMessage({ type: "step" } satisfies DenseHopfieldWorkerRequest);
  }

  function handleReset(): void {
    setIsPlaying(false);
    workerRef.current?.postMessage({ type: "reset" } satisfies DenseHopfieldWorkerRequest);
  }

  function handleClear(): void {
    const blank = createBlankPattern();
    setIsPlaying(false);
    hasAppliedQueryRef.current = false;
    setQueryOnWorker(blank);
    hasAppliedQueryRef.current = false;
  }

  function handleLoadPattern(index: number): void {
    if (!memorySet) {
      return;
    }
    const next = applyPatternNoise(memorySet.patterns[index], corruptionLevel, obfuscationLevel);
    setIsPlaying(false);
    setQueryPattern(next);
    hasAppliedQueryRef.current = false;
  }

  function handlePatternChange(nextPattern: Float32Array): void {
    setIsPlaying(false);
    setQueryPattern(nextPattern);
    hasAppliedQueryRef.current = false;
  }

  const labels = memorySet?.labels ?? [];
  const weightCount = memorySet ? memorySet.patterns.length * PATTERN_SIZE : 0;

  return (
    <div className="page-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Wasm-backed worker runtime</p>
          <h1>Dense Hopfield Network</h1>
          <p className="hero-copy">
            Continuous-state retrieval over bundled MNIST and Fashion-MNIST memories, with editable 28x28 queries,
            softmax attention dynamics, live reconstruction, and an in-browser memory affinity map.
          </p>
          <p className="hero-task">Primary task: {formatPrimaryTasks(modelEntry.primaryTasks)}</p>
        </div>
        <div className="hero-stats">
          <div className="stat-card">
            <span>Pixels</span>
            <strong>{PATTERN_SIZE}</strong>
          </div>
          <div className="stat-card">
            <span>Stored values</span>
            <strong>{weightCount.toLocaleString()}</strong>
          </div>
          <div className="stat-card">
            <span>Status</span>
            <strong>{snapshot.converged ? "stable" : isPlaying ? "running" : isReady ? "ready" : "loading"}</strong>
          </div>
        </div>
      </header>

      {errorMessage ? <div className="error-banner">{errorMessage}</div> : null}

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
          <p className="control-strip-note">{memorySet?.description ?? "Loading dataset exemplars..."}</p>
        </div>

        <div className="control-strip-group">
          <div className="control-strip-header">
            <span className="control-strip-title">Association control</span>
            <button
              type="button"
              className="help-btn"
              aria-expanded={showHelp}
              onClick={() => setShowHelp((current) => !current)}
              title="Dense Hopfield help"
            >
              <CircleHelp size={15} />
            </button>
          </div>
          <label className="field compact-field">
            <span>Inverse temperature</span>
            <input type="range" min="1" max="24" value={beta} onChange={(event) => setBeta(Number(event.target.value))} />
            <strong className="range-value">beta = {beta}</strong>
          </label>
        </div>

        <div className="control-strip-group control-strip-group--wide">
          <div className="control-strip-header">
            <span className="control-strip-title">Retrieval control</span>
            <span className="control-strip-spacer" aria-hidden="true" />
          </div>
          <div className="convergence-strip">
            <div className="convergence-strip-main">
              <div className="convergence-fields">
                <div className="rule-config-grid">
                  <label className="field compact-field">
                    <span>Speed</span>
                    <input type="range" min="1" max="20" value={speed} onChange={(event) => setSpeed(Number(event.target.value))} />
                    <strong className="range-value">{speed} steps/s</strong>
                  </label>
                  <label className="field compact-field">
                    <span>Steps</span>
                    <input
                      type="number"
                      min="1"
                      max="120"
                      step="1"
                      value={maxPlaybackSteps}
                      onChange={(event) => setMaxPlaybackSteps(Number(event.target.value))}
                    />
                  </label>
                </div>
              </div>
              <div className="control-actions">
                <button type="button" className="icon-btn primary" onClick={applyQuery} disabled={!isReady} title="Apply query">
                  <SkipForward size={14} />
                  <span>Apply</span>
                </button>
                {isPlaying ? (
                  <button type="button" className="icon-btn" onClick={handlePause} title="Pause">
                    <Pause size={14} />
                    <span>Pause</span>
                  </button>
                ) : (
                  <button type="button" className="icon-btn" onClick={handlePlay} disabled={!isReady} title="Play">
                    <Play size={14} />
                    <span>Play</span>
                  </button>
                )}
                <button type="button" className="icon-btn" onClick={handleStep} disabled={!isReady || isPlaying} title="Step">
                  <SkipForward size={14} />
                  <span>Step</span>
                </button>
                <button type="button" className="icon-btn" onClick={handleReset} disabled={!isReady} title="Reset">
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
                <p>Pick a stored memory, degrade it if needed, then edit the query directly before running retrieval.</p>
              </div>
              <div className="query-workbench">
                <div className="query-toolbar">
                  <div className="field compact-field">
                    <span>Examples</span>
                    <div className="pattern-picker pattern-picker--compact">
                      {labels.map((label, index) => (
                        <button key={label} type="button" onClick={() => handleLoadPattern(index)} title={`Load ${label}`}>
                          {label}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="query-toolbar--row">
                    <label className="field compact-field">
                      <span>Corruption</span>
                      <input type="range" min="0" max="100" value={corruptionLevel} onChange={(event) => setCorruptionLevel(Number(event.target.value))} title="Invert intensity on a portion of the selected memory before loading it into the query editor" />
                      <strong className="range-value">{corruptionLevel}%</strong>
                    </label>
                    <label className="field compact-field">
                      <span>Obfuscation</span>
                      <input type="range" min="0" max="100" value={obfuscationLevel} onChange={(event) => setObfuscationLevel(Number(event.target.value))} title="Set part of the selected memory to zero before retrieval" />
                      <strong className="range-value">{obfuscationLevel}%</strong>
                    </label>
                  </div>
                </div>
                <div className="input-grid">
                  <GrayscaleCanvas pattern={queryPattern} onChange={handlePatternChange} />
                  <div className="query-actions">
                    <button type="button" onClick={handleClear} title="Clear the query editor">
                      clear
                    </button>
                  </div>
                </div>
              </div>
            </section>

            {memorySet ? (
              <WeightHeatmap
                title="Memory similarity"
                data={memorySet.similarityMatrix}
                side={memorySet.patterns.length}
                maxAbs={affinityMaxAbs}
                caption="Pairwise similarity among stored memories. This is not a neuron-to-neuron connection matrix."
                xLabel="memory j"
                yLabel="memory i"
              />
            ) : null}
          </section>
        </div>

        <div className="right-column">
          <GrayscaleHeatmap
            title="Current reconstruction"
            data={snapshot.state}
            side={PATTERN_SIDE}
            scale={8}
            caption={`Step ${snapshot.step} • mean absolute update ${snapshot.delta.toFixed(4)}`}
          />
          <AttentionPanel labels={labels} attention={snapshot.attention} matchedIndex={snapshot.matchedPatternIndex} />
          <EnergyPlot values={energyHistory} />
          <section className="panel">
            <div className="panel-header">
              <h3>Run state</h3>
              <p>Live summary of the current Dense Hopfield retrieval trajectory.</p>
            </div>
            <dl className="run-stats">
              <div>
                <dt>Matched memory</dt>
                <dd>{snapshot.matchedPatternIndex >= 0 ? labels[snapshot.matchedPatternIndex] : "n/a"}</dd>
              </div>
              <div>
                <dt>Top attention</dt>
                <dd>{(snapshot.topAttention * 100).toFixed(1)}%</dd>
              </div>
              <div>
                <dt>Energy</dt>
                <dd>{snapshot.energy.toFixed(3)}</dd>
              </div>
              <div>
                <dt>Entropy</dt>
                <dd>{snapshot.entropy.toFixed(3)}</dd>
              </div>
              <div>
                <dt>Dataset</dt>
                <dd>{datasetName === "mnist" ? "MNIST" : "Fashion"}</dd>
              </div>
              <div>
                <dt>Beta</dt>
                <dd>{beta}</dd>
              </div>
              <div>
                <dt>Step</dt>
                <dd>{snapshot.step}</dd>
              </div>
              <div>
                <dt>Stable</dt>
                <dd>{snapshot.converged ? "yes" : "no"}</dd>
              </div>
            </dl>
          </section>
        </div>
      </div>

      {showDatasetHelp && memorySet ? (
        <DatasetDialog
          title="Dataset"
          summary={memorySet.description}
          facts={datasetFacts}
          onClose={() => setShowDatasetHelp(false)}
        >
          <MemoryGallery labels={memorySet.labels} patterns={memorySet.patterns} matchedIndex={snapshot.matchedPatternIndex} />
        </DatasetDialog>
      ) : null}

      {showHelp ? <HelpDialog onClose={() => setShowHelp(false)} /> : null}
    </div>
  );
}
