import { useEffect, useRef, useState } from "react";
import { Pause, Play, RotateCcw, SkipForward } from "lucide-react";

import type { HopfieldSnapshot, UpdateRule } from "../core/hopfield";
import {
  clonePattern,
  createBlankPattern,
  getPatternSetById,
  PATTERN_SETS,
  PATTERN_SIDE,
  type PatternSetDefinition,
} from "../core/patternSets";
import type { WorkerRequest, WorkerResponse } from "../core/workerProtocol";
import { ControlPanel } from "../features/hopfield/ControlPanel";
import { EnergyPlot } from "../features/hopfield/EnergyPlot";
import { ValueGridHeatmap, WeightHeatmap } from "../features/hopfield/HeatmapCanvas";
import { PatternCanvas } from "../features/hopfield/PatternCanvas";
import { PatternGallery } from "../features/hopfield/PatternGallery";

type ConvergenceRule = "async-random";

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

export default function HopfieldNetworkPage() {
  const [patternSetId, setPatternSetId] = useState(PATTERN_SETS[0].id);
  const [patternSet, setPatternSet] = useState<PatternSetDefinition>(() => getPatternSetById(PATTERN_SETS[0].id));
  const [updateRule] = useState<UpdateRule>("hebbian");
  const [convergenceRule] = useState<ConvergenceRule>("async-random");
  const [queryPattern, setQueryPattern] = useState<Int8Array>(() => createBlankPattern());
  const [snapshot, setSnapshot] = useState<HopfieldSnapshot>(() => createSnapshot(createBlankPattern()));
  const [weights, setWeights] = useState<Float32Array>(() => new Float32Array(PATTERN_SIDE * PATTERN_SIDE * PATTERN_SIDE * PATTERN_SIDE));
  const [maxWeightAbs, setMaxWeightAbs] = useState(1);
  const [energyHistory, setEnergyHistory] = useState<number[]>([]);
  const [speed, setSpeed] = useState(8);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [workerError, setWorkerError] = useState<string | null>(null);

  const hasAppliedQueryRef = useRef(false);
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
    const worker = workerRef.current;
    if (!worker) {
      return;
    }

    const nextPatternSet = getPatternSetById(patternSetId);
    setPatternSet(nextPatternSet);
    setQueryPattern(createBlankPattern());
    setEnergyHistory([]);
    setSnapshot(createSnapshot(createBlankPattern()));
    setIsReady(false);
    hasAppliedQueryRef.current = false;

    worker.postMessage({
      type: "initialize",
      patternSetId,
      updateRule,
    } satisfies WorkerRequest);
  }, [patternSetId, updateRule]);

  function applyQuery(): void {
    const worker = workerRef.current;
    if (!worker) {
      return;
    }
    hasAppliedQueryRef.current = true;
    const outgoing = queryPattern.slice();
    worker.postMessage({ type: "setQuery", pattern: outgoing } satisfies WorkerRequest, [outgoing.buffer]);
  }

  function handlePlay(): void {
    const worker = workerRef.current;
    if (!worker) {
      return;
    }
    if (!hasAppliedQueryRef.current) {
      applyQuery();
    }
    setIsPlaying(true);
    worker.postMessage({
      type: "play",
      intervalMs: Math.max(40, Math.round(1000 / speed)),
      maxSteps: 80,
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
    workerRef.current?.postMessage({ type: "step" } satisfies WorkerRequest);
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
    const next = clonePattern(patternSet.patterns[index]);
    setQueryPattern(next);
    hasAppliedQueryRef.current = false;
  }

  function handlePatternChange(nextPattern: Int8Array): void {
    setQueryPattern(nextPattern);
    hasAppliedQueryRef.current = false;
  }

  return (
    <div className="page-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Browser-only implementation</p>
          <h1>Hopfield Network</h1>
          <p className="hero-copy">
            Typed-array Hopfield core, worker-driven convergence, editable 28x28 input, live neuron-state heatmap,
            and a full 784x784 connection matrix rendered in the client.
          </p>
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
        <div className="architecture-field">
          <span>Pattern set</span>
          <select value={patternSetId} onChange={(event) => setPatternSetId(event.target.value)}>
            {PATTERN_SETS.map((entry) => (
              <option key={entry.id} value={entry.id}>
                {entry.name}
              </option>
            ))}
          </select>
        </div>
        <div className="architecture-field">
          <span>Update rule</span>
          <select value={updateRule} disabled>
            <option value="hebbian">Hebbian</option>
          </select>
        </div>
        <div className="architecture-field">
          <span>Convergence rule</span>
          <select value={convergenceRule} disabled>
            <option value="async-random">Async random sweep</option>
          </select>
        </div>
        <div className="architecture-actions">
          <span>Run controls</span>
          <div className="top-actions">
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
      </section>

      <div className="dashboard-grid">
        <div className="left-column">
          <ControlPanel
            speed={speed}
            onSpeedChange={setSpeed}
            labels={patternSet.labels}
            onLoadPattern={handleLoadPattern}
          />
          <PatternGallery patternSet={patternSet} matchedIndex={snapshot.matchedPatternIndex} />
        </div>

        <div className="main-column">
          <section className="center-row">
            <section className="panel input-panel">
              <div className="panel-header">
                <h3>Query editor</h3>
                <p>Draw directly on the 28x28 lattice, then apply the pattern and watch convergence sweep by sweep.</p>
              </div>
              <div className="input-grid">
                <PatternCanvas pattern={queryPattern} onChange={handlePatternChange} />
                <div className="query-actions">
                  <button type="button" onClick={handleClear}>
                    clear
                  </button>
                </div>
              </div>
            </section>

            <WeightHeatmap
              title="Connectome heatmap"
              data={weights}
              side={PATTERN_SIDE * PATTERN_SIDE}
              maxAbs={maxWeightAbs}
              caption="Dense Hebbian connection matrix across all 784 neurons."
            />
          </section>

        </div>

        <div className="right-column">
          <ValueGridHeatmap
            title="Current neuron state"
            data={snapshot.state}
            side={PATTERN_SIDE}
            maxAbs={1}
            caption={`Step ${snapshot.step} • ${snapshot.changedCount} neuron flips in last sweep`}
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
    </div>
  );
}
