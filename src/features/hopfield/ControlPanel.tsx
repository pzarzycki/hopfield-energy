interface ControlPanelProps {
  speed: number;
  onSpeedChange: (speed: number) => void;
  labels: string[];
  onLoadPattern: (patternIndex: number) => void;
}

export function ControlPanel({
  speed,
  onSpeedChange,
  labels,
  onLoadPattern,
}: ControlPanelProps) {
  return (
    <section className="panel control-panel">
      <div className="panel-header">
        <h3>Memories</h3>
        <p>Load a stored pattern as a starting point or sketch your own query in the center panel.</p>
      </div>

      <label className="field">
        <span>Playback speed</span>
        <input
          type="range"
          min="1"
          max="20"
          value={speed}
          onChange={(event) => onSpeedChange(Number(event.target.value))}
        />
        <strong>{speed} sweeps/s</strong>
      </label>

      <div className="field">
        <span>Stored patterns</span>
        <div className="pattern-picker">
          {labels.map((label, index) => (
            <button key={label} type="button" onClick={() => onLoadPattern(index)}>
              {label}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}
