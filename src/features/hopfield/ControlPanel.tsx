interface ControlPanelProps {
  labels: string[];
  onLoadPattern: (patternIndex: number) => void;
  corruptionLevel: number;
  onCorruptionLevelChange: (value: number) => void;
  obfuscationLevel: number;
  onObfuscationLevelChange: (value: number) => void;
}

export function ControlPanel({
  labels,
  onLoadPattern,
  corruptionLevel,
  onCorruptionLevelChange,
  obfuscationLevel,
  onObfuscationLevelChange,
}: ControlPanelProps) {
  return (
    <section className="panel control-panel">
      <div className="panel-header">
        <h3>Memories</h3>
        <p>Load a stored pattern as a starting point or sketch your own query in the center panel.</p>
      </div>

      <div className="field">
        <span>Corruption</span>
        <input
          type="range"
          min="0"
          max="100"
          value={corruptionLevel}
          onChange={(event) => onCorruptionLevelChange(Number(event.target.value))}
        />
        <strong className="range-value">{corruptionLevel}% flip bits</strong>
      </div>

      <div className="field">
        <span>Obfuscation</span>
        <input
          type="range"
          min="0"
          max="100"
          value={obfuscationLevel}
          onChange={(event) => onObfuscationLevelChange(Number(event.target.value))}
        />
        <strong className="range-value">{obfuscationLevel}% set to zero</strong>
      </div>

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
