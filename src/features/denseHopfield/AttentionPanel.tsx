interface AttentionPanelProps {
  labels: string[];
  attention: Float32Array;
  matchedIndex: number;
}

export function AttentionPanel({ labels, attention, matchedIndex }: AttentionPanelProps) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h3>Memory attention</h3>
        <p>Softmax weights over stored memories during the current retrieval step.</p>
      </div>
      <div className="attention-list">
        {labels.map((label, index) => {
          const value = attention[index] ?? 0;
          return (
            <div key={label} className={`attention-item${matchedIndex === index ? " is-active" : ""}`}>
              <span className="attention-item__label">{label}</span>
              <div className="attention-item__bar">
                <div className="attention-item__fill" style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} />
              </div>
              <strong>{(value * 100).toFixed(1)}%</strong>
            </div>
          );
        })}
      </div>
    </section>
  );
}
