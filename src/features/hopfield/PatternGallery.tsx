import type { PatternSetDefinition } from "../../core/patternSets";
import { PATTERN_SIDE } from "../../core/patternSets";
import { PatternHeatmap } from "./HeatmapCanvas";

interface PatternGalleryProps {
  patternSet: PatternSetDefinition;
  matchedIndex: number;
}

export function PatternGallery({ patternSet, matchedIndex }: PatternGalleryProps) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h3>Stored memories</h3>
      </div>
      <div className="gallery-grid">
        {patternSet.patterns.map((pattern, index) => (
          <div key={patternSet.labels[index]} className={`gallery-item${matchedIndex === index ? " is-matched" : ""}`}>
            <PatternHeatmap title={patternSet.labels[index]} data={pattern} side={PATTERN_SIDE} scale={3} xLabel="" yLabel="" />
          </div>
        ))}
      </div>
    </section>
  );
}
