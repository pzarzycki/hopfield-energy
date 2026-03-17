import { PATTERN_SIDE } from "../../core/patternSets";
import { GrayscaleHeatmap } from "../hopfield/HeatmapCanvas";

interface MemoryGalleryProps {
  labels: string[];
  patterns: Float32Array[];
  matchedIndex: number;
}

export function MemoryGallery({ labels, patterns, matchedIndex }: MemoryGalleryProps) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h3>Stored memories</h3>
      </div>
      <div className="gallery-grid">
        {patterns.map((pattern, index) => (
          <div key={labels[index]} className={`gallery-item${matchedIndex === index ? " is-matched" : ""}`}>
            <GrayscaleHeatmap title={labels[index]} data={pattern} side={PATTERN_SIDE} scale={3} xLabel="" yLabel="" />
          </div>
        ))}
      </div>
    </section>
  );
}
