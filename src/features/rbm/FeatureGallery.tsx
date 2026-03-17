import type { RBMFeatureMap } from "../../core/rbm";
import { PATTERN_SIDE } from "../../core/patternSets";
import { GrayscaleHeatmap } from "../hopfield/HeatmapCanvas";

interface FeatureGalleryProps {
  features: RBMFeatureMap[];
  activeHiddenIndex: number;
}

export function FeatureGallery({ features, activeHiddenIndex }: FeatureGalleryProps) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h3>All learned features</h3>
        <p>Every hidden-unit receptive field ranked by average absolute weight magnitude. The highlighted tile matches the most active hidden unit for the current query.</p>
      </div>
      <div className="gallery-grid gallery-grid--features">
        {features.map((feature) => (
          <div key={`feature-${feature.hiddenIndex}`} className={`gallery-item${activeHiddenIndex === feature.hiddenIndex ? " is-matched" : ""}`}>
            <GrayscaleHeatmap
              title={`h${feature.hiddenIndex + 1}`}
              data={feature.map}
              side={PATTERN_SIDE}
              scale={3}
              caption="dark = positive support"
              xLabel=""
              yLabel=""
            />
          </div>
        ))}
      </div>
    </section>
  );
}
