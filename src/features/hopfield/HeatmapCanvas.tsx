import { useEffect, useMemo, useRef } from "react";

import { divergingColor, writePatternImage, writeWeightHeatmap } from "../../core/colorMaps";

interface PatternHeatmapProps {
  title: string;
  data: Int8Array | Uint8Array;
  side: number;
  scale?: number;
  caption?: string;
  xLabel?: string;
  yLabel?: string;
}

interface WeightHeatmapProps {
  title: string;
  data: Float32Array;
  side: number;
  maxAbs: number;
  caption?: string;
  xLabel?: string;
  yLabel?: string;
}

interface ValueGridHeatmapProps {
  title: string;
  data: Int8Array | Float32Array;
  side: number;
  maxAbs: number;
  scale?: number;
  caption?: string;
  xLabel?: string;
  yLabel?: string;
}

function AxisFrame({
  children,
  xLabel,
  yLabel,
}: {
  children: React.ReactNode;
  xLabel: string;
  yLabel: string;
}) {
  return (
    <div className="axis-frame">
      <div className="axis-frame__body">
        <span className="axis-frame__y">{yLabel}</span>
        <div className="axis-frame__content">{children}</div>
      </div>
      <span className="axis-frame__x">{xLabel}</span>
    </div>
  );
}

export function PatternHeatmap({
  title,
  data,
  side,
  scale = 10,
  caption,
  xLabel = "grid x",
  yLabel = "grid y",
}: PatternHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imageBuffer = useMemo(() => new Uint8ClampedArray(data.length * 4), [data]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    writePatternImage(imageBuffer, data);
    const imageData = new ImageData(imageBuffer, side, side);
    const bitmapCanvas = document.createElement("canvas");
    bitmapCanvas.width = side;
    bitmapCanvas.height = side;
    const bitmapContext = bitmapCanvas.getContext("2d");
    if (!bitmapContext) {
      return;
    }

    bitmapContext.putImageData(imageData, 0, 0);
    context.imageSmoothingEnabled = false;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(bitmapCanvas, 0, 0, canvas.width, canvas.height);
  }, [data, imageBuffer, side]);

  return (
    <section className="panel">
      <div className="panel-header">
        <h3>{title}</h3>
        {caption ? <p>{caption}</p> : null}
      </div>
      <AxisFrame xLabel={xLabel} yLabel={yLabel}>
        <canvas ref={canvasRef} width={side * scale} height={side * scale} className="heatmap-canvas pattern-heatmap" />
      </AxisFrame>
    </section>
  );
}

export function WeightHeatmap({
  title,
  data,
  side,
  maxAbs,
  caption,
  xLabel = "neuron j",
  yLabel = "neuron i",
}: WeightHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imageBuffer = useMemo(() => new Uint8ClampedArray(data.length * 4), [data]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    writeWeightHeatmap(imageBuffer, data, maxAbs);
    const imageData = new ImageData(imageBuffer, side, side);
    const bitmapCanvas = document.createElement("canvas");
    bitmapCanvas.width = side;
    bitmapCanvas.height = side;
    const bitmapContext = bitmapCanvas.getContext("2d");
    if (!bitmapContext) {
      return;
    }

    bitmapContext.putImageData(imageData, 0, 0);
    context.imageSmoothingEnabled = false;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(bitmapCanvas, 0, 0, canvas.width, canvas.height);
  }, [data, imageBuffer, maxAbs, side]);

  return (
    <section className="panel">
      <div className="panel-header">
        <h3>{title}</h3>
        {caption ? <p>{caption}</p> : null}
      </div>
      <AxisFrame xLabel={xLabel} yLabel={yLabel}>
        <canvas ref={canvasRef} width={side} height={side} className="heatmap-canvas weight-heatmap" />
      </AxisFrame>
      <div className="legend legend--weights">
        <span>negative</span>
        <span className="legend-bar" aria-hidden="true" />
        <span>positive</span>
      </div>
    </section>
  );
}

export function ValueGridHeatmap({
  title,
  data,
  side,
  maxAbs,
  scale = 8,
  caption,
  xLabel = "grid x",
  yLabel = "grid y",
}: ValueGridHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    context.clearRect(0, 0, canvas.width, canvas.height);

    for (let row = 0; row < side; row += 1) {
      for (let column = 0; column < side; column += 1) {
        const value = data[row * side + column];
        const [red, green, blue] = divergingColor(value, maxAbs);
        context.fillStyle = `rgb(${red} ${green} ${blue})`;
        context.fillRect(column * scale, row * scale, scale, scale);
      }
    }

    context.strokeStyle = "rgba(44, 62, 104, 0.18)";
    context.lineWidth = 1;
    for (let offset = 0; offset <= side; offset += 1) {
      const position = offset * scale;
      context.beginPath();
      context.moveTo(0, position);
      context.lineTo(canvas.width, position);
      context.stroke();

      context.beginPath();
      context.moveTo(position, 0);
      context.lineTo(position, canvas.height);
      context.stroke();
    }
  }, [data, maxAbs, scale, side]);

  return (
    <section className="panel">
      <div className="panel-header">
        <h3>{title}</h3>
        {caption ? <p>{caption}</p> : null}
      </div>
      <AxisFrame xLabel={xLabel} yLabel={yLabel}>
        <canvas ref={canvasRef} width={side * scale} height={side * scale} className="heatmap-canvas value-grid-heatmap" />
      </AxisFrame>
      <div className="legend legend--weights">
        <span>negative</span>
        <span className="legend-bar" aria-hidden="true" />
        <span>positive</span>
      </div>
    </section>
  );
}
