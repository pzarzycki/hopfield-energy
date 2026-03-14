import { useEffect, useRef } from "react";

interface EnergyPlotProps {
  values: number[];
}

export function EnergyPlot({ values }: EnergyPlotProps) {
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

    const width = canvas.width;
    const height = canvas.height;
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#f7f9fe";
    context.fillRect(0, 0, width, height);

    context.strokeStyle = "rgba(62, 90, 148, 0.15)";
    context.lineWidth = 1;
    for (let row = 0; row < 4; row += 1) {
      const y = 16 + ((height - 32) * row) / 3;
      context.beginPath();
      context.moveTo(0, y);
      context.lineTo(width, y);
      context.stroke();
    }

    if (values.length === 0) {
      return;
    }

    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;

    context.strokeStyle = "#2458d3";
    context.lineWidth = 2;
    context.beginPath();

    values.forEach((value, index) => {
      const x = values.length === 1 ? width / 2 : (index / (values.length - 1)) * width;
      const y = height - 16 - ((value - min) / range) * (height - 32);
      if (index === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    });

    context.stroke();

    const last = values[values.length - 1];
    const lastX = values.length === 1 ? width / 2 : width;
    const lastY = height - 16 - ((last - min) / range) * (height - 32);
    context.fillStyle = "#d63a44";
    context.beginPath();
    context.arc(lastX, lastY, 4, 0, Math.PI * 2);
    context.fill();
  }, [values]);

  return (
    <section className="panel">
      <div className="panel-header">
        <h3>Energy</h3>
        <p>Compact convergence trace.</p>
      </div>
      <div className="axis-frame">
        <div className="axis-frame__body">
          <span className="axis-frame__y">energy</span>
          <div className="axis-frame__content">
            <canvas ref={canvasRef} width={280} height={120} className="energy-plot energy-plot--compact" />
          </div>
        </div>
        <span className="axis-frame__x">step</span>
      </div>
    </section>
  );
}
