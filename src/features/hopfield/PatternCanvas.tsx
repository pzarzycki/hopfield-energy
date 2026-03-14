import { useEffect, useMemo, useRef } from "react";

import { PATTERN_SIDE } from "../../core/patternSets";

interface PatternCanvasProps {
  pattern: Int8Array;
  onChange: (pattern: Int8Array) => void;
  scale?: number;
  readOnly?: boolean;
}

const DEFAULT_SCALE = 12;
const ON_COLOR = "#131827";
const OFF_COLOR = "#eff3fb";
const UNKNOWN_COLOR = "#cbd5e7";
const GRID_COLOR = "rgba(44, 62, 104, 0.18)";

function copyPattern(pattern: Int8Array): Int8Array {
  return pattern.slice();
}

export function PatternCanvas({
  pattern,
  onChange,
  scale = DEFAULT_SCALE,
  readOnly = false,
}: PatternCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const drawingRef = useRef(false);
  const draftPatternRef = useRef(pattern);
  const lastCellRef = useRef<{ x: number; y: number } | null>(null);

  const canvasSize = useMemo(() => PATTERN_SIDE * scale, [scale]);

  function drawCell(context: CanvasRenderingContext2D, x: number, y: number, value: number): void {
    context.fillStyle = value === 1 ? ON_COLOR : value === 0 ? UNKNOWN_COLOR : OFF_COLOR;
    context.fillRect(x * scale, y * scale, scale, scale);

    context.strokeStyle = GRID_COLOR;
    context.lineWidth = 1;
    context.strokeRect(x * scale, y * scale, scale, scale);
  }

  function renderPattern(source: Int8Array): void {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    context.clearRect(0, 0, canvas.width, canvas.height);

    for (let row = 0; row < PATTERN_SIDE; row += 1) {
      for (let column = 0; column < PATTERN_SIDE; column += 1) {
        const index = row * PATTERN_SIDE + column;
        drawCell(context, column, row, source[index]);
      }
    }
  }

  useEffect(() => {
    if (drawingRef.current) {
      return;
    }

    draftPatternRef.current = pattern;
    renderPattern(pattern);
  }, [pattern, scale]);

  function getCell(event: React.PointerEvent<HTMLCanvasElement>): { x: number; y: number } | null {
    const canvas = canvasRef.current;
    if (!canvas) {
      return null;
    }

    const bounds = canvas.getBoundingClientRect();
    const x = Math.floor(((event.clientX - bounds.left) / bounds.width) * PATTERN_SIDE);
    const y = Math.floor(((event.clientY - bounds.top) / bounds.height) * PATTERN_SIDE);

    if (x < 0 || x >= PATTERN_SIDE || y < 0 || y >= PATTERN_SIDE) {
      return null;
    }

    return { x, y };
  }

  function paintCell(next: Int8Array, x: number, y: number): void {
    if (x < 0 || x >= PATTERN_SIDE || y < 0 || y >= PATTERN_SIDE) {
      return;
    }

    const index = y * PATTERN_SIDE + x;
    if (next[index] === 1) {
      return;
    }

    next[index] = 1;

    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    drawCell(context, x, y, 1);
  }

  function paintStroke(from: { x: number; y: number } | null, to: { x: number; y: number } | null): void {
    if (!to || readOnly) {
      return;
    }

    const next = draftPatternRef.current;
    const start = from ?? to;
    const steps = Math.max(Math.abs(to.x - start.x), Math.abs(to.y - start.y), 1);

    for (let step = 0; step <= steps; step += 1) {
      const t = step / steps;
      const x = Math.round(start.x + (to.x - start.x) * t);
      const y = Math.round(start.y + (to.y - start.y) * t);
      paintCell(next, x, y);
    }
  }

  function handlePointerDown(event: React.PointerEvent<HTMLCanvasElement>): void {
    if (readOnly) {
      return;
    }

    drawingRef.current = true;
    draftPatternRef.current = copyPattern(pattern);
    lastCellRef.current = getCell(event);
    event.currentTarget.setPointerCapture(event.pointerId);
    paintStroke(null, lastCellRef.current);
  }

  function handlePointerMove(event: React.PointerEvent<HTMLCanvasElement>): void {
    if (!drawingRef.current || readOnly) {
      return;
    }

    const nextCell = getCell(event);
    paintStroke(lastCellRef.current, nextCell);
    lastCellRef.current = nextCell;
  }

  function finishStroke(event: React.PointerEvent<HTMLCanvasElement>): void {
    if (readOnly || !drawingRef.current) {
      return;
    }

    drawingRef.current = false;
    lastCellRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }

    onChange(copyPattern(draftPatternRef.current));
  }

  return (
    <canvas
      ref={canvasRef}
      width={canvasSize}
      height={canvasSize}
      className={`pattern-canvas${readOnly ? " is-readonly" : ""}`}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={finishStroke}
      onPointerLeave={finishStroke}
    />
  );
}
