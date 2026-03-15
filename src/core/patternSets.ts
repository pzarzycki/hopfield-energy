import { getDatasetSamples, getFashionLabel, grayscaleToBipolar } from "../utils/mnist-data";

export const PATTERN_SIDE = 28;
export const PATTERN_SIZE = PATTERN_SIDE * PATTERN_SIDE;

export interface PatternSetDefinition {
  id: string;
  name: string;
  description: string;
  labels: string[];
  patterns: Int8Array[];
}

function createPattern(): Int8Array {
  return new Int8Array(PATTERN_SIZE).fill(-1);
}

function setPixel(pattern: Int8Array, x: number, y: number, value = 1): void {
  if (x < 0 || x >= PATTERN_SIDE || y < 0 || y >= PATTERN_SIDE) {
    return;
  }

  pattern[y * PATTERN_SIDE + x] = value;
}

function fillRect(pattern: Int8Array, x: number, y: number, width: number, height: number, value = 1): void {
  for (let row = y; row < y + height; row += 1) {
    for (let column = x; column < x + width; column += 1) {
      setPixel(pattern, column, row, value);
    }
  }
}

function strokeRect(pattern: Int8Array, x: number, y: number, width: number, height: number, thickness = 3): void {
  fillRect(pattern, x, y, width, thickness);
  fillRect(pattern, x, y + height - thickness, width, thickness);
  fillRect(pattern, x, y, thickness, height);
  fillRect(pattern, x + width - thickness, y, thickness, height);
}

function strokeLine(pattern: Int8Array, x0: number, y0: number, x1: number, y1: number, thickness = 3): void {
  const dx = Math.abs(x1 - x0);
  const dy = Math.abs(y1 - y0);
  const sx = x0 < x1 ? 1 : -1;
  const sy = y0 < y1 ? 1 : -1;
  let error = dx - dy;

  while (true) {
    fillRect(pattern, x0 - Math.floor(thickness / 2), y0 - Math.floor(thickness / 2), thickness, thickness);
    if (x0 === x1 && y0 === y1) {
      break;
    }

    const doubledError = error * 2;
    if (doubledError > -dy) {
      error -= dy;
      x0 += sx;
    }
    if (doubledError < dx) {
      error += dx;
      y0 += sy;
    }
  }
}

function fillDiamond(pattern: Int8Array, centerX: number, centerY: number, radius: number): void {
  for (let y = -radius; y <= radius; y += 1) {
    const span = radius - Math.abs(y);
    for (let x = -span; x <= span; x += 1) {
      setPixel(pattern, centerX + x, centerY + y);
    }
  }
}

function fillTriangle(pattern: Int8Array, apexX: number, apexY: number, halfBase: number, height: number): void {
  for (let row = 0; row < height; row += 1) {
    const span = Math.floor((row / Math.max(height - 1, 1)) * halfBase);
    const y = apexY + row;
    for (let x = apexX - span; x <= apexX + span; x += 1) {
      setPixel(pattern, x, y);
    }
  }
}

function renderBitmap(bitmap: string[]): Int8Array {
  const pattern = createPattern();
  const cellWidth = 4;
  const cellHeight = 4;
  const glyphWidth = bitmap[0].length * cellWidth;
  const glyphHeight = bitmap.length * cellHeight;
  const offsetX = Math.floor((PATTERN_SIDE - glyphWidth) / 2);
  const offsetY = Math.floor((PATTERN_SIDE - glyphHeight) / 2);

  for (let row = 0; row < bitmap.length; row += 1) {
    for (let column = 0; column < bitmap[row].length; column += 1) {
      if (bitmap[row][column] === "1") {
        fillRect(pattern, offsetX + column * cellWidth, offsetY + row * cellHeight, cellWidth, cellHeight);
      }
    }
  }

  return pattern;
}

function fillByPredicate(pattern: Int8Array, predicate: (x: number, y: number) => boolean): void {
  for (let y = 0; y < PATTERN_SIDE; y += 1) {
    for (let x = 0; x < PATTERN_SIDE; x += 1) {
      if (predicate(x, y)) {
        setPixel(pattern, x, y);
      }
    }
  }
}


const DIGIT_BITMAPS: Array<{ label: string; bitmap: string[] }> = [
  {
    label: "0",
    bitmap: ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
  },
  {
    label: "1",
    bitmap: ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
  },
  {
    label: "2",
    bitmap: ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
  },
  {
    label: "3",
    bitmap: ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
  },
  {
    label: "4",
    bitmap: ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
  },
  {
    label: "5",
    bitmap: ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
  },
  {
    label: "6",
    bitmap: ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
  },
  {
    label: "7",
    bitmap: ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
  },
  {
    label: "8",
    bitmap: ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
  },
  {
    label: "9",
    bitmap: ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
  },
];

function normalizedCorrelation(a: Int8Array, b: Int8Array): number {
  let dot = 0;
  for (let index = 0; index < a.length; index += 1) {
    dot += a[index] * b[index];
  }
  return dot / a.length;
}

function buildDigitPatternSet(): PatternSetDefinition {
  const labels = DIGIT_BITMAPS.map((item) => item.label);
  const patterns = DIGIT_BITMAPS.map((item) => renderBitmap(item.bitmap));
  const maxCorrelation = Math.max(
    ...patterns.flatMap((pattern, patternIndex) =>
      patterns.slice(patternIndex + 1).map((otherPattern) => Math.abs(normalizedCorrelation(pattern, otherPattern))),
    ),
  );

  return {
    id: "digits-10",
    name: "Digits 0-9",
    description: `Ten binary digit glyphs on a 28x28 grid. Max pairwise correlation ${maxCorrelation.toFixed(2)}.`,
    labels,
    patterns,
  };
}

function buildShapePatternSet(): PatternSetDefinition {
  const shapes: Array<{ label: string; pattern: Int8Array }> = [];

  const ring = createPattern();
  strokeRect(ring, 6, 6, 16, 16, 3);
  shapes.push({ label: "O", pattern: ring });

  const square = createPattern();
  fillRect(square, 7, 7, 14, 14);
  shapes.push({ label: "S", pattern: square });

  const cross = createPattern();
  strokeLine(cross, 5, 5, 22, 22, 3);
  strokeLine(cross, 22, 5, 5, 22, 3);
  shapes.push({ label: "X", pattern: cross });

  const plus = createPattern();
  fillRect(plus, 11, 5, 6, 18);
  fillRect(plus, 5, 11, 18, 6);
  shapes.push({ label: "+", pattern: plus });

  const triangle = createPattern();
  fillTriangle(triangle, 14, 5, 10, 16);
  fillTriangle(triangle, 14, 10, 5, 8);
  shapes.push({ label: "T", pattern: triangle });

  const diamond = createPattern();
  fillDiamond(diamond, 14, 14, 9);
  fillDiamond(diamond, 14, 14, 5);
  fillRect(diamond, 9, 9, 11, 11, -1);
  shapes.push({ label: "D", pattern: diamond });

  const leftArrow = createPattern();
  fillRect(leftArrow, 9, 11, 12, 6);
  strokeLine(leftArrow, 9, 14, 17, 6, 3);
  strokeLine(leftArrow, 9, 14, 17, 22, 3);
  shapes.push({ label: "<", pattern: leftArrow });

  const rightArrow = createPattern();
  fillRect(rightArrow, 7, 11, 12, 6);
  strokeLine(rightArrow, 19, 14, 11, 6, 3);
  strokeLine(rightArrow, 19, 14, 11, 22, 3);
  shapes.push({ label: ">", pattern: rightArrow });

  const verticalBars = createPattern();
  fillRect(verticalBars, 7, 5, 5, 18);
  fillRect(verticalBars, 16, 5, 5, 18);
  shapes.push({ label: "||", pattern: verticalBars });

  const horizontalBars = createPattern();
  fillRect(horizontalBars, 5, 7, 18, 5);
  fillRect(horizontalBars, 5, 16, 18, 5);
  shapes.push({ label: "=", pattern: horizontalBars });

  const labels = shapes.map((shape) => shape.label);
  const patterns = shapes.map((shape) => shape.pattern);
  const maxCorrelation = Math.max(
    ...patterns.flatMap((pattern, patternIndex) =>
      patterns.slice(patternIndex + 1).map((otherPattern) => Math.abs(normalizedCorrelation(pattern, otherPattern))),
    ),
  );

  return {
    id: "shapes-10",
    name: "Shapes 10",
    description: `Ten geometric symbols on a 28x28 grid. Max pairwise correlation ${maxCorrelation.toFixed(2)}.`,
    labels,
    patterns,
  };
}

function buildOrthogonalHatchPatternSet(): PatternSetDefinition {
  const diagonalLeft = createPattern();
  fillByPredicate(diagonalLeft, (x, y) => (x + y) % 4 < 2);

  const diagonalRight = createPattern();
  fillByPredicate(diagonalRight, (x, y) => ((x - y + PATTERN_SIDE * 4) % 4) < 2);

  const horizontal = createPattern();
  fillByPredicate(horizontal, (_x, y) => y % 4 < 2);

  const vertical = createPattern();
  fillByPredicate(vertical, (x) => x % 4 < 2);

  const labels = ["diag \u2196", "diag \u2197", "horiz", "vert"];
  const patterns = [diagonalLeft, diagonalRight, horizontal, vertical];
  const maxCorrelation = Math.max(
    ...patterns.flatMap((pattern, patternIndex) =>
      patterns.slice(patternIndex + 1).map((otherPattern) => Math.abs(normalizedCorrelation(pattern, otherPattern))),
    ),
  );

  return {
    id: "orthogonal-hatches-4",
    name: "Orthogonal Hatches",
    description: `Four 28x28 stripe fields with near-zero pairwise correlation. Max pairwise correlation ${maxCorrelation.toFixed(2)}.`,
    labels,
    patterns,
  };
}

function buildRealDatasetPatternSet(dataset: "mnist" | "fashion-mnist"): PatternSetDefinition {
  const samples = getDatasetSamples(dataset);
  const labels =
    dataset === "mnist" ? samples.map((sample) => String(sample.label)) : samples.map((sample) => getFashionLabel(sample.label));
  const patterns = samples.map((sample) => Int8Array.from(grayscaleToBipolar(sample.pattern, 128)));
  const maxCorrelation = Math.max(
    ...patterns.flatMap((pattern, patternIndex) =>
      patterns.slice(patternIndex + 1).map((otherPattern) => Math.abs(normalizedCorrelation(pattern, otherPattern))),
    ),
  );

  return {
    id: dataset,
    name: dataset === "mnist" ? "MNIST" : "Fashion-MNIST",
    description:
      dataset === "mnist"
        ? `Real grayscale MNIST exemplars, binarized at threshold 128 for classical Hopfield storage. Max pairwise correlation ${maxCorrelation.toFixed(2)}.`
        : `Real grayscale Fashion-MNIST exemplars, binarized at threshold 128 for classical Hopfield storage. Max pairwise correlation ${maxCorrelation.toFixed(2)}.`,
    labels,
    patterns,
  };
}

export const PATTERN_SETS: PatternSetDefinition[] = [
  buildRealDatasetPatternSet("mnist"),
  buildRealDatasetPatternSet("fashion-mnist"),
  buildOrthogonalHatchPatternSet(),
  buildDigitPatternSet(),
  buildShapePatternSet(),
];

export function getPatternSetById(id: string): PatternSetDefinition {
  return PATTERN_SETS.find((patternSet) => patternSet.id === id) ?? PATTERN_SETS[0];
}

export function clonePattern(pattern: Int8Array): Int8Array {
  return pattern.slice();
}

export function createBlankPattern(): Int8Array {
  return createPattern();
}
