export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function lerp(start: number, end: number, amount: number): number {
  return start + (end - start) * amount;
}

function mixColor(a: [number, number, number], b: [number, number, number], t: number): [number, number, number] {
  return [
    Math.round(lerp(a[0], b[0], t)),
    Math.round(lerp(a[1], b[1], t)),
    Math.round(lerp(a[2], b[2], t)),
  ];
}

export function divergingColor(value: number, maxAbs: number): [number, number, number] {
  const safeMax = maxAbs || 1;
  const normalized = clamp(value / safeMax, -1, 1);
  const low: [number, number, number] = [28, 78, 216];
  const mid: [number, number, number] = [244, 246, 250];
  const high: [number, number, number] = [216, 50, 66];

  if (normalized < 0) {
    return mixColor(mid, low, Math.abs(normalized));
  }

  return mixColor(mid, high, normalized);
}

export function writePatternImage(target: Uint8ClampedArray, pattern: Int8Array | Uint8Array): void {
  for (let index = 0; index < pattern.length; index += 1) {
    const offset = index * 4;
    const [red, green, blue] = pattern[index] > 0 ? [22, 27, 45] : [245, 247, 252];
    target[offset] = red;
    target[offset + 1] = green;
    target[offset + 2] = blue;
    target[offset + 3] = 255;
  }
}

export function writeGrayscaleImage(target: Uint8ClampedArray, pattern: Float32Array | Uint8Array): void {
  for (let index = 0; index < pattern.length; index += 1) {
    const offset = index * 4;
    const value = clamp(Number(pattern[index]), 0, 1);
    const shade = Math.round(247 - value * 220);
    target[offset] = shade;
    target[offset + 1] = shade;
    target[offset + 2] = shade;
    target[offset + 3] = 255;
  }
}

export function writeWeightHeatmap(target: Uint8ClampedArray, matrix: Float32Array, maxAbs: number): void {
  for (let index = 0; index < matrix.length; index += 1) {
    const [red, green, blue] = divergingColor(matrix[index], maxAbs);
    const offset = index * 4;
    target[offset] = red;
    target[offset + 1] = green;
    target[offset + 2] = blue;
    target[offset + 3] = 255;
  }
}
