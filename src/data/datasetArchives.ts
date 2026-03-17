export type DatasetName = "mnist" | "fashion-mnist";

export interface DatasetArchiveSample {
  label: number;
  labelName: string;
  pattern: Uint8Array;
}

export interface DatasetArchive {
  id: DatasetName;
  name: string;
  rows: number;
  cols: number;
  memorySamplesPerClass: number;
  samples: DatasetArchiveSample[];
}

export interface DatasetArchiveClassStat {
  label: number;
  labelName: string;
  count: number;
}

export interface DatasetArchiveStats {
  sampleCount: number;
  classCount: number;
  classStats: DatasetArchiveClassStat[];
  minSamplesPerClass: number;
  maxSamplesPerClass: number;
}

export interface DatasetOption {
  id: DatasetName;
  assetPath: string;
}

const MAGIC = "HEDS";
const VERSION = 1;
const textDecoder = new TextDecoder();
const archiveCache = new Map<DatasetName, Promise<DatasetArchive>>();

export const DATASET_OPTIONS: DatasetOption[] = [
  { id: "mnist", assetPath: "datasets/mnist-small.bin" },
  { id: "fashion-mnist", assetPath: "datasets/fashion-mnist-small.bin" },
];

function getDatasetOption(name: DatasetName): DatasetOption {
  const option = DATASET_OPTIONS.find((entry) => entry.id === name);
  if (!option) {
    throw new Error(`Unknown dataset archive: ${name}`);
  }
  return option;
}

export function getDatasetAssetPath(name: DatasetName): string {
  return getDatasetOption(name).assetPath;
}

function resolveAssetUrl(path: string): string {
  const baseUrl = import.meta.env.BASE_URL ?? "/";
  return `${baseUrl}${path}`;
}

function readString(bytes: Uint8Array, offset: number, length: number): string {
  return textDecoder.decode(bytes.subarray(offset, offset + length));
}

export function parseDatasetArchiveBuffer(buffer: ArrayBuffer): DatasetArchive {
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);

  const magic = readString(bytes, 0, 4);
  if (magic !== MAGIC) {
    throw new Error(`Invalid dataset archive magic: expected ${MAGIC}, received ${magic}.`);
  }

  const version = view.getUint16(4, true);
  if (version !== VERSION) {
    throw new Error(`Unsupported dataset archive version ${version}.`);
  }

  const rows = view.getUint16(6, true);
  const cols = view.getUint16(8, true);
  const sampleCount = view.getUint16(10, true);
  const labelNameCount = view.getUint16(12, true);
  const idLength = view.getUint16(14, true);
  const nameLength = view.getUint16(16, true);
  const memorySamplesPerClass = view.getUint16(18, true);

  let offset = 20;
  const id = readString(bytes, offset, idLength) as DatasetName;
  offset += idLength;

  const name = readString(bytes, offset, nameLength);
  offset += nameLength;

  const labelNames: string[] = [];
  for (let index = 0; index < labelNameCount; index += 1) {
    const length = view.getUint16(offset, true);
    offset += 2;
    labelNames.push(readString(bytes, offset, length));
    offset += length;
  }

  const labels = bytes.slice(offset, offset + sampleCount);
  offset += sampleCount;

  const pixelsPerSample = rows * cols;
  const expectedPixels = sampleCount * pixelsPerSample;
  const pixelBytes = bytes.slice(offset, offset + expectedPixels);

  if (pixelBytes.length !== expectedPixels) {
    throw new Error(`Dataset archive ${id} is truncated.`);
  }

  const samples: DatasetArchiveSample[] = [];
  for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex += 1) {
    const label = labels[sampleIndex];
    const start = sampleIndex * pixelsPerSample;
    const pattern = pixelBytes.slice(start, start + pixelsPerSample);
    samples.push({
      label,
      labelName: labelNames[label] ?? String(label),
      pattern,
    });
  }

  return { id, name, rows, cols, memorySamplesPerClass, samples };
}

export async function loadDatasetArchive(name: DatasetName): Promise<DatasetArchive> {
  const cached = archiveCache.get(name);
  if (cached) {
    return cached;
  }

  const option = getDatasetOption(name);
  const next = fetch(resolveAssetUrl(option.assetPath)).then(async (response) => {
    if (!response.ok) {
      throw new Error(`Failed to load dataset archive ${name}.`);
    }
    return parseDatasetArchiveBuffer(await response.arrayBuffer());
  });

  archiveCache.set(name, next);
  return next;
}

export function grayscaleToFloat32(pattern: ArrayLike<number>): Float32Array {
  return Float32Array.from(pattern, (value) => value / 255);
}

export function grayscaleToBipolar(pattern: ArrayLike<number>, threshold = 128): Int8Array {
  return Int8Array.from(pattern, (value) => (value >= threshold ? 1 : -1));
}

export function selectMemorySamples(archive: DatasetArchive, samplesPerClass = archive.memorySamplesPerClass): DatasetArchiveSample[] {
  const selected: DatasetArchiveSample[] = [];
  const seenPerClass = new Map<number, number>();

  for (const sample of archive.samples) {
    const current = seenPerClass.get(sample.label) ?? 0;
    if (current >= samplesPerClass) {
      continue;
    }
    selected.push(sample);
    seenPerClass.set(sample.label, current + 1);
  }

  return selected;
}

export function summarizeDatasetArchive(archive: DatasetArchive): DatasetArchiveStats {
  const counts = new Map<number, DatasetArchiveClassStat>();

  for (const sample of archive.samples) {
    const existing = counts.get(sample.label);
    if (existing) {
      existing.count += 1;
    } else {
      counts.set(sample.label, {
        label: sample.label,
        labelName: sample.labelName,
        count: 1,
      });
    }
  }

  const classStats = [...counts.values()].sort((left, right) => left.label - right.label);
  const sampleCounts = classStats.map((entry) => entry.count);

  return {
    sampleCount: archive.samples.length,
    classCount: classStats.length,
    classStats,
    minSamplesPerClass: sampleCounts.length > 0 ? Math.min(...sampleCounts) : 0,
    maxSamplesPerClass: sampleCounts.length > 0 ? Math.max(...sampleCounts) : 0,
  };
}
