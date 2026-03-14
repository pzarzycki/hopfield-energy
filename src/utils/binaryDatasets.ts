export type BinaryDatasetName = "mnist" | "fashion-mnist";

export interface BinaryDatasetArchive {
  name: BinaryDatasetName;
  split: "train";
  rows: number;
  cols: number;
  count: number;
  threshold: number;
  bits_per_sample: number;
  bytes_per_sample: number;
  label_names: string[];
  packed_images_b64: string;
  labels_b64: string;
}

const archiveCache = new Map<BinaryDatasetName, Promise<BinaryDatasetArchive>>();
const decodedBytesCache = new WeakMap<BinaryDatasetArchive, Uint8Array>();
const decodedLabelsCache = new WeakMap<BinaryDatasetArchive, Uint8Array>();

function decodeBase64(source: string): Uint8Array {
  const decoded = atob(source);
  const bytes = new Uint8Array(decoded.length);
  for (let index = 0; index < decoded.length; index += 1) {
    bytes[index] = decoded.charCodeAt(index);
  }
  return bytes;
}

function getPackedBytes(archive: BinaryDatasetArchive): Uint8Array {
  let cached = decodedBytesCache.get(archive);
  if (!cached) {
    cached = decodeBase64(archive.packed_images_b64);
    decodedBytesCache.set(archive, cached);
  }
  return cached;
}

function getLabels(archive: BinaryDatasetArchive): Uint8Array {
  let cached = decodedLabelsCache.get(archive);
  if (!cached) {
    cached = decodeBase64(archive.labels_b64);
    decodedLabelsCache.set(archive, cached);
  }
  return cached;
}

export async function loadBinaryDataset(name: BinaryDatasetName): Promise<BinaryDatasetArchive> {
  const existing = archiveCache.get(name);
  if (existing) {
    return existing;
  }

  const next = fetch(`/datasets/${name}-train-binary.json`).then(async (response) => {
    if (!response.ok) {
      throw new Error(`Unable to load dataset archive for ${name}.`);
    }
    return (await response.json()) as BinaryDatasetArchive;
  });

  archiveCache.set(name, next);
  return next;
}

export function unpackBinarySample(archive: BinaryDatasetArchive, sampleIndex: number): Uint8Array {
  if (sampleIndex < 0 || sampleIndex >= archive.count) {
    throw new RangeError(`Sample index ${sampleIndex} is out of range for ${archive.name}.`);
  }

  const packed = getPackedBytes(archive);
  const offset = sampleIndex * archive.bytes_per_sample;
  const binary = new Uint8Array(archive.bits_per_sample);

  for (let bitIndex = 0; bitIndex < archive.bits_per_sample; bitIndex += 1) {
    const sourceByte = packed[offset + (bitIndex >> 3)];
    binary[bitIndex] = (sourceByte >> (7 - (bitIndex & 7))) & 1;
  }

  return binary;
}

export function unpackBipolarSample(archive: BinaryDatasetArchive, sampleIndex: number): Int8Array {
  const binary = unpackBinarySample(archive, sampleIndex);
  const bipolar = new Int8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bipolar[index] = binary[index] === 1 ? 1 : -1;
  }
  return bipolar;
}

export function getDatasetLabel(archive: BinaryDatasetArchive, sampleIndex: number): number {
  return getLabels(archive)[sampleIndex];
}

export function getDatasetLabelName(archive: BinaryDatasetArchive, sampleIndex: number): string {
  const label = getDatasetLabel(archive, sampleIndex);
  return archive.label_names[label] ?? String(label);
}
