"""
Assemble mnist-data.ts from base64-encoded dataset output.
"""
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DATASET_OUTPUT = SCRIPT_DIR / 'dataset_output.ts'
MNIST_DATA = ROOT / 'src' / 'utils' / 'mnist-data.ts'

with open(DATASET_OUTPUT, 'r') as f:
    data = f.read().strip()

header = """// Real MNIST + Fashion-MNIST samples. Base64-encoded grayscale (0-255).
// MNIST: https://storage.googleapis.com/cvdf-datasets/mnist/
// Fashion-MNIST: http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/

export interface MNISTSample {
  label: number;
  pattern: number[];  // grayscale 0-255
}

"""

footer = """

// Decode base64 string to number array
function decodeB64(b64: string): number[] {
  const bin = atob(b64);
  const arr = new Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return arr;
}

// Decoded caches
let _digits: number[][] | null = null;
let _fashion: number[][] | null = null;

function getDigits(): number[][] {
  if (!_digits) _digits = B64_DIGITS.map(decodeB64);
  return _digits;
}
function getFashion(): number[][] {
  if (!_fashion) _fashion = B64_FASHION.map(decodeB64);
  return _fashion;
}

export const getMNISTDigits = (): MNISTSample[] => {
  const d = getDigits();
  return d.map((pattern, i) => ({ label: i, pattern }));
};

// Render pattern to ImageData — handles both grayscale (0-255) and bipolar (-1/1)
export const patternToImageData = (pattern: number[]): ImageData => {
  const size = Math.sqrt(pattern.length);
  const imageData = new ImageData(size, size);
  const isBipolar = pattern.some(v => v < 0);

  for (let i = 0; i < pattern.length; i++) {
    const px = i * 4;
    let value: number;
    if (isBipolar) {
      value = pattern[i] === 1 ? 0 : 255;
    } else {
      value = 255 - pattern[i];
    }
    imageData.data[px] = value;
    imageData.data[px + 1] = value;
    imageData.data[px + 2] = value;
    imageData.data[px + 3] = 255;
  }
  return imageData;
};

export const imageDataToPattern = (imageData: ImageData): number[] => {
  const pattern: number[] = [];
  for (let i = 0; i < imageData.data.length; i += 4) {
    pattern.push(255 - imageData.data[i]);
  }
  return pattern;
};

// Convert grayscale [0-255] to bipolar [-1, 1]
export const grayscaleToBipolar = (pattern: number[], threshold = 128): number[] => {
  return pattern.map(v => v >= threshold ? 1 : -1);
};

// Convert bipolar [-1, 1] to grayscale [0, 255]
export const bipolarToGrayscale = (pattern: number[]): number[] => {
  return pattern.map(v => v === 1 ? 255 : 0);
};

export const addNoise = (pattern: number[], noiseLevel: number): number[] => {
  return pattern.map(value => {
    if (Math.random() < noiseLevel) return 255 - value;
    return value;
  });
};

const fashionLabels: Record<number, string> = {
  0: 'T-shirt', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
  5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Boot',
};

export const getFashionLabel = (index: number): string => {
  return fashionLabels[index] || `Item ${index}`;
};

export const getDatasetSamples = (dataset: 'mnist' | 'fashion-mnist'): MNISTSample[] => {
  const source = dataset === 'fashion-mnist' ? getFashion() : getDigits();
  return source.map((pattern, i) => ({ label: i, pattern }));
};

export type MNISTDigit = MNISTSample;

export const bipolarToBinary = (pattern: number[]): number[] => {
  return pattern.map(v => v === 1 ? 1 : 0);
};

export const binaryToBipolar = (pattern: number[]): number[] => {
  return pattern.map(v => v === 1 ? 1 : -1);
};
"""

with open(MNIST_DATA, 'w', newline='\n') as f:
    f.write(header)
    f.write(data)
    f.write(footer)

print("Done!")
