"""
Download MNIST and Fashion-MNIST datasets, extract one sample per class,
output as base64-encoded grayscale bytes.
"""
import base64
import gzip
import os
import struct
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE = SCRIPT_DIR / '.dataset_cache'
CACHE.mkdir(exist_ok=True)
OUTPUT = SCRIPT_DIR / 'dataset_output.ts'

MNIST_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
FASHION_URL = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'

DATASET_FILES = {
    'mnist-images': ['mnist-train-images.gz', 'mnist-images.gz', 'train-images-idx3-ubyte.gz'],
    'mnist-labels': ['mnist-train-labels.gz', 'mnist-labels.gz', 'train-labels-idx1-ubyte.gz'],
    'fashion-images': ['fashion-train-images.gz', 'fashion-images.gz', 'train-images-idx3-ubyte.gz'],
    'fashion-labels': ['fashion-train-labels.gz', 'fashion-labels.gz', 'train-labels-idx1-ubyte.gz'],
}

def resolve_or_download(url, candidate_names):
    for filename in candidate_names:
        path = CACHE / filename
        if path.exists():
            return str(path)

    filename = candidate_names[-1]
    path = CACHE / filename
    print(f'  Downloading {filename}...')
    urllib.request.urlretrieve(url + filename, path)
    return str(path)

def read_idx_images(path):
    with gzip.open(path, 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        return [list(f.read(rows * cols)) for _ in range(n)]

def read_idx_labels(path):
    with gzip.open(path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        return list(f.read(n))

def first_per_class(images, labels):
    """Pick the first occurrence of each class 0-9."""
    result = {}
    for img, lbl in zip(images, labels):
        if lbl not in result:
            result[lbl] = img
        if len(result) == 10:
            break
    return [result[i] for i in range(10)]

def to_base64(pixels):
    """Encode a list of 0-255 values as base64 string."""
    return base64.b64encode(bytes(pixels)).decode('ascii')

# Download
print('Downloading MNIST...')
mnist_imgs = read_idx_images(resolve_or_download(MNIST_URL, DATASET_FILES['mnist-images']))
mnist_lbls = read_idx_labels(resolve_or_download(MNIST_URL, DATASET_FILES['mnist-labels']))

print('Downloading Fashion-MNIST...')
fash_imgs = read_idx_images(resolve_or_download(FASHION_URL, DATASET_FILES['fashion-images']))
fash_lbls = read_idx_labels(resolve_or_download(FASHION_URL, DATASET_FILES['fashion-labels']))

mnist_samples = first_per_class(mnist_imgs, mnist_lbls)
fash_samples = first_per_class(fash_imgs, fash_lbls)

# Write output as TypeScript with base64 strings
with open(OUTPUT, 'w', newline='\n') as f:
    f.write('// Auto-generated. Base64-encoded grayscale 28x28 images (784 bytes each).\n\n')

    f.write('const B64_DIGITS: string[] = [\n')
    for i, pixels in enumerate(mnist_samples):
        f.write(f'  "{to_base64(pixels)}",\n')
    f.write('];\n\n')

    f.write('const B64_FASHION: string[] = [\n')
    for i, pixels in enumerate(fash_samples):
        f.write(f'  "{to_base64(pixels)}",\n')
    f.write('];\n')

print(f'Done! Output: {OUTPUT}')
