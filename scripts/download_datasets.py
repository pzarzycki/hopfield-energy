"""
Download MNIST and Fashion-MNIST datasets, extract one sample per class,
output as base64-encoded grayscale bytes.
"""
import urllib.request, gzip, struct, os, base64

CACHE = '.dataset_cache'
os.makedirs(CACHE, exist_ok=True)

MNIST_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
FASHION_URL = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'

def download(url, filename):
    path = os.path.join(CACHE, filename)
    if not os.path.exists(path):
        print(f'  Downloading {filename}...')
        urllib.request.urlretrieve(url + filename, path)
    return path

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
mnist_imgs = read_idx_images(download(MNIST_URL, 'train-images-idx3-ubyte.gz'))
mnist_lbls = read_idx_labels(download(MNIST_URL, 'train-labels-idx1-ubyte.gz'))

print('Downloading Fashion-MNIST...')
fash_imgs = read_idx_images(download(FASHION_URL, 'train-images-idx3-ubyte.gz'))
fash_lbls = read_idx_labels(download(FASHION_URL, 'train-labels-idx1-ubyte.gz'))

mnist_samples = first_per_class(mnist_imgs, mnist_lbls)
fash_samples = first_per_class(fash_imgs, fash_lbls)

# Write output as TypeScript with base64 strings
with open('scripts/dataset_output.ts', 'w', newline='\n') as f:
    f.write('// Auto-generated. Base64-encoded grayscale 28x28 images (784 bytes each).\n\n')

    f.write('const B64_DIGITS: string[] = [\n')
    for i, pixels in enumerate(mnist_samples):
        f.write(f'  "{to_base64(pixels)}",\n')
    f.write('];\n\n')

    f.write('const B64_FASHION: string[] = [\n')
    for i, pixels in enumerate(fash_samples):
        f.write(f'  "{to_base64(pixels)}",\n')
    f.write('];\n')

print('Done! Output: scripts/dataset_output.ts')
