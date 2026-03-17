"""
Build compact binary dataset archives for MNIST and Fashion-MNIST subsets.

The archive format is:
  magic[4]            = b"HEDS"
  version[u16]        = 1
  rows[u16]
  cols[u16]
  sample_count[u16]
  label_name_count[u16]
  dataset_id_len[u16]
  dataset_name_len[u16]
  dataset_id[bytes]
  dataset_name[bytes]
  repeated label names:
    label_len[u16]
    label_bytes
  labels[sample_count]                  one byte per sample
  memory_sample_count[u16]              recommended per-class memory slice
  grayscale_pixels[rows*cols*N]         one byte per pixel
"""
import gzip
import struct
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
CACHE = SCRIPT_DIR / ".dataset_cache"
OUTPUT_DIR = ROOT / "public" / "datasets"

MAGIC = b"HEDS"
VERSION = 1

MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FASHION_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

DATASETS = {
    "mnist": {
        "display_name": "MNIST",
        "base_url": MNIST_URL,
        "images": ["mnist-train-images.gz", "mnist-images.gz", "train-images-idx3-ubyte.gz"],
        "labels": ["mnist-train-labels.gz", "mnist-labels.gz", "train-labels-idx1-ubyte.gz"],
        "label_names": [str(index) for index in range(10)],
        "output": OUTPUT_DIR / "mnist-small.bin",
        "samples_per_class": 100,
        "memory_samples_per_class": 1,
    },
    "fashion-mnist": {
        "display_name": "Fashion-MNIST",
        "base_url": FASHION_URL,
        "images": ["fashion-train-images.gz", "fashion-images.gz", "train-images-idx3-ubyte.gz"],
        "labels": ["fashion-train-labels.gz", "fashion-labels.gz", "train-labels-idx1-ubyte.gz"],
        "label_names": ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"],
        "output": OUTPUT_DIR / "fashion-mnist-small.bin",
        "samples_per_class": 100,
        "memory_samples_per_class": 1,
    },
}


def resolve_or_download(url: str, candidates: list[str]) -> Path:
    CACHE.mkdir(exist_ok=True)
    for filename in candidates:
        path = CACHE / filename
        if path.exists():
            return path

    path = CACHE / candidates[-1]
    print(f"Downloading {path.name}...")
    urllib.request.urlretrieve(url + path.name, path)
    return path


def read_idx_images(path: Path) -> tuple[int, int, list[bytes]]:
    with gzip.open(path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image magic number {magic} in {path}")
        images = [handle.read(rows * cols) for _ in range(count)]
    return rows, cols, images


def read_idx_labels(path: Path) -> list[int]:
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label magic number {magic} in {path}")
        return list(handle.read(count))


def balanced_samples_per_class(
    images: list[bytes],
    labels: list[int],
    class_count: int,
    samples_per_class: int,
) -> tuple[list[int], list[bytes]]:
    grouped: dict[int, list[bytes]] = {label: [] for label in range(class_count)}

    for image, label in zip(images, labels):
        bucket = grouped.get(label)
        if bucket is None or len(bucket) >= samples_per_class:
            continue
        bucket.append(image)
        if all(len(grouped[class_label]) >= samples_per_class for class_label in range(class_count)):
            break

    if not all(len(grouped[class_label]) >= samples_per_class for class_label in range(class_count)):
        raise ValueError("Did not find enough samples for every class.")

    ordered_labels: list[int] = []
    ordered_images: list[bytes] = []
    for class_label in range(class_count):
        for image in grouped[class_label]:
            ordered_labels.append(class_label)
            ordered_images.append(image)
    return ordered_labels, ordered_images


def write_archive(dataset_id: str, config: dict) -> None:
    rows, cols, images = read_idx_images(resolve_or_download(config["base_url"], config["images"]))
    labels = read_idx_labels(resolve_or_download(config["base_url"], config["labels"]))
    sample_labels, sample_images = balanced_samples_per_class(
        images,
        labels,
        len(config["label_names"]),
        config["samples_per_class"],
    )

    payload = bytearray()
    payload.extend(MAGIC)
    payload.extend(
        struct.pack(
            "<HHHHHHHH",
            VERSION,
            rows,
            cols,
            len(sample_images),
            len(config["label_names"]),
            len(dataset_id.encode("utf-8")),
            len(config["display_name"].encode("utf-8")),
            config["memory_samples_per_class"],
        )
    )
    payload.extend(dataset_id.encode("utf-8"))
    payload.extend(config["display_name"].encode("utf-8"))

    for label_name in config["label_names"]:
        encoded = label_name.encode("utf-8")
        payload.extend(struct.pack("<H", len(encoded)))
        payload.extend(encoded)

    payload.extend(bytes(sample_labels))
    for image in sample_images:
        payload.extend(image)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config["output"].write_bytes(payload)
    print(
        f"Wrote {config['output']} ({len(sample_images)} samples, "
        f"{config['samples_per_class']} per class, {rows}x{cols})"
    )


def main() -> None:
    for dataset_id, config in DATASETS.items():
        write_archive(dataset_id, config)


if __name__ == "__main__":
    main()
