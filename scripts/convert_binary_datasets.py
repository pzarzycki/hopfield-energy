"""
Convert cached MNIST and Fashion-MNIST training sets into packed binary archives
for frontend loading.

Output format:
- public/datasets/mnist-train-binary.json
- public/datasets/fashion-mnist-train-binary.json

Each image is thresholded to binary with threshold 128 and then packed to 98 bytes
(784 bits) per sample. Labels are stored as raw bytes and base64 encoded.
"""

from __future__ import annotations

import base64
import gzip
import json
import os
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PUBLIC_DATASETS = ROOT / "public" / "datasets"
PUBLIC_DATASETS.mkdir(parents=True, exist_ok=True)


DATASET_SPECS = [
    {
        "name": "mnist",
        "image_files": ["mnist-train-images.gz", "train-images-idx3-ubyte.gz"],
        "label_files": ["mnist-train-labels.gz", "train-labels-idx1-ubyte.gz"],
        "label_names": [str(index) for index in range(10)],
    },
    {
        "name": "fashion-mnist",
        "image_files": ["fashion-train-images.gz", "train-images-idx3-ubyte.gz"],
        "label_files": ["fashion-train-labels.gz", "train-labels-idx1-ubyte.gz"],
        "label_names": ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"],
    },
]


def find_cache_file(*candidate_names: str) -> Path:
    search_roots = [
        ROOT / "scripts" / ".dataset_cache",
        ROOT / ".dataset_cache",
    ]
    for search_root in search_roots:
        for candidate_name in candidate_names:
            candidate = search_root / candidate_name
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Could not find cached dataset file among: {', '.join(candidate_names)}")


def read_idx_images(path: Path) -> tuple[int, int, list[bytes]]:
    with gzip.open(path, "rb") as handle:
        _magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        images = [handle.read(rows * cols) for _ in range(count)]
    return rows, cols, images


def read_idx_labels(path: Path) -> list[int]:
    with gzip.open(path, "rb") as handle:
        _magic, count = struct.unpack(">II", handle.read(8))
        labels = list(handle.read(count))
    return labels


def pack_binary_image(image: bytes, threshold: int) -> bytes:
    packed = bytearray((len(image) + 7) // 8)
    for index, value in enumerate(image):
        if value >= threshold:
            packed[index // 8] |= 1 << (7 - (index % 8))
    return bytes(packed)


def convert_dataset(spec: dict[str, object]) -> None:
    image_path = find_cache_file(*spec["image_files"])  # type: ignore[arg-type]
    label_path = find_cache_file(*spec["label_files"])  # type: ignore[arg-type]

    rows, cols, images = read_idx_images(image_path)
    labels = read_idx_labels(label_path)
    threshold = 128

    packed_images = bytearray()
    for image in images:
      packed_images.extend(pack_binary_image(image, threshold))

    archive = {
        "name": spec["name"],
        "split": "train",
        "rows": rows,
        "cols": cols,
        "count": len(images),
        "threshold": threshold,
        "bits_per_sample": rows * cols,
        "bytes_per_sample": (rows * cols + 7) // 8,
        "label_names": spec["label_names"],
        "packed_images_b64": base64.b64encode(bytes(packed_images)).decode("ascii"),
        "labels_b64": base64.b64encode(bytes(labels)).decode("ascii"),
    }

    output_path = PUBLIC_DATASETS / f"{spec['name']}-train-binary.json"
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(archive, handle, separators=(",", ":"))

    print(f"Wrote {output_path.relative_to(ROOT)}")


def main() -> None:
    for spec in DATASET_SPECS:
        convert_dataset(spec)


if __name__ == "__main__":
    main()
