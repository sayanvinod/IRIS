"""
Step 1b — Deduplicated Cluster-Aware Sampling + Train/Val/Test Split

Builds a clean dataset from the full Kaggle IMAGES/ folder by:
  1. Grouping exact byte duplicates (MD5) and perceptually identical images
     (dHash distance == 0) into duplicate clusters, per class.
  2. Keeping one representative image per cluster.
  3. Randomly sampling balanced classes from the deduplicated pool.
  4. Splitting 70/15/15 (seed=42) into YOLO-compatible folders.

This script is read-only on the source IMAGES/ folder.

Usage:
    python 1b_sample_data_clean.py --images-dir ../IMAGES --output-dir data_10k_clean --n-per-class 5000

Outputs:
    data_10k_clean/train|validation|test/{Myopia,Normal}/
    results/dedup_manifest.json   — cluster stats and sampling metadata
"""

import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

SOURCE_CLASS_MAP = {
    "Myopia_images": "Myopia",
    "Normal_images": "Normal",
}


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash_uint64(path: Path, hash_size: int = 8) -> int:
    img = Image.open(path).convert("L").resize(
        (hash_size + 1, hash_size), Image.LANCZOS
    )
    arr = np.asarray(img, dtype=np.int16)
    diff = (arr[:, 1:] > arr[:, :-1]).flatten()
    val = 0
    for b in diff:
        val = (val << 1) | int(b)
    return val


def build_duplicate_clusters(paths: list[Path]) -> tuple[list[list[Path]], dict]:
    """Return cluster lists and stats for one class."""
    n = len(paths)
    uf = UnionFind(n)

    md5_groups = defaultdict(list)
    dhash_groups = defaultdict(list)
    bad = []

    for i, p in enumerate(paths):
        try:
            md5 = md5_of(p)
            dh = dhash_uint64(p)
        except Exception as e:
            bad.append({"path": str(p), "error": str(e)})
            continue
        md5_groups[md5].append(i)
        dhash_groups[dh].append(i)

        if (i + 1) % 10000 == 0:
            print(f"    hashed {i + 1}/{n}", flush=True)

    for indices in md5_groups.values():
        root = indices[0]
        for j in indices[1:]:
            uf.union(root, j)

    for indices in dhash_groups.values():
        root = indices[0]
        for j in indices[1:]:
            uf.union(root, j)

    components = defaultdict(list)
    for i, p in enumerate(paths):
        components[uf.find(i)].append(p)

    clusters = list(components.values())
    representatives = []
    for cluster in clusters:
        rep = min(cluster, key=lambda p: p.name)
        representatives.append(rep)

    stats = {
        "source_images": n,
        "bad_images": len(bad),
        "clusters": len(clusters),
        "representatives": len(representatives),
        "removed_duplicates": n - len(representatives),
        "largest_cluster_size": max(len(c) for c in clusters) if clusters else 0,
        "bad_image_examples": bad[:10],
    }
    return clusters, stats, representatives


def sample_and_split_clean(
    images_dir: Path,
    output_dir: Path,
    n_per_class: int,
    manifest_path: Path,
) -> None:
    random.seed(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "requested_n_per_class": n_per_class,
        "classes": {},
    }

    split_counts = {}
    total_copied = 0

    for src_folder_name, class_name in SOURCE_CLASS_MAP.items():
        src_folder = images_dir / src_folder_name
        if not src_folder.exists():
            raise FileNotFoundError(f"Source folder not found: {src_folder}")

        all_files = sorted(
            p for p in src_folder.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        print(f"\n{class_name}: scanning {len(all_files)} source images...", flush=True)

        clusters, stats, representatives = build_duplicate_clusters(all_files)
        print(
            f"  {stats['representatives']} unique clusters "
            f"({stats['removed_duplicates']} duplicates removed)",
            flush=True,
        )

        actual_n = min(n_per_class, len(representatives))
        if actual_n < n_per_class:
            print(
                f"  WARNING: only {len(representatives)} unique images available; "
                f"using {actual_n} instead of {n_per_class}",
                flush=True,
            )

        sampled = random.sample(representatives, actual_n)
        n_train = int(actual_n * TRAIN_RATIO)
        n_val = int(actual_n * VAL_RATIO)

        splits = {
            "train": sampled[:n_train],
            "validation": sampled[n_train : n_train + n_val],
            "test": sampled[n_train + n_val :],
        }

        split_counts[class_name] = {k: len(v) for k, v in splits.items()}
        manifest["classes"][class_name] = {
            **stats,
            "sampled": actual_n,
            "split_counts": split_counts[class_name],
        }

        for split_name, files in splits.items():
            dest_dir = output_dir / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, dest_dir / f.name)
                total_copied += 1

    manifest["output_dir"] = str(output_dir)
    manifest["total_copied"] = total_copied
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nClean dataset created at: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Seed: {SEED}  |  Requested per class: {n_per_class}")
    print(f"Split ratios: train={TRAIN_RATIO} / val={VAL_RATIO} / test={TEST_RATIO}\n")
    print(f"{'Class':<12} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print("-" * 44)
    for cls, counts in split_counts.items():
        total = sum(counts.values())
        print(f"{cls:<12} {counts['train']:>8} {counts['validation']:>8} {counts['test']:>8} {total:>8}")
    print("-" * 44)
    grand_train = sum(v["train"] for v in split_counts.values())
    grand_val = sum(v["validation"] for v in split_counts.values())
    grand_test = sum(v["test"] for v in split_counts.values())
    print(f"{'TOTAL':<12} {grand_train:>8} {grand_val:>8} {grand_test:>8} {total_copied:>8}")
    print(f"\nDone. {total_copied} deduplicated images copied.")


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicated cluster-aware sampling + YOLO split",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("../IMAGES"),
        help="Path to Kaggle IMAGES/ folder",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_10k_clean"),
        help="Where to write the clean split dataset",
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=5000,
        help="Images to sample per class after deduplication (default: 5000)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/dedup_manifest.json"),
        help="Path for deduplication manifest JSON",
    )
    args = parser.parse_args()

    sample_and_split_clean(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        n_per_class=args.n_per_class,
        manifest_path=args.manifest,
    )


if __name__ == "__main__":
    main()
