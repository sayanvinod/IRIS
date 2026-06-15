"""
Step 10 — Full Source Dataset Leakage / Duplication Audit

Audits the original Kaggle source folder, before any train/val/test split.

Checks:
  1. Exact byte duplicates (MD5) across the whole source pool.
  2. Perceptually identical images (dHash == 0 distance), including images that
     differ by filename or file bytes.
  3. Filename-number gaps among duplicate groups, which can reveal mirrored or
     merged filename ranges.
  4. Optional sampled nearest-neighbor calibration for near duplicates.

This script is read-only. It does not move, delete, or rewrite images.

Usage:
    python 10_audit_full_source.py --images-dir ../IMAGES
    python 10_audit_full_source.py --images-dir ../IMAGES --sample-near 750
"""

import argparse
import hashlib
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SOURCE_CLASS_MAP = {
    "Myopia_images": "Myopia",
    "Normal_images": "Normal",
}


def iter_source_images(images_dir: Path):
    for folder_name, class_name in SOURCE_CLASS_MAP.items():
        class_dir = images_dir / folder_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing source class folder: {class_dir}")
        for p in sorted(class_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                yield class_name, p


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


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    xor = a[:, None] ^ b[None, :]
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    xb = xor.view(np.uint8).reshape(xor.shape[0], xor.shape[1], 8)
    return lut[xb].sum(axis=2).astype(np.int16)


def file_number(path: Path) -> int | None:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else None


def summarize_groups(groups, label: str, lines: list[str], max_examples: int = 25):
    dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
    same_class = []
    cross_class = []
    extra_images = 0

    for key, members in dup_groups.items():
        extra_images += len(members) - 1
        classes = {cls for cls, _ in members}
        if len(classes) > 1:
            cross_class.append((key, members))
        else:
            same_class.append((key, members))

    lines.append(f"{label} duplicate groups: {len(dup_groups)}")
    lines.append(f"  Duplicate images beyond first representative: {extra_images}")
    lines.append(f"  Same-class duplicate groups: {len(same_class)}")
    lines.append(f"  Cross-class duplicate groups / label conflicts: {len(cross_class)}")

    class_group_counts = Counter()
    class_extra_counts = Counter()
    for members in dup_groups.values():
        classes = sorted({cls for cls, _ in members})
        key = "+".join(classes)
        class_group_counts[key] += 1
        for cls, _ in members[1:]:
            class_extra_counts[cls] += 1

    lines.append("  Groups by involved class:")
    for cls_key, n in class_group_counts.most_common():
        lines.append(f"    {cls_key:<15} {n}")

    lines.append("  Extra duplicate images by class:")
    for cls_key, n in class_extra_counts.most_common():
        lines.append(f"    {cls_key:<15} {n}")

    lines.append("")
    lines.append(f"  Example {label} duplicate groups:")
    for _, members in list(dup_groups.items())[:max_examples]:
        lines.append("    " + " | ".join(f"{cls}/{p.name}" for cls, p in members[:8]))
    if len(dup_groups) > max_examples:
        lines.append(f"    ... and {len(dup_groups) - max_examples} more groups")

    return dup_groups


def filename_gap_summary(dup_groups, lines: list[str], label: str):
    gaps = Counter()
    examples_by_gap = defaultdict(list)

    for members in dup_groups.values():
        by_class = defaultdict(list)
        for cls, p in members:
            n = file_number(p)
            if n is not None:
                by_class[cls].append((n, p.name))

        for cls, nums in by_class.items():
            nums = sorted(nums)
            for i in range(len(nums)):
                for j in range(i + 1, len(nums)):
                    gap = abs(nums[j][0] - nums[i][0])
                    gaps[(cls, gap)] += 1
                    if len(examples_by_gap[(cls, gap)]) < 3:
                        examples_by_gap[(cls, gap)].append((nums[i][1], nums[j][1]))

    lines.append("")
    lines.append(f"{label} filename-number gap summary")
    lines.append("-" * 60)
    if not gaps:
        lines.append("No numeric filename gaps found.")
        return

    for (cls, gap), count in gaps.most_common(20):
        examples = "; ".join(f"{a} <-> {b}" for a, b in examples_by_gap[(cls, gap)])
        lines.append(f"  {cls:<8} gap={gap:<8} pairs={count:<6} examples: {examples}")


def sampled_near_calibration(records, dhashes, sample_n: int, seed: int, lines: list[str]):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for cls, p in records:
        if p in dhashes:
            by_class[cls].append((cls, p))

    lines.append("")
    lines.append(f"Sampled nearest-neighbor calibration (sample_n={sample_n}, seed={seed})")
    lines.append("-" * 60)

    for cls in ("Myopia", "Normal"):
        other = "Normal" if cls == "Myopia" else "Myopia"
        sample = rng.sample(by_class[cls], min(sample_n, len(by_class[cls])))
        sample_vals = np.array([dhashes[p] for _, p in sample], dtype=np.uint64)

        same_pool = [item for item in by_class[cls] if item not in sample]
        same_vals = np.array([dhashes[p] for _, p in same_pool], dtype=np.uint64)
        cross_vals = np.array([dhashes[p] for _, p in by_class[other]], dtype=np.uint64)

        d_same = hamming_matrix(sample_vals, same_vals).min(axis=1)
        d_cross = hamming_matrix(sample_vals, cross_vals).min(axis=1)

        lines.append(f"  Source class = {cls} (n={len(sample)})")
        lines.append(
            "    closest SAME class:  "
            f"median={np.median(d_same):.1f} mean={d_same.mean():.2f} "
            f"<=0:{int((d_same == 0).sum())} <=2:{int((d_same <= 2).sum())} "
            f"<=5:{int((d_same <= 5).sum())}"
        )
        lines.append(
            "    closest OTHER class: "
            f"median={np.median(d_cross):.1f} mean={d_cross.mean():.2f} "
            f"<=0:{int((d_cross == 0).sum())} <=2:{int((d_cross <= 2).sum())} "
            f"<=5:{int((d_cross <= 5).sum())}"
        )


def main():
    ap = argparse.ArgumentParser(description="Audit full source image pool for duplicates")
    ap.add_argument("--images-dir", type=Path, default=Path("../IMAGES"))
    ap.add_argument("--report", type=Path, default=Path("results/full_source_audit.txt"))
    ap.add_argument("--sample-near", type=int, default=750)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    lines = []

    def log(s: str = ""):
        print(s, flush=True)
        lines.append(s)

    images_dir = args.images_dir
    records = list(iter_source_images(images_dir))

    log(f"FULL SOURCE AUDIT — {images_dir.resolve()}")
    log("=" * 60)
    log(f"Total images scanned: {len(records)}")
    counts = Counter(cls for cls, _ in records)
    exts = Counter(p.suffix.lower() for _, p in records)
    for cls, n in counts.most_common():
        log(f"  {cls:<8} {n}")
    log(f"Extensions: {dict(exts)}")
    log("")

    log("[1] EXACT BYTE DUPLICATES (MD5)")
    log("-" * 60)
    md5_groups = defaultdict(list)
    for i, (cls, p) in enumerate(records, 1):
        md5_groups[md5_of(p)].append((cls, p))
        if i % 10000 == 0:
            print(f"  MD5 progress: {i}/{len(records)}", flush=True)
    exact_groups = summarize_groups(md5_groups, "MD5", lines)
    filename_gap_summary(exact_groups, lines, "MD5")

    log("")
    log("[2] PERCEPTUALLY IDENTICAL IMAGES (dHash distance == 0)")
    log("-" * 60)
    dhash_groups = defaultdict(list)
    dhashes = {}
    bad_images = []
    for i, (cls, p) in enumerate(records, 1):
        try:
            h = dhash_uint64(p)
        except Exception as e:
            bad_images.append((cls, p, str(e)))
            continue
        dhashes[p] = h
        dhash_groups[h].append((cls, p))
        if i % 10000 == 0:
            print(f"  dHash progress: {i}/{len(records)}", flush=True)

    phash_groups = summarize_groups(dhash_groups, "dHash=0", lines)
    filename_gap_summary(phash_groups, lines, "dHash=0")
    if bad_images:
        log("")
        log(f"Images that could not be decoded: {len(bad_images)}")
        for cls, p, err in bad_images[:20]:
            log(f"  {cls}/{p.name}: {err}")

    if args.sample_near > 0:
        sampled_near_calibration(records, dhashes, args.sample_near, args.seed, lines)

    log("")
    log("INTERPRETATION")
    log("-" * 60)
    log("MD5 groups are byte-for-byte duplicate source files.")
    log("dHash=0 groups are visually/perceptually identical, even if bytes differ.")
    log("Cross-class groups would be label conflicts; same-class groups still matter")
    log("because random splitting can place duplicates in train and test.")
    log("Without patient IDs, this still cannot rule out left/right-eye or repeat-visit")
    log("same-patient leakage.")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines))
    print(f"\nReport written to {args.report}", flush=True)


if __name__ == "__main__":
    main()
