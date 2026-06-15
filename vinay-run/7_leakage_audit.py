"""
Step 7 — Data Leakage Audit

Checks a split dataset (e.g. data_10k/) for the two forms of leakage that would
inflate test-set metrics:

  1. EXACT content duplicates   — identical image bytes (MD5) appearing in more
                                  than one split, regardless of filename.
  2. NEAR duplicates            — visually near-identical images (perceptual
                                  dHash, Hamming distance) shared between the
                                  test set and the train/val sets.

Both are computed on RAW PIXELS, so renamed copies, re-saves, and crops are
caught even when filenames differ.

Usage:
    python 7_leakage_audit.py --data-dir data_10k
    python 7_leakage_audit.py --data-dir data_10k --near-threshold 5 --report results/leakage_data_10k.txt

Notes:
  - dHash is computed on a 9x8 grayscale resize -> 64-bit fingerprint.
  - Hamming distance 0 = pixel-level near-identical; <=5 = strong near-duplicate.
  - This audits the SAMPLED split (what the model actually saw), which is what
    matters for the reported metrics.
"""

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SPLITS = ["train", "validation", "test"]


def iter_images(data_dir: Path):
    for split in SPLITS:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() in IMAGE_EXTS:
                    yield split, cls_dir.name, p


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash_uint64(path: Path, hash_size: int = 8) -> int:
    """Difference hash -> 64-bit integer fingerprint."""
    img = Image.open(path).convert("L").resize(
        (hash_size + 1, hash_size), Image.LANCZOS
    )
    arr = np.asarray(img, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]          # 8x8 boolean
    bits = diff.flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distance between two arrays of uint64 hashes.

    Returns int8 matrix of shape (len(a), len(b)).
    """
    xor = a[:, None] ^ b[None, :]
    # popcount on uint64 via byte view lookup table
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    xb = xor.view(np.uint8).reshape(xor.shape[0], xor.shape[1], 8)
    return lut[xb].sum(axis=2).astype(np.int16)


def main():
    ap = argparse.ArgumentParser(description="Data leakage audit for a split dataset")
    ap.add_argument("--data-dir", type=Path, default=Path("data_10k"))
    ap.add_argument("--near-threshold", type=int, default=5,
                    help="Max Hamming distance to flag as a near-duplicate (default 5)")
    ap.add_argument("--report", type=Path, default=None,
                    help="Optional path to write a text report")
    args = ap.parse_args()

    data_dir = args.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"{data_dir} does not exist")

    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log(f"LEAKAGE AUDIT — {data_dir}")
    log("=" * 60)

    # ---- Gather images ----
    records = list(iter_images(data_dir))
    log(f"Total images scanned: {len(records)}")
    by_split = defaultdict(int)
    for split, cls, _ in records:
        by_split[(split, cls)] += 1
    for (split, cls), n in sorted(by_split.items()):
        log(f"  {split:<12} {cls:<10} {n}")
    log("")

    # ---- 1. EXACT content duplicates (MD5) ----
    log("[1] EXACT CONTENT DUPLICATES (MD5 over raw bytes)")
    log("-" * 60)
    md5_map = defaultdict(list)  # md5 -> list of (split, cls, path)
    md5_cache = {}
    for split, cls, p in records:
        h = md5_of(p)
        md5_cache[p] = h
        md5_map[h].append((split, cls, p))

    exact_dupe_groups = {h: v for h, v in md5_map.items() if len(v) > 1}
    cross_split_exact = []
    cross_class_exact = []
    for h, members in exact_dupe_groups.items():
        splits_involved = {m[0] for m in members}
        classes_involved = {m[1] for m in members}
        if len(splits_involved) > 1:
            cross_split_exact.append((h, members))
        if len(classes_involved) > 1:
            cross_class_exact.append((h, members))

    log(f"Total exact-duplicate groups (>=2 identical images): {len(exact_dupe_groups)}")
    log(f"  ...of which span MORE THAN ONE SPLIT (LEAKAGE): {len(cross_split_exact)}")
    log(f"  ...of which span MORE THAN ONE CLASS (label issue): {len(cross_class_exact)}")
    for h, members in cross_split_exact[:25]:
        log(f"  LEAK exact: " + " | ".join(f"{s}/{c}/{p.name}" for s, c, p in members))
    if len(cross_split_exact) > 25:
        log(f"  ... and {len(cross_split_exact) - 25} more cross-split exact groups")
    log("")

    # ---- 2. NEAR duplicates (perceptual dHash) test vs train/val ----
    log(f"[2] NEAR DUPLICATES (dHash, Hamming <= {args.near_threshold})")
    log("-" * 60)
    log("Computing perceptual hashes...")
    hashes = {}
    for split, cls, p in records:
        try:
            hashes[p] = dhash_uint64(p)
        except Exception as e:
            log(f"  WARN could not hash {p}: {e}")

    def split_arrays(target_split):
        paths, vals = [], []
        for split, cls, p in records:
            if split == target_split and p in hashes:
                paths.append((split, cls, p))
                vals.append(hashes[p])
        return paths, np.array(vals, dtype=np.uint64)

    test_paths, test_vals = split_arrays("test")
    seen_paths, seen_vals = [], []
    for tsplit in ("train", "validation"):
        pp, vv = split_arrays(tsplit)
        seen_paths.extend(pp)
        if len(vv):
            seen_vals.append(vv)
    seen_vals = np.concatenate(seen_vals) if seen_vals else np.array([], dtype=np.uint64)

    log(f"Test images: {len(test_paths)}   |   Train+Val images: {len(seen_paths)}")

    near_hits = []           # (dist, test_record, seen_record)
    exact_phash_hits = 0
    if len(test_vals) and len(seen_vals):
        # chunk over test to bound memory
        CHUNK = 256
        for start in range(0, len(test_vals), CHUNK):
            end = min(start + CHUNK, len(test_vals))
            dmat = hamming_matrix(test_vals[start:end], seen_vals)
            ti, si = np.where(dmat <= args.near_threshold)
            for k in range(len(ti)):
                dist = int(dmat[ti[k], si[k]])
                tr = test_paths[start + ti[k]]
                sr = seen_paths[si[k]]
                if dist == 0:
                    exact_phash_hits += 1
                near_hits.append((dist, tr, sr))

    near_hits.sort(key=lambda x: x[0])
    n_test_with_near = len({id(h[1][2]) for h in near_hits})
    log(f"Test images with >=1 near-duplicate in train/val: {n_test_with_near} "
        f"({100*n_test_with_near/max(len(test_paths),1):.2f}% of test)")
    log(f"Near-duplicate pairs found (Hamming <= {args.near_threshold}): {len(near_hits)}")
    log(f"  ...of which Hamming == 0 (perceptually identical): {exact_phash_hits}")
    log("")
    log("Closest matches (test  <->  train/val):")
    for dist, tr, sr in near_hits[:30]:
        log(f"  dist={dist}  TEST {tr[1]}/{tr[2].name}   ~=   {sr[0]} {sr[1]}/{sr[2].name}")
    if len(near_hits) > 30:
        log(f"  ... and {len(near_hits) - 30} more pairs")
    log("")

    # ---- Verdict ----
    log("VERDICT")
    log("-" * 60)
    leak_exact = len(cross_split_exact)
    leak_near = n_test_with_near
    if leak_exact == 0 and leak_near == 0:
        log("PASS — no exact cross-split duplicates and no near-duplicates between")
        log("       test and train/val at the chosen threshold. No evidence of the")
        log("       file/pixel-level leakage checked here.")
    else:
        log(f"ATTENTION — {leak_exact} exact cross-split dup group(s); "
            f"{leak_near} test image(s) with near-dupes in train/val.")
        log("       Review the pairs above. Note: this cannot detect SAME-PATIENT")
        log("       leakage (e.g. left/right eye, repeat visits) without metadata.")
    log("")
    log("LIMITATION: filename- and pixel-based audits cannot rule out same-patient")
    log("leakage when the dataset ships no patient IDs. Treat that as an external-")
    log("validity caveat, not something this script can resolve.")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("\n".join(lines))
        print(f"\nReport written to {args.report}")


if __name__ == "__main__":
    main()
