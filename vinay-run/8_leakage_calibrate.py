"""
Step 8 — Calibrate the near-duplicate signal.

Two questions:
  (A) What is the Hamming-distance distribution between test and train/val?
  (B) Is the near-dup signal CLASS-SPECIFIC (real duplication) or just
      "all fundus images look alike" (threshold artifact)?

For (B) we compare, for each TEST image, its closest match in train/val
within the SAME class vs. its closest match in the OTHER class. If same-class
distances are systematically much smaller, the duplication is real. If
same-class and cross-class look identical, dHash is only capturing gross
fundus structure and the threshold is too loose.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SPLITS = ["train", "validation", "test"]


def iter_images(data_dir: Path):
    for split in SPLITS:
        sd = data_dir / split
        if not sd.exists():
            continue
        for cls_dir in sorted(sd.iterdir()):
            if cls_dir.is_dir():
                for p in sorted(cls_dir.iterdir()):
                    if p.suffix.lower() in IMAGE_EXTS:
                        yield split, cls_dir.name, p


def dhash_uint64(path: Path, hash_size: int = 8) -> int:
    img = Image.open(path).convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data_10k"))
    args = ap.parse_args()

    records = list(iter_images(args.data_dir))
    hashes = {p: dhash_uint64(p) for _, _, p in records}

    def arrays(split, cls=None):
        paths, vals = [], []
        for s, c, p in records:
            if s == split and (cls is None or c == cls):
                paths.append((s, c, p))
                vals.append(hashes[p])
        return paths, np.array(vals, dtype=np.uint64)

    print(f"CALIBRATION — {args.data_dir}")
    print("=" * 60)

    for cls in ("Myopia", "Normal"):
        other = "Normal" if cls == "Myopia" else "Myopia"
        _, test_v = arrays("test", cls)

        # same-class pool = train+val of SAME class
        same_v = []
        for sp in ("train", "validation"):
            _, v = arrays(sp, cls)
            if len(v):
                same_v.append(v)
        same_v = np.concatenate(same_v)

        # cross-class pool = train+val of OTHER class
        cross_v = []
        for sp in ("train", "validation"):
            _, v = arrays(sp, other)
            if len(v):
                cross_v.append(v)
        cross_v = np.concatenate(cross_v)

        d_same = hamming_matrix(test_v, same_v).min(axis=1)
        d_cross = hamming_matrix(test_v, cross_v).min(axis=1)

        print(f"\nTEST class = {cls}  (n={len(test_v)})")
        print(f"  closest match in SAME class  ({cls}):  "
              f"median={np.median(d_same):.1f}  mean={d_same.mean():.2f}  "
              f"<=0: {(d_same==0).sum()}  <=2: {(d_same<=2).sum()}  <=5: {(d_same<=5).sum()}")
        print(f"  closest match in OTHER class ({other}): "
              f"median={np.median(d_cross):.1f}  mean={d_cross.mean():.2f}  "
              f"<=0: {(d_cross==0).sum()}  <=2: {(d_cross<=2).sum()}  <=5: {(d_cross<=5).sum()}")

    # Overall test->train/val same-class distance histogram
    print("\n" + "=" * 60)
    print("Interpretation:")
    print("  If SAME-class near-dup counts >> OTHER-class counts, the duplication")
    print("  is real and class-specific (true leakage). If they are similar, the")
    print("  near-dup threshold is mostly capturing generic fundus structure.")


if __name__ == "__main__":
    main()
