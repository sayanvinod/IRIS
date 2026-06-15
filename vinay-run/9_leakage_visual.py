"""Visual confirmation of leakage pairs -> figures/leakage_pairs.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

D = Path("data_10k")
pairs = [
    ("MD5-identical (different filenames)",
     D/"train/Myopia/myopia47340.png", D/"test/Myopia/myopia26307.png"),
    ("MD5-identical (different extension/range)",
     D/"train/Myopia/myopia27023.jpg", D/"test/Myopia/myopia47921.jpg"),
    ("dHash=0 (perceptually identical)",
     D/"train/Myopia/myopia6956.png", D/"test/Myopia/myopia10016.png"),
    ("dHash=0 (perceptually identical)",
     D/"train/Myopia/myopia7430.png", D/"test/Myopia/myopia10161.png"),
]

fig, axes = plt.subplots(len(pairs), 2, figsize=(7, 3.2*len(pairs)))
for i, (label, a, b) in enumerate(pairs):
    for j, (p, side) in enumerate([(a, "TRAIN"), (b, "TEST")]):
        ax = axes[i, j]
        if p.exists():
            ax.imshow(Image.open(p))
        ax.set_title(f"{side}\n{p.name}", fontsize=8)
        ax.axis("off")
    axes[i, 0].set_ylabel(label, fontsize=9)
fig.suptitle("Data leakage: identical images split across train/test", fontsize=11)
fig.tight_layout()
out = Path("figures/leakage_pairs.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved {out}")
