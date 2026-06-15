from pathlib import Path
from PIL import Image
import imagehash

TRAIN_DIR = Path(r"C:\Users\sayan\.cache\kagglehub\datasets\yolo_dataset\myopia-image-dataset\versions\1\train")
HOLDOUT_DIR = Path(r"C:\Users\sayan\OneDrive\Cursor-Files\Desktop\myopia_holdout_test")

train_hashes = []

for p in TRAIN_DIR.rglob("*"):
    if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        try:
            img = Image.open(p).convert("RGB")
            h = imagehash.phash(img)
            train_hashes.append((p, h))
        except:
            pass

matches = []

for p in HOLDOUT_DIR.rglob("*"):
    if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        try:
            img = Image.open(p).convert("RGB")
            h = imagehash.phash(img)

            for train_path, train_h in train_hashes:
                dist = h - train_h
                if dist <= 3:   # small distance = near-duplicate
                    matches.append((p, train_path, dist))
                    break
        except:
            pass

print(f"Near-duplicate matches found: {len(matches)}")
for holdout, train_img, dist in matches[:20]:
    print("HOLDOUT:", holdout)
    print("TRAIN:  ", train_img)
    print("DIST:   ", dist)
    print("---")