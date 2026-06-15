from pathlib import Path
from PIL import Image
import imagehash
import shutil

DATASET_ROOT = Path(r"C:\Users\sayan\.cache\kagglehub\datasets\yolo_dataset\myopia-image-dataset\versions\1")
OUTPUT_DUPES = Path(r"C:\Users\sayan\OneDrive\Cursor-Files\Desktop\removed_duplicates")

# Lower = stricter. Start with 0 for exact duplicates only.
# Then try 2 or 3 for near-duplicates.
HASH_DISTANCE_THRESHOLD = 0

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

OUTPUT_DUPES.mkdir(parents=True, exist_ok=True)

for class_dir in ["Normal", "Myopia"]:
    folder = DATASET_ROOT / class_dir
    dupe_folder = OUTPUT_DUPES / class_dir
    dupe_folder.mkdir(parents=True, exist_ok=True)

    kept = []  # list of (path, hash)
    removed_count = 0

    for img_path in sorted(folder.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            h = imagehash.phash(img)
        except Exception as e:
            print(f"Skipping {img_path.name}: {e}")
            continue

        is_duplicate = False

        for kept_path, kept_hash in kept:
            dist = h - kept_hash
            if dist <= HASH_DISTANCE_THRESHOLD:
                print(f"Duplicate found: {img_path.name} ~ {kept_path.name} (dist={dist})")
                shutil.move(str(img_path), str(dupe_folder / img_path.name))
                removed_count += 1
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append((img_path, h))

    print(f"{class_dir}: removed {removed_count} duplicates, kept {len(kept)}")