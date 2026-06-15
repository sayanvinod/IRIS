from pathlib import Path
import random
import shutil

SOURCE = Path(r"C:\Users\sayan\OneDrive\Cursor-Files\Desktop\LIMIT_IMAGES")
DEST = Path(r"C:\Users\sayan\OneDrive\Cursor-Files\Desktop\LIMIT_IMAGES_SPLIT_CLEAN")

SOURCE_CLASS_MAP = {
    "Normal_images": "Normal",
    "Myopia_images": "Myopia",
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
random.seed(42)

for src_name, clean_name in SOURCE_CLASS_MAP.items():
    src_folder = SOURCE / src_name
    files = [p for p in src_folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    random.shuffle(files)

    total = len(files)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": files[:train_end],
        "validation": files[train_end:val_end],
        "test": files[val_end:],
    }

    for split_name, split_files in splits.items():
        out_folder = DEST / split_name / clean_name
        out_folder.mkdir(parents=True, exist_ok=True)

        for img in split_files:
            shutil.copy2(img, out_folder / img.name)

    print(f"{clean_name}: total={total}, train={len(splits['train'])}, val={len(splits['validation'])}, test={len(splits['test'])}")

print("Clean dataset split finished.")