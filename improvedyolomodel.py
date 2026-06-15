from ultralytics import YOLO
from pathlib import Path
from PIL import Image

MODEL_PATH = r"C:\Users\sayan\OneDrive\Cursor-Files\myopia_training\clean_split_final\weights\best.pt"
TEST_ROOT = Path(r"C:\Users\sayan\OneDrive\Cursor-Files\Desktop\myopia_holdout_test")

model = YOLO(MODEL_PATH)

correct = 0
total = 0

for class_dir in ["Normal", "Myopia"]:
    folder = TEST_ROOT / class_dir

    if not folder.exists():
        print(f"Missing folder: {folder}")
        exit()

    for img_path in folder.iterdir():
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            continue

        image = Image.open(img_path).convert("RGB")
        results = model(image)

        probs = results[0].probs
        pred_idx = int(probs.top1)
        pred_conf = float(probs.top1conf)
        pred_label = results[0].names[pred_idx]

        is_correct = pred_label.lower() == class_dir.lower()
        correct += int(is_correct)
        total += 1

        print(
            f"{img_path.name:30} true={class_dir:7} "
            f"pred={pred_label:7} conf={pred_conf:.4f} correct={is_correct}"
        )

print(f"\nAccuracy on external test set: {correct}/{total} = {correct/total:.4f}")