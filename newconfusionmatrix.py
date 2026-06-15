import os
from ultralytics import YOLO

model = YOLO(r"C:\Users\sayan\OneDrive\Cursor-Files\runs\classify\train4\weights\best.pt")
val_dir = r"C:\Users\sayan\.cache\kagglehub\datasets\yolo_dataset\myopia-image-dataset\versions\1\Validation"

true_labels = []
pred_labels = []

class_names = sorted([
    d for d in os.listdir(val_dir)
    if os.path.isdir(os.path.join(val_dir, d))
])

print("Detected classes:", class_names)

for class_name in class_names:
    class_folder = os.path.join(val_dir, class_name)
    files = [f for f in os.listdir(class_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print(f"{class_name}: {len(files)} images")

    for file in files:
        img_path = os.path.join(class_folder, file)
        print("Predicting:", img_path)

        result = model(img_path)[0]
        pred_idx = int(result.probs.top1)

        true_labels.append(class_names.index(class_name))
        pred_labels.append(pred_idx)

print("Final count:", len(true_labels), len(pred_labels))