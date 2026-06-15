import os
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np

model = YOLO(r"C:\Users\sayan\OneDrive\Cursor-Files\ucsdproject\runs\classify\train\weights\best.pt")
val_dir = r"C:\Users\sayan\OneDrive\Cursor-Files\Desktop\myopia_holdout_test"   # change this

true_labels = []
pred_labels = []

class_names = sorted([
    d for d in os.listdir(val_dir)
    if os.path.isdir(os.path.join(val_dir, d))
])

class_to_idx = {name: i for i, name in enumerate(class_names)}

print("Class names:", class_names)
print("Class to idx:", class_to_idx)

for class_name in class_names:
    class_folder = os.path.join(val_dir, class_name)
    files = [f for f in os.listdir(class_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for file in files:
        img_path = os.path.join(class_folder, file)

        result = model(img_path)[0]
        pred_idx = int(result.probs.top1)

        true_labels.append(class_to_idx[class_name])
        pred_labels.append(pred_idx)

print("Final count:", len(true_labels), len(pred_labels))
print("Unique true labels:", np.unique(true_labels))
print("Unique pred labels:", np.unique(pred_labels))

cm = confusion_matrix(true_labels, pred_labels)
print("Confusion matrix:\n", cm)
print("Confusion matrix shape:", cm.shape)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report:\n")
print(classification_report(true_labels, pred_labels, target_names=class_names))