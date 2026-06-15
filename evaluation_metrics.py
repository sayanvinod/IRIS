from ultralytics import YOLO
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# -----------------------
# Load trained model
# -----------------------

model = YOLO(r"C:\Users\sayan\OneDrive\Cursor-Files\runs\classify\train\weights\best.pt")

# -----------------------
# Dataset location
# -----------------------

val_dir = r"C:\Users\sayan\OneDrive\Cursor-Files\Desktop\myopia_holdout_test"

# -----------------------
# Collect predictions
# -----------------------

true_labels = []
pred_labels = []
pred_probs = []

class_names = sorted([
    d for d in os.listdir(val_dir)
    if os.path.isdir(os.path.join(val_dir, d))
])

class_to_idx = {name:i for i,name in enumerate(class_names)}

for class_name in class_names:
    folder = os.path.join(val_dir, class_name)

    for file in os.listdir(folder):

        if not file.lower().endswith((".jpg",".png",".jpeg")):
            continue

        img_path = os.path.join(folder, file)

        result = model(img_path)[0]

        pred = int(result.probs.top1)
        prob = float(result.probs.top1conf)

        true_labels.append(class_to_idx[class_name])
        pred_labels.append(pred)
        pred_probs.append(prob)

print("Images evaluated:", len(true_labels))

# -----------------------
# Metrics
# -----------------------

accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels)
recall = recall_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)

print("\nClassification Report\n")
print(classification_report(true_labels, pred_labels, target_names=class_names))

# -----------------------
# Confusion Matrix
# -----------------------

cm = confusion_matrix(true_labels, pred_labels)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# -----------------------
# Normalized Confusion Matrix
# -----------------------

cm_norm = confusion_matrix(true_labels, pred_labels, normalize="true")

disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
disp.plot(cmap="Blues", values_format=".2f")
plt.title("Normalized Confusion Matrix")
plt.show()

# -----------------------
# Precision Recall F1 Graph
# -----------------------

metrics = [precision, recall, f1]
names = ["Precision", "Recall", "F1 Score"]

plt.bar(names, metrics)
plt.ylim(0,1)
plt.title("Model Performance Metrics")
plt.show()