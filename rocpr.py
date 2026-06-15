import os
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# =========================
# LOAD MODEL
# =========================
model = YOLO(r"C:\Users\sayan\OneDrive\Cursor-Files\runs\classify\train4\weights\best.pt")

# =========================
# VALIDATION FOLDER
# =========================
val_dir = r"C:\Users\sayan\OneDrive\Cursor-Files\Desktop\myopia_holdout_test"

# =========================
# CLASS ORDER
# Make sure this matches your folders
# =========================
class_names = sorted([
    d for d in os.listdir(val_dir)
    if os.path.isdir(os.path.join(val_dir, d))
])

print("Detected classes:", class_names)

class_to_idx = {name: i for i, name in enumerate(class_names)}

# We want probability of the positive class: "Myopia"
positive_class_name = "Myopia"
positive_class_idx = class_to_idx[positive_class_name]

true_labels = []
pred_probs = []

# =========================
# COLLECT TRUE LABELS + PROBABILITIES
# =========================
for class_name in class_names:
    class_folder = os.path.join(val_dir, class_name)

    for file in os.listdir(class_folder):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(class_folder, file)
        result = model(img_path, verbose=False)[0]

        # true label
        true_labels.append(class_to_idx[class_name])

        # probability for the positive class (Myopia)
        probs = result.probs.data.cpu().numpy()
        pred_probs.append(float(probs[positive_class_idx]))

true_labels = np.array(true_labels)
pred_probs = np.array(pred_probs)

print("Collected labels:", len(true_labels))
print("Collected probabilities:", len(pred_probs))
print("Sample probabilities:", pred_probs[:10])

# =========================
# ROC CURVE
# =========================
fpr, tpr, _ = roc_curve(true_labels, pred_probs, pos_label=positive_class_idx)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve", pad=12)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300)
plt.show()

# =========================
# PRECISION-RECALL CURVE
# =========================
precision, recall, _ = precision_recall_curve(
    true_labels == positive_class_idx,
    pred_probs
)
ap_score = average_precision_score(true_labels == positive_class_idx, pred_probs)

plt.figure(figsize=(6, 6))
plt.plot(recall, precision, linewidth=2, label=f"AP = {ap_score:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve", pad=12)
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("pr_curve.png", dpi=300)
plt.show()