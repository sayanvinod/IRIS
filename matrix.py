import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve

# =========================================================
# 1. ENTER YOUR CUSTOM CONFUSION MATRIX HERE
# =========================================================
# Format:
# rows = true labels
# columns = predicted labels
#
#           Pred Normal   Pred Myopia
# True Normal    TN            FP
# True Myopia    FN            TP

cm = np.array([
    [373, 627],   # True Normal
    [13, 987]     # True Myopia
])

class_names = ["Normal", "Myopia"]

# =========================================================
# 2. COMPUTE METRICS FROM CUSTOM MATRIX
# =========================================================

TN, FP = cm[0]
FN, TP = cm[1]

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

print("=== Metrics from Custom Confusion Matrix ===")
print(f"Accuracy:    {accuracy:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}")
print(f"F1 Score:    {f1:.4f}")
print(f"Specificity: {specificity:.4f}")

# =========================================================
# 3. PLOT CUSTOM CONFUSION MATRIX
# =========================================================

plt.figure(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", colorbar=False, values_format="d")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("custom_confusion_matrix.png", dpi=300)
plt.show()

# =========================================================
# 4. PLOT NORMALIZED CONFUSION MATRIX
# =========================================================

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
disp.plot(cmap="Blues", colorbar=False, values_format=".2f")
plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.savefig("normalized_confusion_matrix.png", dpi=300)
plt.show()

# =========================================================
# 5. PLOT METRICS BAR CHART
# =========================================================

metric_names = ["Accuracy", "Precision", "Recall", "F1", "Specificity"]
metric_values = [accuracy, precision, recall, f1, specificity]

plt.figure(figsize=(8, 5))
bars = plt.bar(metric_names, metric_values)
plt.ylim(0, 1.12)
plt.ylabel("Score")
plt.title("Performance Metrics", pad=16)

for bar, val in zip(bars, metric_values):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        min(val + 0.02, 1.09),
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.tight_layout()
plt.show()