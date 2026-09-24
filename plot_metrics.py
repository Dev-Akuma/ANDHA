import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Create figures directory
Path("figures").mkdir(parents=True, exist_ok=True)

# Set publication style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

df = pd.read_csv("pilot_results.csv")

# 1. accuracy_comparison.png
plt.figure(figsize=(8, 5))
ax = sns.barplot(data=df, x="augmentation_percent", y="test_accuracy", hue="modality")
plt.title("Test Accuracy Comparison across Modalities & Augmentation")
plt.xlabel("Synthetic Augmentation Level (%)")
plt.ylabel("Test Accuracy")
plt.ylim(0, 0.5)
for i in ax.containers:
    ax.bar_label(i, fmt='%.3f', padding=3)
plt.tight_layout()
plt.savefig("figures/accuracy_comparison.png", dpi=300)
plt.close()

# 2. macro_f1_comparison.png
plt.figure(figsize=(8, 5))
ax = sns.barplot(data=df, x="augmentation_percent", y="test_f1", hue="modality")
plt.title("Macro F1 Score Comparison")
plt.xlabel("Synthetic Augmentation Level (%)")
plt.ylabel("Test Macro F1")
plt.ylim(0, 0.4)
for i in ax.containers:
    ax.bar_label(i, fmt='%.3f', padding=3)
plt.tight_layout()
plt.savefig("figures/macro_f1_comparison.png", dpi=300)
plt.close()

# 3. precision_recall_comparison.png
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(data=df, x="augmentation_percent", y="test_precision", hue="modality", ax=axes[0])
axes[0].set_title("Test Macro Precision")
axes[0].set_xlabel("Synthetic Augmentation Level (%)")
axes[0].set_ylabel("Precision")
axes[0].set_ylim(0, 0.4)
sns.barplot(data=df, x="augmentation_percent", y="test_recall", hue="modality", ax=axes[1])
axes[1].set_title("Test Macro Recall")
axes[1].set_xlabel("Synthetic Augmentation Level (%)")
axes[1].set_ylabel("Recall")
axes[1].set_ylim(0, 0.4)
plt.tight_layout()
plt.savefig("figures/precision_recall_comparison.png", dpi=300)
plt.close()

# 4. training_time_comparison.png
plt.figure(figsize=(8, 5))
ax = sns.barplot(data=df, x="augmentation_percent", y="training_time_s", hue="modality")
plt.title("Training Time Comparison (15 Epochs)")
plt.xlabel("Synthetic Augmentation Level (%)")
plt.ylabel("Total Training Time (seconds)")
for i in ax.containers:
    ax.bar_label(i, fmt='%.1f', padding=3)
plt.tight_layout()
plt.savefig("figures/training_time_comparison.png", dpi=300)
plt.close()

# 5. latency_comparison.png
plt.figure(figsize=(8, 5))
ax = sns.barplot(data=df, x="augmentation_percent", y="test_latency_ms", hue="modality")
plt.title("Inference Latency per Sample (CPU)")
plt.xlabel("Synthetic Augmentation Level (%)")
plt.ylabel("Latency (milliseconds)")
for i in ax.containers:
    ax.bar_label(i, fmt='%.1f', padding=3)
plt.tight_layout()
plt.savefig("figures/latency_comparison.png", dpi=300)
plt.close()

# 6. accuracy_heatmap.png
plt.figure(figsize=(6, 4))
acc_pivot = df.pivot(index="modality", columns="augmentation_percent", values="test_accuracy")
sns.heatmap(acc_pivot, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Test Accuracy'})
plt.title("Test Accuracy Heatmap")
plt.xlabel("Augmentation (%)")
plt.ylabel("Modality")
plt.tight_layout()
plt.savefig("figures/accuracy_heatmap.png", dpi=300)
plt.close()

# 7. macro_f1_heatmap.png
plt.figure(figsize=(6, 4))
f1_pivot = df.pivot(index="modality", columns="augmentation_percent", values="test_f1")
sns.heatmap(f1_pivot, annot=True, fmt=".3f", cmap="YlOrRd", cbar_kws={'label': 'Test Macro F1'})
plt.title("Test Macro F1 Heatmap")
plt.xlabel("Augmentation (%)")
plt.ylabel("Modality")
plt.tight_layout()
plt.savefig("figures/macro_f1_heatmap.png", dpi=300)
plt.close()

# 8. confusion_matrix.png
with open("pilot_confusion_matrices.json", "r") as f:
    cms = json.load(f)

# Classes in alphabetical order as processed by make_dataloaders
classes = ['bed', 'before', 'bowling', 'candy', 'computer', 'cool', 'drink', 'go', 'mother', 'thin']
cm_run2 = np.array(cms["run_2"])

plt.figure(figsize=(8, 6))
sns.heatmap(cm_run2, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix: Run 2 (Hands, 50% Augmentation)")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=300)
plt.close()

print("All metrics figures generated successfully.")
