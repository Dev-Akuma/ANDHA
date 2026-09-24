import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as patches
from pathlib import Path

Path("figures").mkdir(parents=True, exist_ok=True)

def draw_box(ax, text, x, y, w, h, bg_color="#e0e0e0"):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=bg_color, ec="black", lw=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=12, fontweight='bold', wrap=True)

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=2, color="black"))

# 1. Pipeline Diagram
fig, ax = plt.subplots(figsize=(10, 4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis('off')

draw_box(ax, "WLASL100\nVideo Dataset", 0.5, 1.5, 2, 1, "#add8e6")
draw_arrow(ax, 2.5, 2, 3.5, 2)
draw_box(ax, "MediaPipe\nLandmark Extraction", 3.5, 1.5, 2.5, 1, "#add8e6")
draw_arrow(ax, 6, 2, 7, 2)
draw_box(ax, "Synthetic Data\nAugmentation", 7, 1.5, 2, 1, "#98fb98")

draw_arrow(ax, 1.5, 1.5, 1.5, 0.5)
draw_box(ax, "Top-10 Class\nStratification", 0.5, -0.5, 2, 1, "#ffcccb")
# Just simple horizontal flow for the pilot
plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 4))
ax.set_xlim(0, 12)
ax.set_ylim(0, 4)
ax.axis('off')

# Improved pipeline flow
draw_box(ax, "1. Video Input\n(Subset)", 0.2, 1.5, 1.8, 1, "#e6f2ff")
draw_arrow(ax, 2.0, 2, 2.5, 2)
draw_box(ax, "2. Modality Extraction\n(Hands/Face/Pose)", 2.5, 1.5, 2.5, 1, "#cce6ff")
draw_arrow(ax, 5.0, 2, 5.5, 2)
draw_box(ax, "3. Synthetic Augmentation\n(Rotate, Scale, Temp, Jitter)", 5.5, 1.5, 3.0, 1, "#e6ffe6")
draw_arrow(ax, 8.5, 2, 9.0, 2)
draw_box(ax, "4. BiGRU Classifier\n(Sequence Modeling)", 9.0, 1.5, 2.5, 1, "#ffe6e6")

plt.tight_layout()
plt.savefig("figures/pipeline_diagram.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Model Architecture Diagram
fig, ax = plt.subplots(figsize=(6, 8))
ax.set_xlim(0, 6)
ax.set_ylim(0, 10)
ax.axis('off')

draw_box(ax, "Input Sequence\n(30 frames x Features)", 1.5, 8.5, 3, 1, "#f2f2f2")
draw_arrow(ax, 3, 8.5, 3, 7.5)
draw_box(ax, "BiGRU Layer 1\n(128 units, dropout=0.3)", 1.5, 6.5, 3, 1, "#e6e6fa")
draw_arrow(ax, 3, 6.5, 3, 5.5)
draw_box(ax, "BiGRU Layer 2\n(128 units, dropout=0.3)", 1.5, 4.5, 3, 1, "#e6e6fa")
draw_arrow(ax, 3, 4.5, 3, 3.5)
draw_box(ax, "Layer Norm + Dropout", 1.5, 2.5, 3, 1, "#fff0f5")
draw_arrow(ax, 3, 2.5, 3, 1.5)
draw_box(ax, "Linear Classifier\n(10 classes)", 1.5, 0.5, 3, 1, "#ffe4e1")

plt.tight_layout()
plt.savefig("figures/model_architecture.png", dpi=300, bbox_inches='tight')
plt.close()

print("Diagrams generated successfully.")
