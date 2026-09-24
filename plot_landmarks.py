import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Need to import our custom tasks adapter
import extract_landmarks

# Find a real video
video_paths = list(Path("WSLA100/wlasl100-new/WLASL_100/train").glob("**/*.mp4"))
if not video_paths:
    raise FileNotFoundError("No videos found to generate landmark image")

video_path = video_paths[0]
cap = cv2.VideoCapture(str(video_path))
success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Failed to read video frame")

FRAME_W, FRAME_H = 640, 480
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame = cv2.resize(frame, (FRAME_W, FRAME_H))

# Load our local holistic model
model_path = Path("WSLA100/processed_landmarks/holistic_landmarker.task")
holistic = extract_landmarks.create_holistic(model_path)
results = holistic.process(frame)

def extract_pts(landmarks):
    if not landmarks: return []
    pts = landmarks if isinstance(landmarks, list) else landmarks.landmark
    return [(p.x * FRAME_W, p.y * FRAME_H) for p in pts]

lh = extract_pts(results.left_hand_landmarks)
rh = extract_pts(results.right_hand_landmarks)
face = extract_pts(results.face_landmarks)
pose = extract_pts(results.pose_landmarks)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# 1. Original
axes[0].imshow(frame)
axes[0].set_title("Original Frame")
axes[0].axis("off")

def plot_on_ax(ax, pts, color, size):
    if pts:
        pts = np.array(pts)
        ax.scatter(pts[:, 0], pts[:, 1], c=color, s=size)

# 2. Hands
axes[1].imshow(frame)
plot_on_ax(axes[1], lh, 'red', 10)
plot_on_ax(axes[1], rh, 'blue', 10)
axes[1].set_title("Hands Modality")
axes[1].axis("off")

# 3. Hands + Face
axes[2].imshow(frame)
plot_on_ax(axes[2], lh, 'red', 10)
plot_on_ax(axes[2], rh, 'blue', 10)
plot_on_ax(axes[2], face, 'green', 1)
axes[2].set_title("Hand + Face Modality")
axes[2].axis("off")

# 4. Hands + Face + Pose
axes[3].imshow(frame)
plot_on_ax(axes[3], lh, 'red', 10)
plot_on_ax(axes[3], rh, 'blue', 10)
plot_on_ax(axes[3], face, 'green', 1)
plot_on_ax(axes[3], pose, 'magenta', 15)
axes[3].set_title("Hand + Face + Pose Modality")
axes[3].axis("off")

plt.tight_layout()
plt.savefig("figures/landmark_modalities.png", dpi=300)
plt.close()

holistic.close()
print("Landmark modalities figure generated.")
