# --- Cell 0 ---
# Install the dependencies used by the local extraction and training scripts.
%pip install -r requirements.txt
# --- Cell 1 ---
# Download the WLASL100 videos with KaggleHub and copy them into the project folder.
from pathlib import Path
import shutil
import kagglehub

DATASET_ID = "thtrnphc/wlasl100-new"
PROJECT_ROOT = Path.cwd()
DOWNLOAD_ROOT = PROJECT_ROOT / "WSLA100"
DATASET_DESTINATION = DOWNLOAD_ROOT / "wlasl100-new"

print("Downloading dataset from Kaggle...")
downloaded_path = Path(kagglehub.dataset_download(DATASET_ID))
print(f"KaggleHub cache: {downloaded_path}")

# KaggleHub may return a directory containing versions/1.
source_path = downloaded_path / "versions" / "1"
if not source_path.exists():
    source_path = downloaded_path

DATASET_DESTINATION.mkdir(parents=True, exist_ok=True)
shutil.copytree(source_path, DATASET_DESTINATION, dirs_exist_ok=True)

video_paths = sorted(DATASET_DESTINATION.glob("**/*.mp4"))
print(f"Dataset copied to: {DATASET_DESTINATION.resolve()}")
print(f"Videos available: {len(video_paths)}")
# --- Cell 2 ---
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

print("Working directory:", Path.cwd())

dataset_root = Path("WSLA100/wlasl100-new/WLASL_100")
output_root = Path("WSLA100/processed_landmarks")
TARGET_FRAMES = 30

if not dataset_root.exists():
    print("Path not found. Available directories:")
    for path in Path.cwd().iterdir():
        print(" -", path)

    raise FileNotFoundError(f"Dataset path does not exist: {dataset_root.resolve()}")

video_paths = sorted(dataset_root.glob("**/*.mp4"))

if not video_paths:
    raise FileNotFoundError(f"No videos found below {dataset_root.resolve()}")

INPUT_ROOT = dataset_root
OUTPUT_ROOT = output_root

print(f"Dataset: {dataset_root.resolve()}")
print(f"Videos found: {len(video_paths)}")
print(f"Landmark output: {OUTPUT_ROOT.resolve()}")
print(f"Target frames per video: {TARGET_FRAMES}")
# --- Cell 3 ---
import urllib.request

POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
FACE_LANDMARKS = 468
HOLISTIC_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task"
FRAME_SIZE = (640, 480)

def landmark_array(landmarks, count):
    if landmarks is None or len(landmarks) == 0:
        return np.zeros((count, 3), dtype=np.float32)
    points = landmarks if isinstance(landmarks, list) else landmarks.landmark
    coordinates = np.asarray([[point.x, point.y, point.z] for point in points], dtype=np.float32)
    if len(coordinates) != count:
        padded = np.zeros((count, 3), dtype=np.float32)
        padded[: min(count, len(coordinates))] = coordinates[:count]
        return padded
    return coordinates

def frame_features(results, modality):
    parts = [
        landmark_array(results.left_hand_landmarks, HAND_LANDMARKS),
        landmark_array(results.right_hand_landmarks, HAND_LANDMARKS),
    ]
    if modality in {"hand_face", "hand_face_pose"}:
        parts.append(landmark_array(results.face_landmarks, FACE_LANDMARKS))
    if modality == "hand_face_pose":
        parts.append(landmark_array(results.pose_landmarks, POSE_LANDMARKS))
    return np.concatenate(parts).reshape(-1)

def sample_frame_indices(frame_count, target_frames=TARGET_FRAMES):
    if frame_count <= 0:
        raise ValueError("Video contains no readable frames")
    return np.linspace(0, frame_count - 1, target_frames).round().astype(int)

class TasksHolisticAdapter:
    def __init__(self, landmarker, image_format):
        self.landmarker = landmarker
        self.image_format = image_format

    def process(self, rgb_frame):
        image = mp.Image(image_format=self.image_format, data=rgb_frame)
        return self.landmarker.detect(image)

    def close(self):
        try:
            self.landmarker.close()
        except RuntimeError:
            pass

def create_holistic(model_path):
    if hasattr(mp, "solutions"):
        return mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False,
        )

    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import HolisticLandmarker, HolisticLandmarkerOptions

    if not model_path.exists():
        print(f"Downloading MediaPipe Holistic model to: {model_path}")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HOLISTIC_MODEL_URL, model_path)
    options = HolisticLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model_path)))
    return TasksHolisticAdapter(HolisticLandmarker.create_from_options(options), mp.ImageFormat.SRGB)

def extract_video(video_path, holistic, target_frames=TARGET_FRAMES, modality="hands"):
    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_frame_indices(frame_count, target_frames)
    features = []
    next_index = 0

    for frame_index in range(frame_count):
        success, frame = capture.read()
        if not success:
            break
        if frame_index != indices[next_index]:
            continue
        frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)
        features.append(frame_features(results, modality))
        if next_index == len(indices) - 1:
            break
        next_index += 1

    capture.release()
    if not features:
        raise ValueError("No readable sampled frames")
    if len(features) < target_frames:
        features.extend([features[-1]] * (target_frames - len(features)))
    if len(features) > target_frames:
        features = features[:target_frames]
    return np.asarray(features, dtype=np.float32)

# --- Cell 4 ---
def extract_dataset(input_root=INPUT_ROOT, output_root=OUTPUT_ROOT, modality="hands", limit=None, target_frames=TARGET_FRAMES):
    videos = sorted(input_root.glob("*/**/*.mp4"))
    if limit is not None:
        videos = videos[:limit]
    if not videos:
        raise FileNotFoundError(f"No .mp4 files found below {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    holistic = create_holistic(output_root / "holistic_landmarker.task")
    failures = []
    
    try:
        for video_path in tqdm(videos, desc=f"Extracting {modality}"):
            relative_path = video_path.relative_to(input_root).with_suffix(".npz")
            output_path = output_root / modality / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists():
                continue
            try:
                keypoints = extract_video(video_path, holistic, target_frames, modality)
                np.savez_compressed(
                    output_path,
                    keypoints=keypoints,
                    label=video_path.parent.name,
                    video_id=video_path.stem,
                    split=video_path.parts[-3],
                )
            except (OSError, ValueError, cv2.error) as error:
                failures.append(f"{video_path}: {error}")
    finally:
        holistic.close()

    print(f"Processed: {len(videos) - len(failures)} / {len(videos)}")
    if failures:
        print("Failures:")
        print("\n".join(failures))


# Start with five videos locally. Set to True only after dependencies are installed.
RUN_EXTRACTION = False
if RUN_EXTRACTION:
    extract_dataset(limit=5, modality="hands")

# --- Cell 6 ---
def _coordinates(sequence):
    if sequence.ndim != 2 or sequence.shape[1] % 3:
        raise ValueError("Expected a (frames, features) array with xyz triples")
    return sequence.reshape(sequence.shape[0], -1, 3)


def rotate(sequence, degrees, axis=1):
    radians = np.deg2rad(degrees)
    cosine, sine = np.cos(radians), np.sin(radians)
    rotation = np.eye(3, dtype=np.float32)
    other_axis = (axis + 1) % 3
    rotation[other_axis, other_axis] = cosine
    rotation[other_axis, axis] = -sine
    rotation[axis, other_axis] = sine
    return (_coordinates(sequence) @ rotation.T).reshape(sequence.shape).astype(np.float32)


def scale(sequence, factor):
    return (sequence * factor).astype(np.float32)


def jitter(sequence, standard_deviation=0.01, rng=None):
    generator = rng or np.random.default_rng()
    return (sequence + generator.normal(0.0, standard_deviation, size=sequence.shape)).astype(np.float32)


def temporal_resample(sequence, factor):
    if factor <= 0:
        raise ValueError("factor must be positive")
    frame_count = sequence.shape[0]
    source_positions = np.linspace(0, frame_count - 1, frame_count)
    center = (frame_count - 1) / 2
    positions = np.clip((source_positions - center) / factor + center, 0, frame_count - 1)
    output = np.empty_like(sequence, dtype=np.float32)
    for feature_index in range(sequence.shape[1]):
        output[:, feature_index] = np.interp(positions, source_positions, sequence[:, feature_index])
    return output


def augment(sequence, rng=None):
    generator = rng or np.random.default_rng()
    output = rotate(sequence, generator.uniform(-15.0, 15.0))
    output = scale(output, generator.uniform(0.8, 1.2))
    output = jitter(output, rng=generator)
    return temporal_resample(output, generator.uniform(0.8, 1.2))


def augmentation_count(sample_count, percentage):
    if percentage not in {25, 50, 100}:
        raise ValueError("percentage must be 25, 50, or 100")
    return int(np.ceil(sample_count * percentage / 100))
# --- Cell 8 ---
class BiGRUClassifier(nn.Module):
    def __init__(self, input_size, class_count, hidden_size=128, layers=2, dropout=0.3):
        super().__init__()
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, class_count),
        )

    def forward(self, sequence):
        encoded, _ = self.encoder(sequence)
        return self.classifier(encoded[:, -1])


# Shape-only check using the hands modality: (batch, frames, features) -> (batch, classes).
model = BiGRUClassifier(input_size=126, class_count=100)
example_batch = torch.zeros(2, TARGET_FRAMES, 126)
logits = model(example_batch)
print(f"Input shape: {tuple(example_batch.shape)}")
print(f"Output shape: {tuple(logits.shape)}")
# --- Cell 11 ---
def summarize_video_dataset(root=INPUT_ROOT):
    summary = {}
    for split in ("train", "val", "test"):
        paths = sorted((root / split).glob("**/*.mp4"))
        summary[split] = {
            "videos": len(paths),
            "classes": len({path.parent.name for path in paths}),
        }
    return summary


video_summary = summarize_video_dataset()
for split, values in video_summary.items():
    print(f"{split}: {values['videos']} videos, {values['classes']} classes")


def summarize_landmarks(root=OUTPUT_ROOT, modality="hands"):
    paths = sorted((root / modality).glob("**/*.npz"))
    if not paths:
        print(f"No extracted files found for modality={modality!r}")
        return None
    sample = np.load(paths[0], allow_pickle=False)
    keypoints = sample["keypoints"]
    print(f"Files: {len(paths)}")
    print(f"Sample: {paths[0]}")
    print(f"Keypoint shape: {keypoints.shape}")
    print(f"Feature width: {keypoints.shape[-1]}")
    return keypoints.shape


# Run after the five-video extraction has completed.
# summarize_landmarks(modality="hands")
# --- Cell 13 ---
def generate_augmented_set(sequences, labels, percentage, rng=None):
    if len(sequences) != len(labels):
        raise ValueError("sequences and labels must have the same length")
    if not sequences:
        return [], []
    extra_count = augmentation_count(len(sequences), percentage)
    generator = rng or np.random.default_rng()
    augmented_sequences = []
    augmented_labels = []
    for index in range(extra_count):
        source_index = index % len(sequences)
        augmented_sequences.append(augment(sequences[source_index], rng=generator))
        augmented_labels.append(labels[source_index])
    return augmented_sequences, augmented_labels


# Example after loading a collection of training arrays:
# augmented_x, augmented_y = generate_augmented_set(training_x, training_y, 25)
# print(len(augmented_x), augmented_x[0].shape)
# --- Cell 15 ---
from torch.utils.data import DataLoader, Dataset


class LandmarkDataset(Dataset):
    def __init__(self, paths, class_to_index, augmentation_percent=0, seed=42):
        self.class_to_index = class_to_index
        self.seed = seed
        self.records = [(path, False) for path in paths]
        if augmentation_percent:
            extra_count = augmentation_count(len(paths), augmentation_percent)
            self.records.extend((paths[index % len(paths)], True) for index in range(extra_count))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, should_augment = self.records[index]
        with np.load(path, allow_pickle=False) as data:
            sequence = data["keypoints"].astype(np.float32)
            label = str(data["label"])
        if should_augment:
            sequence = augment(sequence, rng=np.random.default_rng(self.seed + index))
        return torch.from_numpy(sequence), self.class_to_index[label]


def make_dataloaders(modality="hands", augmentation_percent=0, batch_size=32, workers=0):
    modality_root = OUTPUT_ROOT / modality
    train_paths = sorted((modality_root / "train").glob("**/*.npz"))
    val_paths = sorted((modality_root / "val").glob("**/*.npz"))
    test_paths = sorted((modality_root / "test").glob("**/*.npz"))
    if not train_paths or not val_paths or not test_paths:
        raise FileNotFoundError("Extract train, val, and test landmark files before creating loaders")

    classes = sorted({path.parent.name for path in train_paths})
    class_to_index = {name: index for index, name in enumerate(classes)}
    datasets = {
        "train": LandmarkDataset(train_paths, class_to_index, augmentation_percent),
        "val": LandmarkDataset(val_paths, class_to_index),
        "test": LandmarkDataset(test_paths, class_to_index),
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split, dataset in datasets.items()
    }
    return loaders, class_to_index


# Example after full hands-only extraction:
# loaders, class_to_index = make_dataloaders("hands", augmentation_percent=0)
# print({split: len(loader.dataset) for split, loader in loaders.items()})
# --- Cell 17 ---
from sklearn.metrics import accuracy_score, f1_score
import time


def train_one_epoch(model, loader, optimizer, loss_function, device):
    model.train()
    total_loss = 0.0
    targets, predictions = [], []
    for sequences, labels in loader:
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(sequences)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        targets.extend(labels.detach().cpu().numpy())
        predictions.extend(logits.argmax(dim=1).detach().cpu().numpy())
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(targets, predictions),
        "f1": f1_score(targets, predictions, average="macro", zero_division=0),
    }


def evaluate(model, loader, device):
    model.eval()
    targets, predictions = [], []
    total_loss = 0.0
    loss_function = nn.CrossEntropyLoss()
    start = time.perf_counter()
    with torch.no_grad():
        for sequences, labels in loader:
            sequences, labels = sequences.to(device), labels.to(device)
            logits = model(sequences)
            total_loss += loss_function(logits, labels).item() * labels.size(0)
            targets.extend(labels.cpu().numpy())
            predictions.extend(logits.argmax(dim=1).cpu().numpy())
    elapsed = time.perf_counter() - start
    sample_count = len(loader.dataset)
    return {
        "loss": total_loss / sample_count,
        "accuracy": accuracy_score(targets, predictions),
        "f1": f1_score(targets, predictions, average="macro", zero_division=0),
        "latency_ms": elapsed * 1000 / sample_count,
    }


def train_model(loaders, class_to_index, epochs=10, hidden_size=128, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = loaders["train"].dataset[0][0].shape[-1]
    model = BiGRUClassifier(input_size, len(class_to_index), hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    history = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, loaders["train"], optimizer, loss_function, device)
        validation_metrics = evaluate(model, loaders["val"], device)
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in validation_metrics.items()}})
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"val accuracy {validation_metrics['accuracy']:.4f} | "
            f"val F1 {validation_metrics['f1']:.4f}"
        )

    test_metrics = evaluate(model, loaders["test"], device)
    return model, history, test_metrics
# --- Cell 19 ---
import itertools
import pandas as pd

EXPERIMENTS = list(itertools.product(
    ("hands", "hand_face", "hand_face_pose"),
    (0, 25, 50, 100),
))


def run_experiment_matrix(epochs=10, batch_size=32, hidden_size=128, workers=0):
    results = []
    for run_number, (modality, augmentation_percent) in enumerate(EXPERIMENTS, start=1):
        print(f"\nRun {run_number}/12: modality={modality}, augmentation={augmentation_percent}%")
        loaders, class_to_index = make_dataloaders(
            modality=modality,
            augmentation_percent=augmentation_percent,
            batch_size=batch_size,
            workers=workers,
        )
        _, history, test_metrics = train_model(
            loaders,
            class_to_index,
            epochs=epochs,
            hidden_size=hidden_size,
        )
        best_validation = max(history, key=lambda row: row["val_f1"])
        results.append({
            "run": run_number,
            "modality": modality,
            "augmentation_percent": augmentation_percent,
            "best_val_accuracy": best_validation["val_accuracy"],
            "best_val_f1": best_validation["val_f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_f1": test_metrics["f1"],
            "test_latency_ms": test_metrics["latency_ms"],
        })
    return pd.DataFrame(results)


RUN_EXPERIMENTS = False
if RUN_EXPERIMENTS:
    results_df = run_experiment_matrix(epochs=10)
    results_df.to_csv("experiment_results.csv", index=False)
    display(results_df)
else:
    print(f"Prepared {len(EXPERIMENTS)} experiments. Set RUN_EXPERIMENTS = True to run them.")
# --- Cell 21 ---
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

EXPERIMENT_CONFIG = {
    "frames": TARGET_FRAMES,
    "modalities": ["hands", "hand_face", "hand_face_pose"],
    "augmentation_percentages": [0, 25, 50, 100],
    "epochs": 10,
    "batch_size": 32,
    "seed": SEED,
}

print(EXPERIMENT_CONFIG)
# --- Cell 24 ---
RUN_SIMULATION = False

if RUN_SIMULATION:
    # Run from the project directory after the five-video hands extraction.
    %run simulate_research.py
else:
    print("Simulation is ready. Set RUN_SIMULATION = True to run it.")
