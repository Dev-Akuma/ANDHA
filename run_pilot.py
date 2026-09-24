import json
import time
import random
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

TARGET_FRAMES = 30
OUTPUT_ROOT = Path("WSLA100/processed_landmarks")
MODALITIES = ["hands", "hand_face", "hand_face_pose"]
AUGMENTATIONS = [0, 50, 100]

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
    return int(np.ceil(sample_count * percentage / 100))

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
        raise FileNotFoundError(f"Missing {modality} NPZ files")

    classes = sorted({path.parent.name for path in train_paths})
    class_to_index = {name: index for index, name in enumerate(classes)}
    datasets = {
        "train": LandmarkDataset(train_paths, class_to_index, augmentation_percent, seed=SEED),
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

def evaluate(model, loader, device, get_cm=False):
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
    metrics = {
        "loss": total_loss / sample_count,
        "accuracy": accuracy_score(targets, predictions),
        "f1": f1_score(targets, predictions, average="macro", zero_division=0),
        "precision": precision_score(targets, predictions, average="macro", zero_division=0),
        "recall": recall_score(targets, predictions, average="macro", zero_division=0),
        "latency_ms": elapsed * 1000 / sample_count,
    }
    if get_cm:
        metrics["confusion_matrix"] = confusion_matrix(targets, predictions).tolist()
    return metrics

def train_model(loaders, class_to_index, epochs=10, hidden_size=128, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = loaders["train"].dataset[0][0].shape[-1]
    model = BiGRUClassifier(input_size, len(class_to_index), hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    history = []
    
    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, loaders["train"], optimizer, loss_function, device)
        validation_metrics = evaluate(model, loaders["val"], device)
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in validation_metrics.items()}})
    training_time_s = time.perf_counter() - start_time

    test_metrics = evaluate(model, loaders["test"], device, get_cm=True)
    test_metrics["training_time_s"] = training_time_s
    return model, history, test_metrics

def main():
    EXPERIMENTS = list(itertools.product(MODALITIES, AUGMENTATIONS))
    
    # 1. Print subsets
    print("--- Pilot Configuration & Subsets ---")
    summary = {}
    total_videos = 0
    modality = "hands" # use hands to count
    for split in ["train", "val", "test"]:
        paths = list((OUTPUT_ROOT / modality / split).glob("**/*.npz"))
        classes = len({p.parent.name for p in paths})
        print(f"[{modality}] {split}: {len(paths)} videos, {classes} classes")
    
    print("\n--- Starting Experiments ---")
    results = []
    cms = {}
    
    for run_number, (modality, augmentation_percent) in enumerate(EXPERIMENTS, start=1):
        print(f"\nRun {run_number}/{len(EXPERIMENTS)}: modality={modality}, augmentation={augmentation_percent}%")
        loaders, class_to_index = make_dataloaders(modality=modality, augmentation_percent=augmentation_percent, batch_size=32)
        print(f"Train dataset size (including augmentation): {len(loaders['train'].dataset)}")
        
        _, history, test_metrics = train_model(loaders, class_to_index, epochs=15)
        
        best_validation = max(history, key=lambda row: row["val_f1"])
        
        res_row = {
            "run": run_number,
            "modality": modality,
            "augmentation_percent": augmentation_percent,
            "best_val_accuracy": best_validation["val_accuracy"],
            "best_val_f1": best_validation["val_f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_latency_ms": test_metrics["latency_ms"],
            "training_time_s": test_metrics["training_time_s"],
        }
        results.append(res_row)
        cms[f"run_{run_number}"] = test_metrics.pop("confusion_matrix")
        
        print(f"-> Test F1: {res_row['test_f1']:.4f}, Train Time: {res_row['training_time_s']:.1f}s")
        
    df = pd.DataFrame(results)
    df.to_csv("pilot_results.csv", index=False)
    with open("pilot_confusion_matrices.json", "w") as f:
        json.dump(cms, f)
    
    print("\n--- Final Results ---")
    print(df.to_string())

if __name__ == "__main__":
    main()
