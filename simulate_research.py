"""Run a tiny end-to-end rehearsal of the landmark research workflow."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from augmentations import augment
from model import BiGRUClassifier


class SimulationDataset(Dataset):
    def __init__(self, samples: list[tuple[np.ndarray, int]], augment_samples: int = 0) -> None:
        self.samples = samples
        self.augment_samples = augment_samples

    def __len__(self) -> int:
        return len(self.samples) + self.augment_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sequence, label = self.samples[index % len(self.samples)]
        if index >= len(self.samples):
            sequence = augment(sequence, rng=np.random.default_rng(index))
        return torch.from_numpy(sequence.astype(np.float32)), label


def make_simulation_data(paths: list[Path]) -> tuple[list[tuple[np.ndarray, int]], list[tuple[np.ndarray, int]], list[tuple[np.ndarray, int]]]:
    """Make a small controlled fixture from five real extracted sequences.

    This intentionally repeats perturbed examples across splits for a pipeline
    rehearsal only; it is not suitable for reporting research results.
    """
    templates = [np.load(path, allow_pickle=False)["keypoints"].astype(np.float32) for path in paths]
    splits = {"train": [], "val": [], "test": []}
    for label, template in enumerate(templates):
        for split in splits:
            for repeat in range(3):
                noise = np.random.default_rng(label * 10 + repeat).normal(0, 0.002, template.shape)
                splits[split].append(((template + noise).astype(np.float32), label))
    return splits["train"], splits["val"], splits["test"]


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    labels, predictions = [], []
    start = time.perf_counter()
    with torch.no_grad():
        for sequences, targets in loader:
            logits = model(sequences.to(device))
            labels.extend(targets.numpy())
            predictions.extend(logits.argmax(dim=1).cpu().numpy())
    elapsed_ms = (time.perf_counter() - start) * 1000 / len(loader.dataset)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "latency_ms": elapsed_ms,
    }


def main() -> None:
    paths = sorted(Path("WSLA100/processed_landmarks/hands").glob("**/*.npz"))
    if len(paths) < 5:
        raise FileNotFoundError("Run the five-video hands extraction before the simulation")
    train, validation, test = make_simulation_data(paths[:5])
    train_loader = DataLoader(SimulationDataset(train, augment_samples=len(train) // 4), batch_size=8, shuffle=True)
    validation_loader = DataLoader(SimulationDataset(validation), batch_size=8)
    test_loader = DataLoader(SimulationDataset(test), batch_size=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiGRUClassifier(input_size=126, class_count=5, hidden_size=32, layers=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(1, 3):
        model.train()
        for sequences, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(sequences.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, validation_loader, device)
        print(f"Epoch {epoch}/2 | val accuracy={metrics['accuracy']:.3f} | val F1={metrics['f1']:.3f}")

    test_metrics = evaluate(model, test_loader, device)
    results = pd.DataFrame([{
        "run": "simulation",
        "modality": "hands",
        "augmentation_percent": 25,
        "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_metrics["f1"],
        "test_latency_ms": test_metrics["latency_ms"],
        "device": str(device),
        "warning": "Pipeline rehearsal only; repeated perturbed templates across splits.",
    }])
    results.to_csv("simulation_results.csv", index=False)
    print(results.to_string(index=False))
    print("Saved: simulation_results.csv")


if __name__ == "__main__":
    main()