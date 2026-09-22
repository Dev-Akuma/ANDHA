"""Sequence classifier used by the WLASL baseline experiments."""

from __future__ import annotations

import torch
from torch import nn


class BiGRUClassifier(nn.Module):
    def __init__(self, input_size: int, class_count: int, hidden_size: int = 128, layers: int = 2, dropout: float = 0.3):
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

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(sequence)
        return self.classifier(encoded[:, -1])