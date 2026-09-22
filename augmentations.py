"""Augment fixed-length landmark sequences for WLASL experiments."""

from __future__ import annotations

import numpy as np


def _coordinates(sequence: np.ndarray) -> np.ndarray:
    if sequence.ndim != 2 or sequence.shape[1] % 3:
        raise ValueError("Expected a (frames, features) array with xyz triples")
    return sequence.reshape(sequence.shape[0], -1, 3)


def rotate(sequence: np.ndarray, degrees: float, axis: int = 1) -> np.ndarray:
    """Rotate every xyz landmark around the sequence's coordinate origin."""
    radians = np.deg2rad(degrees)
    cosine, sine = np.cos(radians), np.sin(radians)
    rotation = np.eye(3, dtype=np.float32)
    other_axis = (axis + 1) % 3
    rotation[other_axis, other_axis] = cosine
    rotation[other_axis, axis] = -sine
    rotation[axis, other_axis] = sine
    return (_coordinates(sequence) @ rotation.T).reshape(sequence.shape).astype(np.float32)


def scale(sequence: np.ndarray, factor: float) -> np.ndarray:
    return (sequence * factor).astype(np.float32)


def jitter(sequence: np.ndarray, standard_deviation: float = 0.01, rng: np.random.Generator | None = None) -> np.ndarray:
    generator = rng or np.random.default_rng()
    noise = generator.normal(0.0, standard_deviation, size=sequence.shape)
    return (sequence + noise).astype(np.float32)


def temporal_resample(sequence: np.ndarray, factor: float) -> np.ndarray:
    """Stretch or compress time while retaining the original frame count."""
    if factor <= 0:
        raise ValueError("factor must be positive")
    frame_count = sequence.shape[0]
    source_positions = np.linspace(0, frame_count - 1, frame_count)
    center = (frame_count - 1) / 2
    stretched_positions = np.clip((source_positions - center) / factor + center, 0, frame_count - 1)
    output = np.empty_like(sequence, dtype=np.float32)
    for feature_index in range(sequence.shape[1]):
        output[:, feature_index] = np.interp(stretched_positions, source_positions, sequence[:, feature_index])
    return output


def augment(sequence: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply one random spatial and temporal transformation combination."""
    generator = rng or np.random.default_rng()
    output = rotate(sequence, generator.uniform(-15.0, 15.0))
    output = scale(output, generator.uniform(0.8, 1.2))
    output = jitter(output, rng=generator)
    return temporal_resample(output, generator.uniform(0.8, 1.2))


def augmentation_count(sample_count: int, percentage: int) -> int:
    if percentage not in {25, 50, 100}:
        raise ValueError("percentage must be 25, 50, or 100")
    return int(np.ceil(sample_count * percentage / 100))