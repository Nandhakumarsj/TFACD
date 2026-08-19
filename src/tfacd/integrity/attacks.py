from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def sign_flip(state: Mapping[str, np.ndarray], scale: float = 1.0) -> dict[str, np.ndarray]:
    return {key: (-scale * np.asarray(value)).astype(np.asarray(value).dtype) for key, value in state.items()}


def gaussian_noise(state: Mapping[str, np.ndarray], sigma: float, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        key: (np.asarray(value) + rng.normal(0.0, sigma, size=np.asarray(value).shape)).astype(np.asarray(value).dtype)
        for key, value in state.items()
    }


def gradual_scaling(
    candidate: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    round_index: int,
    growth_rate: float = 1.0,
) -> dict[str, np.ndarray]:
    """Scale a client's update delta by a factor that grows every round.

    Simulates a client that starts near-benign (evading a first-round threshold)
    and escalates its deviation from the global model over time. `round_index` is
    0-based; the scale factor is `1 + growth_rate * round_index`.
    """
    scale = 1.0 + growth_rate * round_index
    return {
        key: (np.asarray(reference[key]) + scale * (np.asarray(candidate[key]) - np.asarray(reference[key]))).astype(
            np.asarray(candidate[key]).dtype
        )
        for key in reference
    }


def label_flip_to_normal(y: np.ndarray, normal_index: int, fraction: float = 1.0, seed: int = 0) -> np.ndarray:
    """Relabel a fraction of non-Normal samples as Normal.

    Simulates a poisoning client trying to teach the global model to ignore
    real attacks, the textbook label-flipping goal for an IDS.
    """
    rng = np.random.default_rng(seed)
    y_poisoned = y.copy()
    attack_indices = np.flatnonzero(y != normal_index)
    flip_count = int(round(len(attack_indices) * fraction))
    flip_indices = rng.choice(attack_indices, size=flip_count, replace=False) if flip_count else np.array([], dtype=int)
    y_poisoned[flip_indices] = normal_index
    return y_poisoned
