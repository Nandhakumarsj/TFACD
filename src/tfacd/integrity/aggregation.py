from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

State = Mapping[str, np.ndarray]


def weighted_average(states: Sequence[State], weights: Sequence[float]) -> dict[str, np.ndarray]:
    if not states or len(states) != len(weights):
        raise ValueError("states and weights must be non-empty and aligned")
    normalized = np.asarray(weights, dtype=np.float64)
    if np.any(normalized < 0) or normalized.sum() <= 0:
        raise ValueError("weights must be non-negative with positive sum")
    normalized /= normalized.sum()
    return {
        key: np.tensordot(normalized, np.stack([np.asarray(state[key]) for state in states]), axes=(0, 0)).astype(np.asarray(states[0][key]).dtype)
        for key in states[0].keys()
    }


def coordinate_median(states: Sequence[State]) -> dict[str, np.ndarray]:
    if not states:
        raise ValueError("states must be non-empty")
    return {
        key: np.median(np.stack([np.asarray(state[key]) for state in states]), axis=0).astype(np.asarray(states[0][key]).dtype)
        for key in states[0]
    }


def trimmed_mean(states: Sequence[State], trim_ratio: float) -> dict[str, np.ndarray]:
    if not 0 <= trim_ratio < 0.5:
        raise ValueError("trim_ratio must be in [0, 0.5)")
    if not states:
        raise ValueError("states must be non-empty")
    n = len(states)
    trim = int(np.floor(n * trim_ratio))
    result: dict[str, np.ndarray] = {}
    for key in states[0]:
        stacked = np.sort(np.stack([np.asarray(state[key]) for state in states]), axis=0)
        kept = stacked[trim : n - trim] if trim else stacked
        result[key] = kept.mean(axis=0).astype(np.asarray(states[0][key]).dtype)
    return result
