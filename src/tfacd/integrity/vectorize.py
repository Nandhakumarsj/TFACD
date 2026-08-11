from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def flatten_state(state: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(state[key], dtype=np.float32).ravel() for key in sorted(state)])


def flatten_delta(candidate: Mapping[str, np.ndarray], reference: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            (np.asarray(candidate[key], dtype=np.float32) - np.asarray(reference[key], dtype=np.float32)).ravel()
            for key in sorted(reference)
        ]
    )
