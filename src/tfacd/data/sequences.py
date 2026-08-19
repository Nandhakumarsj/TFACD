from __future__ import annotations

import numpy as np


def make_sequences(
    x: np.ndarray,
    y: np.ndarray,
    sequence_length: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create fixed windows from an already correctly ordered array.

    Do not call this on shuffled rows and then claim temporal learning.
    The target is the final row label in each window.
    """
    if sequence_length < 1 or stride < 1:
        raise ValueError("sequence_length and stride must be positive")
    if len(x) != len(y):
        raise ValueError("x and y length mismatch")
    if sequence_length == 1:
        return x[:, None, :].astype(np.float32), y.astype(np.int64)
    if len(x) < sequence_length:
        raise ValueError("Not enough rows for one sequence")
    starts = range(0, len(x) - sequence_length + 1, stride)
    seq_x = np.stack([x[start : start + sequence_length] for start in starts])
    seq_y = np.asarray([y[start + sequence_length - 1] for start in starts])
    return seq_x.astype(np.float32), seq_y.astype(np.int64)
