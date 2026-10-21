from __future__ import annotations

from collections.abc import Mapping

import numpy as np

def sign_flip(
    client_state: Mapping[str, np.ndarray],
    global_state: Mapping[str, np.ndarray],
    scale: float = 1.0,
) -> dict[str, np.ndarray]:
    """Reverse a client's floating-point model updates around the global state.

    For each floating-point state entry:
        delta = client_state - global_state
        poisoned = global_state - scale * delta

    Non-floating-point buffers are preserved from the locally trained client
    state because integer counters are not meaningful signed model updates.

    Args:
        client_state: State dictionary after local client training.
        global_state: Global model state distributed at the start of the round.
        scale: Positive attack-strength multiplier.

    Returns:
        A new state dictionary containing the poisoned model state.
    """
    if scale < 0:
        raise ValueError("scale must be non-negative")

    if client_state.keys() != global_state.keys():
        raise ValueError(
            "client_state and global_state must have identical keys"
        )

    poisoned: dict[str, np.ndarray] = {}

    for key, global_value in global_state.items():
        client_value = np.asarray(client_state[key])
        global_value = np.asarray(global_value)

        if client_value.shape != global_value.shape:
            raise ValueError(
                f"Shape mismatch for {key!r}: "
                f"client={client_value.shape}, global={global_value.shape}"
            )

        if np.issubdtype(client_value.dtype, np.floating):
            delta = client_value - global_value

            poisoned[key] = (
                global_value - scale * delta
            ).astype(client_value.dtype, copy=False)
        else:
            # Preserve integer buffers instead of applying arithmetic to them.
            poisoned[key] = client_value.copy()

    return poisoned

def state_delta_l2_norm(
    candidate: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
) -> float:
    """L2 norm of floating-point state deltas; integer buffers are excluded."""
    if candidate.keys() != reference.keys():
        raise ValueError("candidate and reference must have identical keys")
    squared = 0.0
    for key, reference_value in reference.items():
        ref = np.asarray(reference_value)
        value = np.asarray(candidate[key])
        if value.shape != ref.shape:
            raise ValueError(f"Shape mismatch for {key!r}: candidate={value.shape}, reference={ref.shape}")
        if np.issubdtype(value.dtype, np.floating):
            delta = value.astype(np.float64, copy=False) - ref.astype(np.float64, copy=False)
            squared += float(np.dot(delta.ravel(), delta.ravel()))
    return float(np.sqrt(squared))


def gaussian_noise(
    client_state: Mapping[str, np.ndarray],
    global_state: Mapping[str, np.ndarray],
    sigma: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Perturb a client update with update-normalized Gaussian noise.

    Let delta = W_client - W_global. The scalar ``sigma`` is a dimensionless
    relative strength: epsilon is sampled elementwise from N(0, (sigma * RMS(delta))^2),
    where RMS(delta) = ||delta||_2 / sqrt(number of floating-point parameters).
    Thus the expected noise L2 norm is approximately ``sigma * ||delta||_2``
    across a large model, rather than applying an arbitrary 0.5 to absolute
    weights whose scales can differ by orders of magnitude.

    Integer/non-floating buffers are preserved from the client's trained state.
    Inputs are never mutated. A zero update produces zero perturbation, avoiding
    noise unrelated to a client's actual contribution.
    """
    if not np.isfinite(float(sigma)) or sigma < 0:
        raise ValueError("sigma must be finite and non-negative")
    if client_state.keys() != global_state.keys():
        raise ValueError("client_state and global_state must have identical keys")

    total_elements = 0
    update_norm = state_delta_l2_norm(client_state, global_state)
    for key, global_value in global_state.items():
        client_value = np.asarray(client_state[key])
        global_value = np.asarray(global_value)
        if client_value.shape != global_value.shape:
            raise ValueError(
                f"Shape mismatch for {key!r}: client={client_value.shape}, global={global_value.shape}"
            )
        if np.issubdtype(client_value.dtype, np.floating):
            total_elements += client_value.size

    noise_std = sigma * update_norm / np.sqrt(total_elements) if total_elements and update_norm else 0.0
    rng = np.random.default_rng(seed)
    poisoned: dict[str, np.ndarray] = {}
    for key, global_value in global_state.items():
        client_value = np.asarray(client_state[key])
        if np.issubdtype(client_value.dtype, np.floating):
            noise = rng.normal(0.0, noise_std, size=client_value.shape)
            poisoned[key] = (client_value.astype(np.float64, copy=False) + noise).astype(
                client_value.dtype, copy=False
            )
        else:
            poisoned[key] = client_value.copy()
    return poisoned


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
