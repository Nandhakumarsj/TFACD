from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_samples: int = 1,
    max_attempts: int = 100,
    min_class_samples: int = 0,
) -> list[np.ndarray]:
    """`min_samples` only ever checked each client's TOTAL sample count - a rare
    class could be reduced to exactly 0 examples on some client and still pass,
    since the guard's per-client total stayed well above the threshold from
    other classes alone (verified empirically). `min_class_samples` is a second,
    opt-in guard (default 0 = off, preserving prior behavior) requiring every
    client to see at least that many examples of EVERY class. Deliberately not
    the default: Dirichlet non-IID partitioning exists specifically to simulate
    label skew, so a strong "every client sees every class well" guarantee
    would contradict the method's own purpose at low alpha - this only closes
    the "a class is completely invisible to a client" failure mode.
    """
    if num_clients < 2:
        raise ValueError("num_clients must be >= 2")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)

    for _ in range(max_attempts):
        client_indices: list[list[int]] = [[] for _ in range(num_clients)]
        client_class_counts: list[dict[Any, int]] = [{} for _ in range(num_clients)]
        for cls in classes:
            cls_idx = np.flatnonzero(labels == cls)
            rng.shuffle(cls_idx)
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            cuts = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
            for client_id, chunk in enumerate(np.split(cls_idx, cuts)):
                client_indices[client_id].extend(chunk.tolist())
                client_class_counts[client_id][cls] = len(chunk)
        result = []
        for values in client_indices:
            arr = np.asarray(values, dtype=np.int64)
            rng.shuffle(arr)
            result.append(arr)
        total_ok = min(map(len, result)) >= min_samples
        class_ok = min_class_samples <= 0 or all(
            min(counts.get(cls, 0) for counts in client_class_counts) >= min_class_samples for cls in classes
        )
        if total_ok and class_ok:
            return result
    reason = f"min_samples={min_samples}"
    if min_class_samples > 0:
        reason += f" and min_class_samples={min_class_samples}"
    raise RuntimeError(f"Could not create partitions satisfying {reason} after {max_attempts} attempts")


def iid_partition(labels: np.ndarray, num_clients: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(labels))
    return [part.astype(np.int64) for part in np.array_split(indices, num_clients)]


def save_partitions(partitions: list[np.ndarray], output_dir: str | Path, labels: np.ndarray) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for client_id, indices in enumerate(partitions):
        np.save(out / f"client_{client_id}.npy", indices)
        unique, counts = np.unique(labels[indices], return_counts=True)
        manifest[str(client_id)] = {
            "samples": int(len(indices)),
            "label_counts": {str(int(k)): int(v) for k, v in zip(unique, counts)},
        }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
