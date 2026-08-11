from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_samples: int = 1,
    max_attempts: int = 100,
) -> list[np.ndarray]:
    if num_clients < 2:
        raise ValueError("num_clients must be >= 2")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)

    for _ in range(max_attempts):
        client_indices: list[list[int]] = [[] for _ in range(num_clients)]
        for cls in classes:
            cls_idx = np.flatnonzero(labels == cls)
            rng.shuffle(cls_idx)
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            cuts = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
            for client_id, chunk in enumerate(np.split(cls_idx, cuts)):
                client_indices[client_id].extend(chunk.tolist())
        result = []
        for values in client_indices:
            arr = np.asarray(values, dtype=np.int64)
            rng.shuffle(arr)
            result.append(arr)
        if min(map(len, result)) >= min_samples:
            return result
    raise RuntimeError("Could not create partitions satisfying min_samples")


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
