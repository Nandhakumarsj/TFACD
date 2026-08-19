from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from tfacd.data.dataset import SequenceDataset
from tfacd.data.preprocess import load_prepared
from tfacd.data.sequences import make_sequences


@lru_cache(maxsize=4)
def _load_prepared_cached(output_dir: str):
    # client_loaders() is called once per client per Flower train()/evaluate()
    # message - measured live: load_prepared() takes ~8-13s per call (fully
    # decompressing all six train/val/test arrays from prepared.npz, two-thirds
    # of which - val/test - this function never even reads), so an unguarded
    # call reloads the same ~100MB file dozens of times across one run. Fancy
    # indexing (x[indices]) below always returns a fresh copy, never a view, so
    # sharing this cached object across calls/clients is safe - nothing mutates
    # it in place. maxsize=4, not 1: tolerates a handful of distinct output_dirs
    # in one process without unbounded growth, though in practice there's one.
    return load_prepared(output_dir)


def client_loaders(config: dict, client_id: int, batch_size: int):
    prepared = _load_prepared_cached(config["data"]["output_dir"])
    indices = np.load(Path(config["data"]["output_dir"]) / "partitions" / f"client_{client_id}.npy")
    x = prepared.x_train[indices]
    y = prepared.y_train[indices]
    seq_len = int(config["data"].get("sequence_length", 1))
    stride = int(config["data"].get("sequence_stride", 1))
    x_seq, y_seq = make_sequences(x, y, seq_len, stride)
    split = max(1, int(0.8 * len(y_seq)))
    train = SequenceDataset(x_seq[:split], y_seq[:split])
    val = SequenceDataset(x_seq[split:], y_seq[split:]) if split < len(y_seq) else SequenceDataset(x_seq[-1:], y_seq[-1:])
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size, shuffle=False),
    )
