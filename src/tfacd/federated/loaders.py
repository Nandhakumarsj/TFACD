from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from tfacd.data.dataset import SequenceDataset
from tfacd.data.preprocess import load_prepared
from tfacd.data.sequences import make_sequences


def client_loaders(config: dict, client_id: int, batch_size: int):
    prepared = load_prepared(config["data"]["output_dir"])
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
