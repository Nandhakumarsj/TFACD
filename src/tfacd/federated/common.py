from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from tfacd.models.cnn_bilstm import CNNBiLSTM


def model_from_metadata(config: dict[str, Any]) -> CNNBiLSTM:
    metadata = json.loads((Path(config["data"]["output_dir"]) / "metadata.json").read_text(encoding="utf-8"))
    m = config["model"]
    return CNNBiLSTM(
        feature_dim=metadata["feature_dim"],
        num_classes=metadata["num_classes"],
        conv_channels=m["conv_channels"],
        kernel_size=m["kernel_size"],
        pooled_features=m["pooled_features"],
        lstm_hidden=m["lstm_hidden"],
        lstm_layers=m["lstm_layers"],
        bidirectional=m["bidirectional"],
        dropout=m["dropout"],
    )


def device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
