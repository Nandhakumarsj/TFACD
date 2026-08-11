from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class CNNBiLSTM(nn.Module):
    """CNN over per-flow features, BiLSTM over ordered flow windows."""

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        conv_channels: Sequence[int] = (64, 128),
        kernel_size: int = 3,
        pooled_features: int = 16,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or num_classes < 2:
            raise ValueError("feature_dim must be positive and num_classes >= 2")
        padding = kernel_size // 2
        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in conv_channels:
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True),
                ]
            )
            in_channels = out_channels
        layers.append(nn.AdaptiveAvgPool1d(pooled_features))
        self.feature_extractor = nn.Sequential(*layers)
        cnn_embedding = in_channels * pooled_features
        self.temporal = nn.LSTM(
            input_size=cnn_embedding,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        output_dim = lstm_hidden * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
            nn.Linear(output_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, sequence, features], got {tuple(x.shape)}")
        batch, steps, features = x.shape
        flow_tensor = x.reshape(batch * steps, 1, features)
        embeddings = self.feature_extractor(flow_tensor).flatten(start_dim=1)
        temporal_input = embeddings.reshape(batch, steps, -1)
        temporal_output, _ = self.temporal(temporal_input)
        final_state = temporal_output[:, -1, :]
        return self.classifier(final_state)
