from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float


def _metrics(loss: float, labels: list[int], predictions: list[int]) -> EpochMetrics:
    return EpochMetrics(
        loss=loss,
        accuracy=accuracy_score(labels, predictions),
        macro_precision=precision_score(labels, predictions, average="macro", zero_division=0),
        macro_recall=recall_score(labels, predictions, average="macro", zero_division=0),
        macro_f1=f1_score(labels, predictions, average="macro", zero_division=0),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    proximal_mu: float = 0.0,
    global_params: Sequence[torch.Tensor] | None = None,
) -> EpochMetrics:
    model.train()
    total_loss = 0.0
    labels: list[int] = []
    predictions: list[int] = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if proximal_mu > 0.0 and global_params is not None:
            # FedProx (https://arxiv.org/abs/1812.06127): penalize local drift from the
            # global round weights so heterogeneous clients don't diverge from each other.
            proximal_term = sum(
                (local_p - global_p.to(device)).norm(2) for local_p, global_p in zip(model.parameters(), global_params)
            )
            loss = loss + (proximal_mu / 2) * proximal_term
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss encountered")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        labels.extend(y.detach().cpu().tolist())
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
    return _metrics(total_loss / max(len(loader.dataset), 1), labels, predictions)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> EpochMetrics:
    model.eval()
    total_loss = 0.0
    labels: list[int] = []
    predictions: list[int] = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        labels.extend(y.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return _metrics(total_loss / max(len(loader.dataset), 1), labels, predictions)


@torch.inference_mode()
def predict_all(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int]]:
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        labels.extend(y.tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return labels, predictions


def class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = counts.sum() / np.maximum(counts * num_classes, 1.0)
    return torch.tensor(weights, dtype=torch.float32)
