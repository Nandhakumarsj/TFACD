from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from tfacd.data.dataset import SequenceDataset
from tfacd.data.preprocess import load_prepared
from tfacd.data.sequences import make_sequences
from tfacd.models.cnn_bilstm import CNNBiLSTM
from tfacd.training.engine import class_weights, evaluate, predict_all, train_one_epoch


def build_model(config: dict[str, Any], feature_dim: int, num_classes: int) -> CNNBiLSTM:
    m = config["model"]
    return CNNBiLSTM(
        feature_dim=feature_dim,
        num_classes=num_classes,
        conv_channels=m["conv_channels"],
        kernel_size=m["kernel_size"],
        pooled_features=m["pooled_features"],
        lstm_hidden=m["lstm_hidden"],
        lstm_layers=m["lstm_layers"],
        bidirectional=m["bidirectional"],
        dropout=m["dropout"],
    )


def run_centralized(config: dict[str, Any]) -> Path:
    prepared = load_prepared(config["data"]["output_dir"])
    seq_len = int(config["data"].get("sequence_length", 1))
    stride = int(config["data"].get("sequence_stride", 1))
    x_train, y_train = make_sequences(prepared.x_train, prepared.y_train, seq_len, stride)
    x_val, y_val = make_sequences(prepared.x_val, prepared.y_val, seq_len, stride)
    x_test, y_test = make_sequences(prepared.x_test, prepared.y_test, seq_len, stride)

    cfg = config["training"]
    batch_size = int(cfg["batch_size"])
    train_loader = DataLoader(SequenceDataset(x_train, y_train), batch_size=batch_size, shuffle=True, num_workers=int(cfg["num_workers"]))
    val_loader = DataLoader(SequenceDataset(x_val, y_val), batch_size=batch_size, shuffle=False, num_workers=int(cfg["num_workers"]))
    test_loader = DataLoader(SequenceDataset(x_test, y_test), batch_size=batch_size, shuffle=False, num_workers=int(cfg["num_workers"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, prepared.feature_dim, prepared.num_classes).to(device)
    weights = class_weights(y_train, prepared.num_classes).to(device) if cfg.get("class_weighting", True) else None
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))

    best_f1 = -1.0
    patience = int(cfg["patience"])
    stale = 0
    output = Path("artifacts/models")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "centralized_best.pt"
    history = []

    for epoch in range(1, int(cfg["epochs"]) + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train": asdict(train_metrics), "validation": asdict(val_metrics)})
        print(f"epoch={epoch} train_f1={train_metrics.macro_f1:.4f} val_f1={val_metrics.macro_f1:.4f}")
        if val_metrics.macro_f1 > best_f1:
            best_f1 = val_metrics.macro_f1
            stale = 0
            torch.save({"state_dict": model.state_dict(), "feature_dim": prepared.feature_dim, "num_classes": prepared.num_classes}, checkpoint)
        else:
            stale += 1
            if stale >= patience:
                break

    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(payload["state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)

    metadata = json.loads((Path(config["data"]["output_dir"]) / "metadata.json").read_text(encoding="utf-8"))
    class_names = [str(c) for c in metadata["classes"]]
    test_labels, test_predictions = predict_all(model, test_loader, device)
    present = sorted(set(test_labels) | set(test_predictions))
    report = classification_report(
        test_labels,
        test_predictions,
        labels=present,
        target_names=[class_names[i] for i in present],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(test_labels, test_predictions, labels=present).tolist()

    (output / "centralized_metrics.json").write_text(
        json.dumps(
            {
                "history": history,
                "test": asdict(test_metrics),
                "test_per_class_report": report,
                "test_confusion_matrix": {"labels": [class_names[i] for i in present], "matrix": matrix},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"test_macro_f1={test_metrics.macro_f1:.4f}")
    worst = sorted(
        ((name, metrics["recall"]) for name, metrics in report.items() if isinstance(metrics, dict) and "recall" in metrics),
        key=lambda item: item[1],
    )[:5]
    print("lowest recall classes:", worst)
    return checkpoint
