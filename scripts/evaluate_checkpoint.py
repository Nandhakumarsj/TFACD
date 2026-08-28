from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tfacd.common.config import load_config
from tfacd.data.dataset import SequenceDataset
from tfacd.data.preprocess import load_prepared
from tfacd.data.sequences import as_model_input
from tfacd.federated.common import model_from_metadata
from tfacd.training.engine import evaluate, predict_all
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/edge_iiot.yaml")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--label", default=None, help="Name shown in the printed summary")
args = parser.parse_args()

config = load_config(args.config)
metadata = json.loads((Path(config["data"]["output_dir"]) / "metadata.json").read_text(encoding="utf-8"))
class_names = [str(c) for c in metadata["classes"]]

prepared = load_prepared(config["data"]["output_dir"])
seq_len = int(config["data"].get("sequence_length", 1))
stride = int(config["data"].get("sequence_stride", 1))
x_test, y_test = as_model_input(prepared.x_test, prepared.y_test, seq_len, stride)
test_loader = DataLoader(SequenceDataset(x_test, y_test), batch_size=int(config["training"]["batch_size"]), shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model_from_metadata(config).to(device)

payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
model.load_state_dict(state_dict)

criterion = torch.nn.CrossEntropyLoss()
metrics = evaluate(model, test_loader, criterion, device)
labels, predictions = predict_all(model, test_loader, device)
present = sorted(set(labels) | set(predictions))
report = classification_report(
    labels, predictions, labels=present, target_names=[class_names[i] for i in present],
    output_dict=True, zero_division=0,
)

name = args.label or Path(args.checkpoint).name
print(f"[{name}] test_macro_f1={metrics.macro_f1:.4f} test_accuracy={metrics.accuracy:.4f}")
worst = sorted(
    ((n, m["recall"]) for n, m in report.items() if isinstance(m, dict) and "recall" in m),
    key=lambda item: item[1],
)[:5]
print(f"[{name}] lowest recall classes:", worst)
