from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from tfacd.common.config import load_config
from tfacd.federated.common import model_from_metadata

parser = argparse.ArgumentParser(description="Binary external validation of the Edge-IIoTset-trained IDS on IoT-23 Zeek flows.")
parser.add_argument("--csv", required=True, help="Path to IoT-23 conn.log.labeled")
parser.add_argument("--config", default="configs/edge_iiot.yaml")
parser.add_argument("--checkpoint", default="artifacts/models/flower_ftil_final.pt")
parser.add_argument("--max-rows", type=int, default=200000)
parser.add_argument("--output", default="artifacts/external_validation/iot23_binary_report.json")
args = parser.parse_args()


def read_zeek(path: Path, max_rows: int | None) -> pd.DataFrame:
    fields = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#fields"):
                fields = line.rstrip("\n").split("\t")[1:]
                break
    if not fields:
        raise ValueError("Could not find #fields header in conn.log.labeled")
    return pd.read_csv(path, sep="\t", comment="#", names=fields, nrows=max_rows, low_memory=False)


def map_external(frame: pd.DataFrame, input_columns: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for column in input_columns:
        out[column] = np.nan

    mapping = {
        "id.orig_h": "ip.src_host",
        "id.resp_h": "ip.dst_host",
        "id.orig_p": "tcp.srcport",
        "id.resp_p": "tcp.dstport",
        "duration": "flow.duration",
        "orig_bytes": "orig_bytes",
        "resp_bytes": "resp_bytes",
        "orig_pkts": "orig_pkts",
        "resp_pkts": "resp_pkts",
        "orig_ip_bytes": "orig_ip_bytes",
        "resp_ip_bytes": "resp_ip_bytes",
        "service": "service",
        "proto": "proto",
        "conn_state": "conn_state",
        "history": "history",
    }
    for source, target in mapping.items():
        if source in frame and target in out:
            out[target] = frame[source]

    if "ts" in frame:
        if "frame.time" in out:
            out["frame.time"] = pd.to_datetime(frame["ts"], unit="s", errors="coerce")

    if "proto" in frame and "ip.proto" in out:
        proto = frame["proto"].astype(str).str.lower()
        out["ip.proto"] = proto.map({"tcp": 6, "udp": 17, "icmp": 1, "icmp6": 58})

    # Any categorical project feature that has no direct external equivalent is
    # left missing. The fitted OneHotEncoder handles unknown/missing values via
    # its training-time imputer. The report exposes feature coverage so this
    # transfer experiment cannot be mistaken for an in-domain test.
    return out[input_columns]


def load_model(config, checkpoint):
    metadata = json.loads((Path(config["data"]["output_dir"]) / "metadata.json").read_text(encoding="utf-8"))
    model = model_from_metadata(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model.load_state_dict(state)
    model.eval()
    return model, device, metadata


config = load_config(args.config)
frame = read_zeek(Path(args.csv), args.max_rows)
model, device, metadata = load_model(config, args.checkpoint)
preprocessor = __import__("joblib").load(Path(config["data"]["output_dir"]) / "preprocessor.joblib")
features = map_external(frame, list(metadata["input_columns"]))
encoded = preprocessor.transform(features).astype(np.float32)
x = torch.from_numpy(encoded[:, None, :]).to(device)

with torch.inference_mode():
    logits = model(x)
    pred_class = logits.argmax(dim=1).cpu().numpy()

if "label" not in frame.columns and "detailed-label" not in frame.columns:
    raise ValueError("IoT-23 file must contain 'label' or 'detailed-label' for binary benign/attack evaluation")
label_col = "label" if "label" in frame.columns else "detailed-label"
raw = frame[label_col].astype(str).str.lower()
y_true = (~raw.str.startswith("benign")).astype(int).to_numpy()
y_pred = (pred_class != metadata["classes"].index("Normal")).astype(int)

report = {
    "dataset": "IoT-23",
    "evaluation": "binary attack-vs-benign external validation",
    "source_file": str(Path(args.csv)),
    "rows_evaluated": int(len(frame)),
    "feature_dim": int(encoded.shape[1]),
    "input_feature_count": int(len(metadata["input_columns"])),
    "feature_coverage": float((features.notna().any(axis=0)).mean()),
    "label_column": label_col,
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
    "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    "false_positive_rate": float(((y_pred == 1) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1)),
    "false_negative_rate": float(((y_pred == 0) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1)),
    "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    "warning": "This is schema-adapted cross-dataset binary validation; missing external features are left missing and handled by the fitted Edge-IIoTset preprocessor. Do not present it as a matched-schema 15-class evaluation.",
}
path = Path(args.output)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(report, indent=2))
