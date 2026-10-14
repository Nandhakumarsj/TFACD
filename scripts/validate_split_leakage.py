from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from tfacd.common.config import load_config
from tfacd.data.data_inspector import ATTACK_TYPE_CANDIDATES, LABEL_CANDIDATES, TIME_CANDIDATES, _find_candidate
from tfacd.data.preprocess import _feature_columns, _resolve_target_columns, _split_temporal_ids_by_session
from tfacd.data.temporal import build_sequence_index

parser = argparse.ArgumentParser(description="Audit train/validation/test splitting for flow/session leakage and chronology.")
parser.add_argument("--config", default="configs/edge_iiot.yaml")
parser.add_argument("--output", default="artifacts/data/leakage_audit.json")
args = parser.parse_args()

config = load_config(args.config)
cfg = config["data"]
frame = pd.read_csv(cfg["raw_csv"], nrows=cfg.get("max_rows"), low_memory=False)
label_col, attack_type, target_col = _resolve_target_columns(frame, cfg)
timestamp_column = cfg.get("temporal", {}).get("timestamp_column", cfg.get("timestamp_column", "auto"))
if timestamp_column == "auto":
    timestamp_column = _find_candidate(list(frame.columns), TIME_CANDIDATES)
if not timestamp_column:
    raise SystemExit("No timestamp column found; a session-temporal leakage audit cannot be performed.")

seq_meta, windows = build_sequence_index(
    frame,
    sequence_length=1,
    stride=1,
    timestamp_column=timestamp_column,
    group_columns=cfg.get("temporal", {}).get("group_columns"),
    inactivity_seconds=cfg.get("temporal", {}).get("inactivity_seconds", 30.0),
    label_column=target_col,
)
train_ids, val_ids, test_ids = _split_temporal_ids_by_session(
    seq_meta,
    float(cfg.get("test_size", 0.15)),
    float(cfg.get("validation_size", 0.15)),
    seed=int(config.get("seed", 42)),
    chronological=True,
)
train_sessions = set(seq_meta.iloc[train_ids]["session_id"])
val_sessions = set(seq_meta.iloc[val_ids]["session_id"])
test_sessions = set(seq_meta.iloc[test_ids]["session_id"])

session_overlap = {
    "train_val": sorted(train_sessions & val_sessions),
    "train_test": sorted(train_sessions & test_sessions),
    "val_test": sorted(val_sessions & test_sessions),
}
chronology = {
    "train_end": seq_meta.iloc[train_ids]["end_time"].max() if len(train_ids) else None,
    "val_start": seq_meta.iloc[val_ids]["start_time"].min() if len(val_ids) else None,
    "val_end": seq_meta.iloc[val_ids]["end_time"].max() if len(val_ids) else None,
    "test_start": seq_meta.iloc[test_ids]["start_time"].min() if len(test_ids) else None,
}
chronology_ok = (
    chronology["train_end"] is not None
    and chronology["val_start"] is not None
    and chronology["val_end"] is not None
    and chronology["test_start"] is not None
    and chronology["train_end"] <= chronology["val_start"]
    and chronology["val_end"] <= chronology["test_start"]
)

report = {
    "split_strategy": "session_temporal",
    "timestamp_column": timestamp_column,
    "inactivity_seconds": cfg.get("temporal", {}).get("inactivity_seconds", 30.0),
    "sequences_total": int(len(seq_meta)),
    "rows_train": int(len(train_ids)),
    "rows_validation": int(len(val_ids)),
    "rows_test": int(len(test_ids)),
    "sessions_train": int(len(train_sessions)),
    "sessions_validation": int(len(val_sessions)),
    "sessions_test": int(len(test_sessions)),
    "session_overlap": session_overlap,
    "chronology": chronology,
    "chronology_ok": bool(chronology_ok),
    "leakage_free": not any(session_overlap.values()) and chronology_ok,
}
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(report, indent=2))
if not report["leakage_free"]:
    raise SystemExit(2)
