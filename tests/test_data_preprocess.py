import json
from pathlib import Path

import numpy as np
import pandas as pd

from tfacd.data.data_inspector import inspect_all, inspect_temporal
from tfacd.data.preprocess import heldout_indices, preprocess


def _flow_frame() -> pd.DataFrame:
    rows = []
    for flow in range(2):
        base = f"2021-01-01 00:00:{flow:02d}"
        for step in range(6):
            rows.append(
                {
                    "frame.time": f"{base}.{step * 10:06d}",
                    "ip.src_host": "A" if step % 2 == 0 else "B",
                    "ip.dst_host": "B" if step % 2 == 0 else "A",
                    "tcp.srcport": 1000 + flow,
                    "tcp.dstport": 80,
                    "ip.proto": 6,
                    "pkt_size": float(step + flow),
                    "Attack_label": 0 if flow == 0 else 1,
                    "Attack_type": "Normal" if flow == 0 else "DDoS_TCP",
                }
            )
    return pd.DataFrame(rows)


def _write_config(tmp_path: Path, csv_path: Path, sequence_length: int = 1) -> dict:
    return {
        "seed": 42,
        "data": {
            "raw_csv": str(csv_path),
            "output_dir": str(tmp_path / "artifacts"),
            "label_column": "auto",
            "attack_type_column": "auto",
            "timestamp_column": "auto",
            "test_size": 0.25,
            "validation_size": 0.25,
            "max_rows": None,
            "sequence_length": sequence_length,
            "sequence_stride": 1,
            "identifier_patterns": [
                "^frame\\.time$",
                "^ip\\.src_host$",
                "^ip\\.dst_host$",
            ],
            "temporal": {
                "timestamp_column": "frame.time",
                "inactivity_seconds": 30,
                "group_columns": None,
                "audit_sample_rows": None,
            },
        },
    }


def test_row_based_preprocess_shape(tmp_path):
    csv_path = tmp_path / "rows.csv"
    frame = _flow_frame()
    frame.to_csv(csv_path, index=False)
    config = _write_config(tmp_path, csv_path, sequence_length=1)

    result = preprocess(config)
    assert result.x_train.ndim == 2
    assert result.feature_dim == result.x_train.shape[1]

    metadata = json.loads((Path(config["data"]["output_dir"]) / "metadata.json").read_text())
    assert metadata["split_mode"] == "row"
    assert metadata["temporal_windows"] is False


def test_temporal_preprocess_builds_3d_windows(tmp_path):
    csv_path = tmp_path / "flows.csv"
    _flow_frame().to_csv(csv_path, index=False)
    config = _write_config(tmp_path, csv_path, sequence_length=2)

    result = preprocess(config)
    assert result.x_train.ndim == 3
    assert result.x_train.shape[1] == 2
    assert len(result.y_train) == result.x_train.shape[0]

    metadata = json.loads((Path(config["data"]["output_dir"]) / "metadata.json").read_text())
    assert metadata["split_mode"] == "temporal_sequence"
    assert metadata["temporal_windows"] is True
    assert metadata["sequence_length"] == 2


def test_inspect_temporal_on_flow_csv(tmp_path):
    csv_path = tmp_path / "flows.csv"
    _flow_frame().to_csv(csv_path, index=False)
    config = _write_config(tmp_path, csv_path)

    report = inspect_all(config)
    assert "temporal_audit" in report
    assert report["temporal_audit"]["available"] is True
    assert report["temporal_audit"]["flows_with_multiple_rows"] >= 1


def test_heldout_indices_temporal_uses_saved_rows(tmp_path):
    csv_path = tmp_path / "flows.csv"
    _flow_frame().to_csv(csv_path, index=False)
    config = _write_config(tmp_path, csv_path, sequence_length=2)
    preprocess(config)

    indices = heldout_indices(config)
    assert len(indices) >= 1
    assert (Path(config["data"]["output_dir"]) / "heldout_row_indices.npy").exists()
