import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


def test_numeric_features_are_coerced_before_transformer(tmp_path):
    csv_path = tmp_path / "numeric_features.csv"
    pd.DataFrame(
        {
            "duration": [0.5, 1.1, 2.2, 3.3, 0.9, 1.4],
            "pkt_size": [64, 128, 96, 55, 77, 90],
            "sensor_type": ["temp", "temp", "flow", "flow", "temp", "flow"],
            "Attack_label": [0, 1, 0, 1, 0, 1],
        }
    ).to_csv(csv_path, index=False)
    config = _write_config(tmp_path, csv_path)
    config["data"]["label_column"] = "Attack_label"
    config["data"]["attack_type_column"] = "auto"

    result = preprocess(config)
    metadata = json.loads((Path(config["data"]["output_dir"]) / "metadata.json").read_text())

    assert metadata["numeric_columns"] == ["duration", "pkt_size"]
    assert "sensor_type" in metadata["categorical_columns"]
    assert result.x_train.dtype.kind == "f"


def test_mixed_type_categorical_columns_are_normalized(tmp_path):
    """Mixed string/float object columns must be made uniform before
    OneHotEncoder sees them. This reproduces the sklearn failure:
    ``TypeError: Encoders require their input argument must be uniformly
    strings or numbers. Got ['float', 'str']``.
    """
    csv_path = tmp_path / "mixed_categorical.csv"
    pd.DataFrame(
        {
            "sensor_label": ["temp", 1.0, "flow", np.nan, "pressure", 2.0],
            "duration": [0.5, 1.1, 2.2, 3.3, 0.9, 1.4],
            "Attack_label": [0, 1, 0, 1, 0, 1],
        }
    ).to_csv(csv_path, index=False)
    config = _write_config(tmp_path, csv_path)
    config["data"]["label_column"] = "Attack_label"

    result = preprocess(config)

    assert result.x_train.ndim == 2
    assert result.x_train.dtype.kind == "f"


def test_temporal_splits_are_session_safe(tmp_path):
    csv_path = tmp_path / "session_safe.csv"
    rows = []
    for session in range(4):
        for i in range(6):
            rows.append(
                {
                    # Two-minute gaps force the sessionizer to create four
                    # distinct sessions with the default 30-second inactivity
                    # threshold.
                    "frame.time": f"2024-01-01 00:{session * 2:02d}:00.{i:02d}",
                    "ip.src_host": "10.0.0.1",
                    "ip.dst_host": "10.0.0.2",
                    # Keep one stable 5-tuple per synthetic session so each
                    # intended session remains one flow/session.
                    "tcp.srcport": 1000 + session,
                    "tcp.dstport": 80,
                    "ip.proto": 6,
                    "pkt_size": float(i + session),
                    "Attack_type": "Normal" if session % 2 == 0 else "DDoS_TCP",
                }
            )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    config = _write_config(tmp_path, csv_path, sequence_length=3)

    preprocess_module = __import__(
        "tfacd.data.preprocess",
        fromlist=["_split_temporal_ids_by_session", "build_sequence_index"],
    )
    seq_meta, _ = preprocess_module.build_sequence_index(
        pd.read_csv(csv_path),
        sequence_length=3,
        stride=1,
        timestamp_column="frame.time",
        group_columns=None,
        inactivity_seconds=30.0,
        label_column="Attack_type",
    )
    train_ids, val_ids, test_ids = preprocess_module._split_temporal_ids_by_session(
        seq_meta,
         test_size=0.25,
         val_size=0.25,
         seed=42,
     )

    assert set(train_ids).isdisjoint(set(val_ids))
    assert set(train_ids).isdisjoint(set(test_ids))
    assert set(val_ids).isdisjoint(set(test_ids))
    assert seq_meta.iloc[train_ids]["session_id"].nunique() == 2
    assert seq_meta.iloc[val_ids]["session_id"].nunique() == 1
    assert seq_meta.iloc[test_ids]["session_id"].nunique() == 1


def test_temporal_fallback_raises_instead_of_leaking(tmp_path):
    """REVIEW FIX (P0): when there are too few sessions to give every split at
    least one session, preprocess() must fail loudly rather than silently
    falling back to a random sequence-level split (which can leak overlapping
    windows from the same session across train/test)."""
    csv_path = tmp_path / "single_session.csv"
    rows = []
    for i in range(6):
        rows.append(
            {
                "frame.time": f"2024-01-01 00:00:00.{i:02d}",
                "ip.src_host": "10.0.0.1",
                "ip.dst_host": "10.0.0.2",
                "Attack_label": 0,
                # Stable flow key: all six rows belong to one session.
                "tcp.srcport": 1000,
                "tcp.dstport": 80,
                "ip.proto": 6,
                "pkt_size": float(i),
                "Attack_type": "Normal",
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    config = _write_config(tmp_path, csv_path, sequence_length=3)

    with pytest.raises(ValueError, match="Session-safe temporal split failed"):
        preprocess(config)


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
