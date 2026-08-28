from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from tfacd.data.data_inspector import (
    ATTACK_TYPE_CANDIDATES,
    LABEL_CANDIDATES,
    TIME_CANDIDATES,
    _find_candidate,
    identifier_like_columns,
)
from tfacd.data.temporal import build_sequence_index


@dataclass
class PreparedData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    num_classes: int
    feature_dim: int


def _resolve_column(frame: pd.DataFrame, configured: str, candidates: list[str]) -> str:
    if configured and configured != "auto":
        if configured not in frame.columns:
            raise KeyError(f"Configured column '{configured}' not found")
        return configured
    found = _find_candidate(list(frame.columns), candidates)
    if found is None:
        raise KeyError(f"Could not auto-detect a column from candidates: {candidates}")
    return found


def _resolve_target_columns(frame: pd.DataFrame, cfg: dict[str, Any]) -> tuple[str, str | None, str]:
    label_col = _resolve_column(frame, cfg.get("label_column", "auto"), LABEL_CANDIDATES)
    attack_type = _find_candidate(list(frame.columns), ATTACK_TYPE_CANDIDATES)
    target_col = attack_type if attack_type and frame[attack_type].nunique() > 2 else label_col
    return label_col, attack_type, target_col


def _feature_columns(
    frame: pd.DataFrame,
    cfg: dict[str, Any],
    label_col: str,
    attack_type: str | None,
    target_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    explicit_drop = set(cfg.get("drop_columns", []))
    explicit_drop.update({label_col, target_col})
    if attack_type:
        explicit_drop.add(attack_type)

    x = frame.drop(columns=[c for c in explicit_drop if c in frame.columns]).copy()

    leakage_tokens = {"attack_label", "attack_type", "attack", "label", "class", "target"}
    auto_drop = [
        column
        for column in x.columns
        if column.lower().strip().replace(" ", "_") in leakage_tokens
    ]
    auto_drop += identifier_like_columns(list(x.columns), cfg.get("identifier_patterns", []))
    x = x.drop(columns=sorted(set(auto_drop)), errors="ignore")
    return x.replace([np.inf, -np.inf], np.nan), sorted(set(auto_drop))


def _build_transformer(x_train: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric = list(x_train.select_dtypes(include=["number", "bool"]).columns)
    categorical = [column for column in x_train.columns if column not in numeric]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=64)),
        ]
    )
    transformer = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return transformer, numeric, categorical


def _save_prepared(
    out: Path,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    transformer: ColumnTransformer,
    encoder: LabelEncoder,
    metadata: dict[str, Any],
    heldout_row_indices: np.ndarray | None = None,
) -> PreparedData:
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "prepared.npz",
        x_train=x_train,
        y_train=y_train.astype(np.int64),
        x_val=x_val,
        y_val=y_val.astype(np.int64),
        x_test=x_test,
        y_test=y_test.astype(np.int64),
    )
    joblib.dump(transformer, out / "preprocessor.joblib")
    joblib.dump(encoder, out / "label_encoder.joblib")
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if heldout_row_indices is not None:
        np.save(out / "heldout_row_indices.npy", heldout_row_indices.astype(np.int64))

    feature_dim = int(x_train.shape[-1])
    return PreparedData(
        x_train,
        y_train.astype(np.int64),
        x_val,
        y_val.astype(np.int64),
        x_test,
        y_test.astype(np.int64),
        metadata["num_classes"],
        feature_dim,
    )


def _sequences_to_tensor(
    transformer: ColumnTransformer,
    features: pd.DataFrame,
    window_indices: np.ndarray,
    sequence_ids: np.ndarray,
) -> np.ndarray:
    """Materialize flow-based windows as [N, seq_len, feature_dim]."""
    selected = window_indices[sequence_ids]
    unique_rows, inverse = np.unique(selected.reshape(-1), return_inverse=True)
    encoded = transformer.transform(features.iloc[unique_rows]).astype(np.float32)
    seq_len = selected.shape[1]
    return encoded[inverse].reshape(len(sequence_ids), seq_len, -1)


def _preprocess_row_based(config: dict[str, Any], frame: pd.DataFrame) -> PreparedData:
    cfg = config["data"]
    label_col, attack_type, target_col = _resolve_target_columns(frame, cfg)
    x, auto_drop = _feature_columns(frame, cfg, label_col, attack_type, target_col)
    y_raw = frame[target_col].astype(str).fillna("UNKNOWN")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    seed = int(config.get("seed", 42))
    test_size = float(cfg.get("test_size", 0.15))
    val_size = float(cfg.get("validation_size", 0.15))

    idx = np.arange(len(y))
    idx_train_val, idx_test, y_train_val, y_test = train_test_split(
        idx, y, test_size=test_size, random_state=seed, stratify=y
    )
    relative_val = val_size / (1.0 - test_size)
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_train_val,
        y_train_val,
        test_size=relative_val,
        random_state=seed,
        stratify=y_train_val,
    )
    x_train, x_val, x_test = x.iloc[idx_train], x.iloc[idx_val], x.iloc[idx_test]

    transformer, numeric, categorical = _build_transformer(x_train)
    x_train_arr = transformer.fit_transform(x_train).astype(np.float32)
    x_val_arr = transformer.transform(x_val).astype(np.float32)
    x_test_arr = transformer.transform(x_test).astype(np.float32)

    auto_drop = sorted(set(auto_drop))
    metadata = {
        "split_mode": "row",
        "target_column": target_col,
        "input_columns": list(x.columns),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "feature_dim": int(x_train_arr.shape[1]),
        "num_classes": int(len(encoder.classes_)),
        "classes": encoder.classes_.tolist(),
        "auto_dropped_leakage": auto_drop,
        "sequence_length": 1,
        "temporal_windows": False,
    }
    return _save_prepared(
        Path(cfg["output_dir"]),
        x_train_arr,
        y_train,
        x_val_arr,
        y_val,
        x_test_arr,
        y_test,
        transformer,
        encoder,
        metadata,
        heldout_row_indices=idx_test,
    )


def _preprocess_temporal(config: dict[str, Any], frame: pd.DataFrame) -> PreparedData:
    cfg = config["data"]
    temporal_cfg = cfg.get("temporal", {})
    seq_len = int(cfg.get("sequence_length", 1))
    stride = int(cfg.get("sequence_stride", 1))

    label_col, attack_type, target_col = _resolve_target_columns(frame, cfg)
    timestamp_column = temporal_cfg.get("timestamp_column", cfg.get("timestamp_column", "auto"))
    if timestamp_column == "auto":
        timestamp_column = _find_candidate(list(frame.columns), TIME_CANDIDATES)
    if not timestamp_column:
        raise ValueError(
            "sequence_length > 1 requires a timestamp column for flow/session windowing"
        )

    seq_meta, window_indices = build_sequence_index(
        frame,
        sequence_length=seq_len,
        stride=stride,
        timestamp_column=timestamp_column,
        group_columns=temporal_cfg.get("group_columns"),
        inactivity_seconds=temporal_cfg.get("inactivity_seconds", 30.0),
        label_column=target_col,
    )
    if len(seq_meta) == 0:
        raise ValueError(
            "No flow/session windows were produced. Lower sequence_length or inspect the dataset."
        )

    encoder = LabelEncoder()
    y = encoder.fit_transform(seq_meta["label"].astype(str))

    seed = int(config.get("seed", 42))
    test_size = float(cfg.get("test_size", 0.15))
    val_size = float(cfg.get("validation_size", 0.15))
    seq_ids = np.arange(len(seq_meta))

    trainval_ids, test_ids, y_trainval, y_test = train_test_split(
        seq_ids, y, test_size=test_size, random_state=seed, stratify=y
    )
    relative_val = val_size / (1.0 - test_size)
    train_ids, val_ids, _, _ = train_test_split(
        trainval_ids,
        y_trainval,
        test_size=relative_val,
        random_state=seed,
        stratify=y_trainval,
    )

    features, _ = _feature_columns(frame, cfg, label_col, attack_type, target_col)
    train_rows = np.unique(window_indices[train_ids].reshape(-1))
    transformer, numeric, categorical = _build_transformer(features.iloc[train_rows])
    transformer.fit(features.iloc[train_rows])

    x_train_arr = _sequences_to_tensor(transformer, features, window_indices, train_ids)
    x_val_arr = _sequences_to_tensor(transformer, features, window_indices, val_ids)
    x_test_arr = _sequences_to_tensor(transformer, features, window_indices, test_ids)

    heldout_row_indices = np.unique(window_indices[test_ids][:, -1])

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    seq_meta.iloc[test_ids].to_json(out / "test_sequence_index.json", orient="records", indent=2)

    metadata = {
        "split_mode": "temporal_sequence",
        "target_column": target_col,
        "input_columns": list(features.columns),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "feature_dim": int(x_train_arr.shape[-1]),
        "num_classes": int(len(encoder.classes_)),
        "classes": encoder.classes_.tolist(),
        "sequence_length": seq_len,
        "sequence_stride": stride,
        "temporal_windows": True,
        "timestamp_column": timestamp_column,
        "inactivity_seconds": temporal_cfg.get("inactivity_seconds", 30.0),
        "sequences_total": int(len(seq_meta)),
        "sequences_train": int(len(train_ids)),
        "sequences_val": int(len(val_ids)),
        "sequences_test": int(len(test_ids)),
    }
    return _save_prepared(
        out,
        x_train_arr,
        y[train_ids],
        x_val_arr,
        y[val_ids],
        x_test_arr,
        y[test_ids],
        transformer,
        encoder,
        metadata,
        heldout_row_indices=heldout_row_indices,
    )


def preprocess(config: dict[str, Any]) -> PreparedData:
    cfg = config["data"]
    raw_path = Path(cfg["raw_csv"])
    chunks = []
    for chunk in pd.read_csv(raw_path, nrows=cfg.get("max_rows"), dtype=str, chunksize=100000):
        chunks.append(chunk)
    frame = pd.concat(chunks, ignore_index=True)

    seq_len = int(cfg.get("sequence_length", 1))
    if seq_len > 1:
        return _preprocess_temporal(config, frame)
    if seq_len != 1:
        raise ValueError("sequence_length must be >= 1")
    return _preprocess_row_based(config, frame)


def load_prepared(output_dir: str | Path) -> PreparedData:
    out = Path(output_dir)
    arrays = np.load(out / "prepared.npz")
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    return PreparedData(
        arrays["x_train"],
        arrays["y_train"],
        arrays["x_val"],
        arrays["y_val"],
        arrays["x_test"],
        arrays["y_test"],
        metadata["num_classes"],
        metadata["feature_dim"],
    )


def heldout_indices(config: dict[str, Any]) -> np.ndarray:
    """Return held-out raw CSV row indices for streaming replay checks.

    Row-based preprocessing reproduces the stratified test split directly.
    Temporal preprocessing reads the saved final-row indices of test windows.
    """
    cfg = config["data"]
    out = Path(cfg["output_dir"])
    saved = out / "heldout_row_indices.npy"
    if saved.exists():
        return np.load(saved)

    raw_path = Path(cfg["raw_csv"])
    header = pd.read_csv(raw_path, nrows=0)
    label_col = _resolve_column(header, cfg.get("label_column", "auto"), LABEL_CANDIDATES)
    attack_type = _find_candidate(list(header.columns), ATTACK_TYPE_CANDIDATES)

    usecols = {label_col} | ({attack_type} if attack_type else set())
    chunks = []
    for chunk in pd.read_csv(raw_path, nrows=cfg.get("max_rows"), usecols=list(usecols), dtype=str, chunksize=100000):
        chunks.append(chunk)
    target_frame = pd.concat(chunks, ignore_index=True)
    target_col = attack_type if attack_type and target_frame[attack_type].nunique() > 2 else label_col
    y_raw = target_frame[target_col].astype(str).fillna("UNKNOWN")
    y = LabelEncoder().fit_transform(y_raw)

    seed = int(config.get("seed", 42))
    test_size = float(cfg.get("test_size", 0.15))
    val_size = float(cfg.get("validation_size", 0.15))

    idx = np.arange(len(y))
    idx_trainval, idx_test, y_trainval, y_test = train_test_split(
        idx, y, test_size=test_size, random_state=seed, stratify=y
    )
    relative_val = val_size / (1.0 - test_size)
    train_test_split(idx_trainval, y_trainval, test_size=relative_val, random_state=seed, stratify=y_trainval)

    prepared_path = out / "prepared.npz"
    if prepared_path.exists():
        metadata_path = out / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        if metadata.get("split_mode", "row") == "row":
            prepared_y_test = np.load(prepared_path)["y_test"]
            if not np.array_equal(y_test, prepared_y_test):
                raise AssertionError(
                    "heldout_indices() reproduced a different split than prepared.npz's y_test - "
                    "preprocess()'s split logic or config (seed/test_size/validation_size/max_rows) "
                    "changed without this function being updated to match. Refusing to return "
                    "indices that would silently replay the wrong rows."
                )
    return idx_test
