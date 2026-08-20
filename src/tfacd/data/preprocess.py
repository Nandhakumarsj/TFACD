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

from tfacd.data.inspect import ATTACK_TYPE_CANDIDATES, LABEL_CANDIDATES, _find_candidate, identifier_like_columns


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


def preprocess(config: dict[str, Any]) -> PreparedData:
    cfg = config["data"]
    raw_path = Path(cfg["raw_csv"])
    frame = pd.read_csv(raw_path, nrows=cfg.get("max_rows"), low_memory=False)

    label_col = _resolve_column(frame, cfg.get("label_column", "auto"), LABEL_CANDIDATES)
    attack_type = _find_candidate(list(frame.columns), ATTACK_TYPE_CANDIDATES)
    target_col = attack_type if attack_type and frame[attack_type].nunique() > 2 else label_col

    explicit_drop = set(cfg.get("drop_columns", []))
    explicit_drop.update({label_col, target_col})
    if attack_type:
        explicit_drop.add(attack_type)

    x = frame.drop(columns=[c for c in explicit_drop if c in frame.columns]).copy()
    y_raw = frame[target_col].astype(str).fillna("UNKNOWN")

    # Guard against direct target leakage through duplicate semantic columns.
    leakage_tokens = {"attack_label", "attack_type", "attack", "label", "class", "target"}
    auto_drop = [
        column for column in x.columns
        if column.lower().strip().replace(" ", "_") in leakage_tokens
    ]
    # Guard against high-cardinality identifier columns (hosts, timestamps) that do not
    # generalize beyond the testbed that generated this capture.
    auto_drop += identifier_like_columns(list(x.columns), cfg.get("identifier_patterns", []))
    x = x.drop(columns=sorted(set(auto_drop)), errors="ignore")

    # Replace non-finite numeric values before the sklearn pipeline.
    x = x.replace([np.inf, -np.inf], np.nan)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    seed = int(config.get("seed", 42))
    test_size = float(cfg.get("test_size", 0.15))
    val_size = float(cfg.get("validation_size", 0.15))

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )
    relative_val = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=relative_val,
        random_state=seed,
        stratify=y_train_val,
    )

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

    x_train_arr = transformer.fit_transform(x_train).astype(np.float32)
    x_val_arr = transformer.transform(x_val).astype(np.float32)
    x_test_arr = transformer.transform(x_test).astype(np.float32)

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "prepared.npz",
        x_train=x_train_arr,
        y_train=y_train.astype(np.int64),
        x_val=x_val_arr,
        y_val=y_val.astype(np.int64),
        x_test=x_test_arr,
        y_test=y_test.astype(np.int64),
    )
    joblib.dump(transformer, out / "preprocessor.joblib")
    joblib.dump(encoder, out / "label_encoder.joblib")
    metadata = {
        "target_column": target_col,
        "input_columns": list(x.columns),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "feature_dim": int(x_train_arr.shape[1]),
        "num_classes": int(len(encoder.classes_)),
        "classes": encoder.classes_.tolist(),
        "auto_dropped_leakage": auto_drop,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return PreparedData(
        x_train_arr,
        y_train.astype(np.int64),
        x_val_arr,
        y_val.astype(np.int64),
        x_test_arr,
        y_test.astype(np.int64),
        len(encoder.classes_),
        x_train_arr.shape[1],
    )


def load_prepared(output_dir: str | Path) -> PreparedData:
    out = Path(output_dir)
    arrays = np.load(out / "prepared.npz")
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    return PreparedData(
        arrays["x_train"], arrays["y_train"], arrays["x_val"], arrays["y_val"],
        arrays["x_test"], arrays["y_test"], metadata["num_classes"], metadata["feature_dim"]
    )


def heldout_indices(config: dict[str, Any]) -> np.ndarray:
    """Re-derives the exact raw-row indices preprocess() set aside as the held-out
    test split, straight from the raw CSV's label column - without re-running the
    full pipeline or touching the (300+MB) preprocessor.joblib. Streaming replay
    can then feed these rows back through a StreamingFeatureExtractor for an
    honest accuracy readout: any deviation from Gate 2/3's offline macro-F1
    indicates a pipeline bug, not generalization, because these are the same
    rows already scored offline.

    Reuses the same target-column resolution and split calls as preprocess()
    precisely so the two can't silently drift apart; asserted below rather than
    assumed, in case they ever do anyway.
    """
    cfg = config["data"]
    raw_path = Path(cfg["raw_csv"])
    header = pd.read_csv(raw_path, nrows=0)
    label_col = _resolve_column(header, cfg.get("label_column", "auto"), LABEL_CANDIDATES)
    attack_type = _find_candidate(list(header.columns), ATTACK_TYPE_CANDIDATES)

    usecols = {label_col} | ({attack_type} if attack_type else set())
    target_frame = pd.read_csv(raw_path, nrows=cfg.get("max_rows"), usecols=list(usecols), low_memory=False)
    target_col = attack_type if attack_type and target_frame[attack_type].nunique() > 2 else label_col
    y_raw = target_frame[target_col].astype(str).fillna("UNKNOWN")
    y = LabelEncoder().fit_transform(y_raw)

    seed = int(config.get("seed", 42))
    test_size = float(cfg.get("test_size", 0.15))
    val_size = float(cfg.get("validation_size", 0.15))

    idx = np.arange(len(y))
    idx_trainval, idx_test, y_trainval, y_test = train_test_split(idx, y, test_size=test_size, random_state=seed, stratify=y)
    relative_val = val_size / (1.0 - test_size)
    train_test_split(idx_trainval, y_trainval, test_size=relative_val, random_state=seed, stratify=y_trainval)

    prepared_path = Path(cfg["output_dir"]) / "prepared.npz"
    if prepared_path.exists():
        prepared_y_test = np.load(prepared_path)["y_test"]
        if not np.array_equal(y_test, prepared_y_test):
            raise AssertionError(
                "heldout_indices() reproduced a different split than prepared.npz's y_test - "
                "preprocess()'s split logic or config (seed/test_size/validation_size/max_rows) "
                "changed without this function being updated to match. Refusing to return "
                "indices that would silently replay the wrong rows."
            )
    return idx_test
