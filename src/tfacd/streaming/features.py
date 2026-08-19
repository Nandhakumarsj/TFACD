"""Feature extraction over the ALREADY-FITTED training-plane transformer.

Owns the leakage invariant: only metadata["input_columns"] (the 56 non-dropped
columns) ever reach the fitted transformer - raw ip.src_host/ip.dst_host/
Attack_type can flow freely through sources.py's record dicts without risk,
since they're simply never selected here.

Also owns the fix for a real, verified silent-corruption risk: a chunked CSV
read (sources.py) infers dtypes per chunk instead of over the whole file, so
categorical columns can arrive here as float64 instead of str/object.
Reconstructing via pd.DataFrame(records) from already-parsed Python values
(not re-parsing CSV text) does not re-infer dtypes the same way a chunk read
does - verified against 8 real columns that silently corrupt under the naive
"transform the chunk directly" path (the fitted OneHotEncoder(handle_unknown=
"ignore") accepts the wrong dtype without error and emits an all-zero block).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class StreamingFeatureExtractor:
    def __init__(self, output_dir: str | Path):
        out = Path(output_dir)
        self.transformer = joblib.load(out / "preprocessor.joblib")
        self.metadata: dict[str, Any] = json.loads((out / "metadata.json").read_text(encoding="utf-8"))

    def transform(self, records: list[dict]) -> np.ndarray:
        if not records:
            return np.empty((0, self.metadata["feature_dim"]), dtype=np.float32)

        frame = pd.DataFrame(records)
        missing = [c for c in self.metadata["input_columns"] if c not in frame.columns]
        if missing:
            raise KeyError(f"records are missing required columns: {missing}")

        for column in self.metadata["numeric_columns"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        # categorical_columns are deliberately left as-is (object/str) - the
        # fitted OneHotEncoder expects string categories, matching training.

        frame = frame[self.metadata["input_columns"]].replace([np.inf, -np.inf], np.nan)
        arr = self.transformer.transform(frame).astype(np.float32)  # .transform only - never .fit/.fit_transform
        if arr.shape[1] != self.metadata["feature_dim"]:
            raise ValueError(f"transform produced {arr.shape[1]} features, expected {self.metadata['feature_dim']}")
        return arr
