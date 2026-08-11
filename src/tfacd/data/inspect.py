from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from tfacd.common.config import load_config

LABEL_CANDIDATES = ["attack_label", "label", "class", "target", "binary_label"]
ATTACK_TYPE_CANDIDATES = ["attack_type", "attack", "category", "attack_category"]
TIME_CANDIDATES = ["timestamp", "time", "frame.time", "ts"]
GROUP_CANDIDATES = ["flow_id", "session_id", "device_id", "sensor", "source"]


def identifier_like_columns(columns: list[str], patterns: list[str]) -> list[str]:
    compiled = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    return [column for column in columns if any(pattern.search(column) for pattern in compiled)]


def _find_candidate(columns: list[str], candidates: list[str]) -> str | None:
    lookup = {column.lower().strip(): column for column in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    for column in columns:
        lower = column.lower().strip()
        if any(candidate in lower for candidate in candidates):
            return column
    return None


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_csv(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = config["data"]
    csv_path = Path(data_cfg["raw_csv"]).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. Update data.raw_csv in the YAML config."
        )

    max_rows = data_cfg.get("max_rows")
    frame = pd.read_csv(csv_path, nrows=max_rows, low_memory=False)
    columns = list(frame.columns)

    label = _find_candidate(columns, LABEL_CANDIDATES)
    attack_type = _find_candidate(columns, ATTACK_TYPE_CANDIDATES)
    timestamp = _find_candidate(columns, TIME_CANDIDATES)
    group = _find_candidate(columns, GROUP_CANDIDATES)

    identifier_like = identifier_like_columns(columns, data_cfg.get("identifier_patterns", []))

    likely_leakage = []
    for column in columns:
        lower = column.lower()
        if column in {label, attack_type} or any(token in lower for token in ["attack_name", "attack type"]):
            likely_leakage.append(column)

    report = {
        "path": str(csv_path.resolve()),
        "sha256": _sha256(csv_path),
        "rows_loaded": int(len(frame)),
        "columns": len(columns),
        "column_names": columns,
        "dtypes": {key: str(value) for key, value in frame.dtypes.items()},
        "candidate_columns": {
            "label": label,
            "attack_type": attack_type,
            "timestamp": timestamp,
            "group": group,
        },
        "identifier_like_columns": identifier_like,
        "likely_leakage_columns": sorted(set(likely_leakage)),
        "missing_fraction": frame.isna().mean().sort_values(ascending=False).head(30).to_dict(),
        "duplicate_rows": int(frame.duplicated().sum()),
        "unique_counts": frame.nunique(dropna=False).sort_values().head(30).to_dict(),
        "temporal_warning": (
            "No clear timestamp/group column was detected. Do not claim temporal sequence learning "
            "from arbitrary row order."
            if timestamp is None and group is None
            else "Candidate ordering metadata detected; verify that rows were not shuffled."
        ),
    }
    for column in [label, attack_type]:
        if column:
            report[f"distribution::{column}"] = frame[column].value_counts(dropna=False).head(100).to_dict()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/edge_iiot.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    report = inspect_csv(config)
    output_dir = Path(config["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "schema_report.json"
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report["candidate_columns"], indent=2))
    print(report["temporal_warning"])
    print(f"Saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
