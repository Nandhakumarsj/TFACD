from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalAudit:
    rows: int
    timestamp_column: str
    timestamp_valid_rows: int
    input_order_non_decreasing: bool
    unique_flow_keys: int
    flows_with_multiple_rows: int
    median_rows_per_flow: float
    p95_rows_per_flow: float
    max_rows_per_flow: int
    median_interarrival_ms: float | None
    p95_interarrival_ms: float | None
    group_columns: list[str]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def parse_timestamp(
    frame: pd.DataFrame, timestamp_column: str = "frame.time"
) -> pd.Series:
    if timestamp_column not in frame.columns:
        raise KeyError(f"Timestamp column '{timestamp_column}' not found")
    return pd.to_datetime(frame[timestamp_column], errors="coerce")


def resolve_group_columns(
    frame: pd.DataFrame, configured: list[str] | None = None
) -> list[str]:
    if configured:
        missing = [c for c in configured if c not in frame.columns]
        if missing:
            raise KeyError(f"Configured temporal grouping columns missing: {missing}")
        return configured

    candidates = [
        "ip.src_host",
        "ip.dst_host",
        "tcp.srcport",
        "tcp.dstport",
        "udp.port",
        "udp.stream",
        "ip.proto",
    ]
    return [c for c in candidates if c in frame.columns]


def _endpoint_pair(a: Any, b: Any) -> tuple[str, str]:
    left, right = str(a), str(b)
    return (left, right) if left <= right else (right, left)


def add_flow_key(
    frame: pd.DataFrame,
    group_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Add a bidirectional flow key without losing IP/port correspondence.

    For TCP, A:1000 -> B:80 and B:80 -> A:1000 must map to the same key.
    Identity fields are metadata only and must not be used as ML features.
    """
    out = frame.copy()
    columns = resolve_group_columns(out, group_columns)
    if not columns:
        raise ValueError("No usable flow grouping columns were found")

    if {
        "ip.src_host",
        "ip.dst_host",
        "tcp.srcport",
        "tcp.dstport",
    }.issubset(columns):
        src_ep = (
            out["ip.src_host"].fillna("<NA>").astype(str)
            + ":"
            + out["tcp.srcport"].fillna("<NA>").astype(str)
        )
        dst_ep = (
            out["ip.dst_host"].fillna("<NA>").astype(str)
            + ":"
            + out["tcp.dstport"].fillna("<NA>").astype(str)
        )

        ordered = pd.DataFrame({"src": src_ep, "dst": dst_ep}).apply(
            lambda row: tuple(sorted((row["src"], row["dst"]))),
            axis=1,
        )
        out["__endpoint_a"] = ordered.map(lambda x: x[0])
        out["__endpoint_b"] = ordered.map(lambda x: x[1])

        key_columns = ["__endpoint_a", "__endpoint_b"] + [
            c
            for c in columns
            if c
            not in {
                "ip.src_host",
                "ip.dst_host",
                "tcp.srcport",
                "tcp.dstport",
            }
        ]

    elif {"ip.src_host", "ip.dst_host"}.issubset(columns):
        endpoints = out.apply(
            lambda row: _endpoint_pair(
                row["ip.src_host"], row["ip.dst_host"]
            ),
            axis=1,
            result_type="expand",
        )
        out["__endpoint_a"] = endpoints[0]
        out["__endpoint_b"] = endpoints[1]

        key_columns = ["__endpoint_a", "__endpoint_b"] + [
            c for c in columns if c not in {"ip.src_host", "ip.dst_host"}
        ]
    else:
        key_columns = columns

    out["__flow_key"] = (
        out[key_columns].fillna("<NA>").astype(str).agg("|".join, axis=1)
    )
    return out, columns


def sessionize(
    frame: pd.DataFrame,
    timestamp_column: str = "frame.time",
    group_columns: list[str] | None = None,
    inactivity_seconds: float | None = 30.0,
) -> pd.DataFrame:
    """Order rows within flows and split long inactivity gaps into sessions."""
    out = frame.copy()
    out["__timestamp"] = parse_timestamp(out, timestamp_column)
    out = out.dropna(subset=["__timestamp"]).copy()
    out, _ = add_flow_key(out, group_columns)
    out["__original_index"] = out.index.to_numpy(dtype=np.int64)

    out = out.sort_values(
        ["__flow_key", "__timestamp"],
        kind="mergesort",
    ).reset_index(drop=True)

    if inactivity_seconds is None:
        out["__session_id"] = out["__flow_key"]
    else:
        gaps = (
            out.groupby("__flow_key", sort=False)["__timestamp"]
            .diff()
            .dt.total_seconds()
        )
        boundaries = gaps.isna() | gaps.gt(float(inactivity_seconds))
        session_number = (
            boundaries.groupby(out["__flow_key"]).cumsum().astype(int)
        )
        out["__session_id"] = (
            out["__flow_key"] + "|session=" + session_number.astype(str)
        )

    return out


def build_sequence_index(
    frame: pd.DataFrame,
    sequence_length: int,
    stride: int = 1,
    timestamp_column: str = "frame.time",
    group_columns: list[str] | None = None,
    inactivity_seconds: float | None = 30.0,
    label_column: str = "Attack_type",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build windows that never cross flow/session boundaries.

    The target label is the final row label in each sequence.
    """
    if sequence_length < 1 or stride < 1:
        raise ValueError("sequence_length and stride must be positive")
    if label_column not in frame.columns:
        raise KeyError(f"Label column '{label_column}' not found")

    ordered = sessionize(
        frame,
        timestamp_column=timestamp_column,
        group_columns=group_columns,
        inactivity_seconds=inactivity_seconds,
    )

    records: list[dict[str, Any]] = []
    windows: list[list[int]] = []

    for session_id, group in ordered.groupby("__session_id", sort=False):
        if len(group) < sequence_length:
            continue

        labels = group[label_column].astype(str).to_numpy()
        raw_indices = group["__original_index"].to_numpy(dtype=np.int64)
        timestamps = group["__timestamp"].to_numpy()

        for start in range(0, len(group) - sequence_length + 1, stride):
            end = start + sequence_length
            windows.append(raw_indices[start:end].tolist())
            records.append(
                {
                    "session_id": session_id,
                    "start_time": pd.Timestamp(timestamps[start]).isoformat(),
                    "end_time": pd.Timestamp(
                        timestamps[end - 1]
                    ).isoformat(),
                    "label": labels[end - 1],
                    "length": sequence_length,
                }
            )

    index_array = (
        np.asarray(windows, dtype=np.int64)
        if windows
        else np.empty((0, sequence_length), dtype=np.int64)
    )

    return pd.DataFrame.from_records(records), index_array


def audit_temporal_file(
    csv_path: str | Path,
    timestamp_column: str = "frame.time",
    group_columns: list[str] | None = None,
    sample_rows: int | None = None,
) -> TemporalAudit:
    """Profile timestamp validity, ordering, flow cardinality and inter-arrival."""
    frame = pd.read_csv(csv_path, nrows=sample_rows, low_memory=False)
    timestamps = parse_timestamp(frame, timestamp_column)

    valid = timestamps.notna()
    valid_ts = timestamps.loc[valid]
    non_decreasing = bool(valid_ts.is_monotonic_increasing)

    keyed, resolved = add_flow_key(frame.loc[valid].copy(), group_columns)
    counts = keyed["__flow_key"].value_counts()

    ordered_ts = valid_ts.sort_values().reset_index(drop=True)
    deltas = ordered_ts.diff().dropna().dt.total_seconds() * 1000.0
    deltas = deltas[deltas >= 0]

    return TemporalAudit(
        rows=len(frame),
        timestamp_column=timestamp_column,
        timestamp_valid_rows=int(valid.sum()),
        input_order_non_decreasing=non_decreasing,
        unique_flow_keys=int(counts.size),
        flows_with_multiple_rows=int((counts > 1).sum()),
        median_rows_per_flow=float(counts.median()) if len(counts) else 0.0,
        p95_rows_per_flow=float(counts.quantile(0.95)) if len(counts) else 0.0,
        max_rows_per_flow=int(counts.max()) if len(counts) else 0,
        median_interarrival_ms=float(deltas.median()) if len(deltas) else None,
        p95_interarrival_ms=float(deltas.quantile(0.95)) if len(deltas) else None,
        group_columns=resolved,
    )
