"""Feeds raw records into the streaming IDS pipeline.

No live MQTT/Modbus/OPC-UA/SCADA broker exists in this project - this replays
a static Edge-IIoTset capture instead. A real broker source would implement
the same RecordSource protocol; same documented-seam pattern as
capability_enforcement.py's CapabilityExecutor/SimulatedExecutor.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Protocol

import pandas as pd


class RecordSource(Protocol):
    def records(self) -> Iterator[dict]: ...


class CsvReplaySource:
    """Replays rows from a static CSV capture as raw record dicts.

    Reads with dtype=str deliberately: a chunked read otherwise infers dtypes
    PER CHUNK instead of over the whole file, and pandas silently returns
    float64 for columns that are genuinely object/str over the full file but
    happen to look numeric-or-all-NaN within one chunk. Verified against this
    dataset: 8 of 14 categorical columns corrupt this way at chunk sizes
    1000-50000, no exception raised - the fitted OneHotEncoder(handle_unknown=
    "ignore") then silently emits an all-zero block instead of the real
    category. features.py reconstructs proper dtypes downstream via
    pd.DataFrame(records) + targeted pd.to_numeric, which does not re-infer
    the same way a chunked read does.
    """

    def __init__(self, csv_path: str | Path, chunk_size: int = 1024, max_records: int | None = None, row_indices: Sequence[int] | None = None):
        self.csv_path = Path(csv_path)
        self.chunk_size = chunk_size
        self.max_records = max_records
        self.row_indices = set(row_indices) if row_indices is not None else None

    def records(self) -> Iterator[dict]:
        emitted = 0
        row_index = 0
        for chunk in pd.read_csv(self.csv_path, chunksize=self.chunk_size, dtype=str, low_memory=False):
            for record in chunk.to_dict(orient="records"):
                if self.row_indices is None or row_index in self.row_indices:
                    yield record
                    emitted += 1
                    if self.max_records is not None and emitted >= self.max_records:
                        return
                row_index += 1


class InMemorySource:
    """Replays an already-materialized list of records - lets a caller scan the
    CSV once (e.g. via CsvReplaySource) and run the pipeline multiple times
    over the same sample without paying the file-scan cost again."""

    def __init__(self, records: list[dict]):
        self._records = records

    def records(self) -> Iterator[dict]:
        yield from self._records
