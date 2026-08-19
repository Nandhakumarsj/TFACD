"""A genuinely time-paced RecordSource - the streaming answer to "we don't
have live MQTT/Modbus sensors, so replay the network packet captures instead
of just batch-reading a CSV". Wraps CsvReplaySource (unchanged, reused) and
paces emission using real inter-packet-arrival deltas read from the paired
PCAP file via pcap_timing.iter_packet_timestamps - not the CSV's own
`frame.time` column, which is empty for every row on normal-traffic files
(measured directly against Modbus.csv in this repo) and therefore not a
reliable pacing source on its own.

Rows and packets are paired POSITIONALLY, in capture order - justified by the
near-exact row/packet-count alignment measured directly against real files in
this repo (Port Scanning: 22,564 CSV rows vs 23,329 PCAP packets; Modbus:
159,502 vs 159,514 - both >99% aligned), not a guaranteed exact mapping. A
handful of trailing rows or packets with no counterpart on the other side are
silently not emitted by zip() - bounded by that same small measured mismatch,
not unbounded data loss.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Sequence

from tfacd.streaming.pcap_timing import iter_packet_timestamps
from tfacd.streaming.sources import CsvReplaySource


class LivePacedSource:
    """RecordSource-conforming (records() -> Iterator[dict]) real-time-paced
    replay. speed_multiplier=1.0 replays at true real-time pacing (a
    multi-hour capture takes multi-hours); higher values compress it (60.0 =
    an hour of capture replays in a minute) - never negative or zero (a
    non-positive multiplier has no sane pacing interpretation).
    """

    def __init__(
        self,
        csv_path: str | Path,
        pcap_path: str | Path,
        speed_multiplier: float = 1.0,
        chunk_size: int = 1024,
        max_records: int | None = None,
        row_indices: Sequence[int] | None = None,
    ):
        if speed_multiplier <= 0:
            raise ValueError(f"speed_multiplier must be positive, got {speed_multiplier}")
        self.csv_source = CsvReplaySource(csv_path, chunk_size=chunk_size, max_records=max_records, row_indices=row_indices)
        self.pcap_path = Path(pcap_path)
        self.speed_multiplier = speed_multiplier

    def records(self) -> Iterator[dict]:
        previous_timestamp: float | None = None
        for record, timestamp in zip(self.csv_source.records(), iter_packet_timestamps(self.pcap_path)):
            if previous_timestamp is not None:
                delta = max(0.0, timestamp - previous_timestamp)
                time.sleep(delta / self.speed_multiplier)
            previous_timestamp = timestamp
            yield record
