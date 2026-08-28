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
        max_target_row = max(self.row_indices) if self.row_indices else None
        for chunk in pd.read_csv(self.csv_path, chunksize=self.chunk_size, dtype=str, engine="python"):
            for record in chunk.to_dict(orient="records"):
                if self.row_indices is None or row_index in self.row_indices:
                    yield record
                    emitted += 1
                    if self.max_records is not None and emitted >= self.max_records:
                        return
                if max_target_row is not None and row_index >= max_target_row:
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


class SocketStreamSource:
    """Live TCP JSON-lines source.

    Acts as a **server**: binds to *host:port*, accepts one incoming connection,
    and yields newline-delimited JSON records until the peer closes or
    *max_records* are consumed.

    This server-side model suits the TFACD use-case where an upstream forwarder
    (Zeek, Suricata, a custom agent) pushes records to this process rather than
    this process pulling from a remote endpoint.

    Use InMemorySource in unit tests to avoid real network I/O.
    """

    def __init__(self, host: str, port: int, max_records: int | None = None, timeout_seconds: float = 30.0):
        self.host = host
        self.port = port
        self.max_records = max_records
        self.timeout_seconds = timeout_seconds

    def records(self) -> Iterator[dict]:
        import json
        import socket

        emitted = 0
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)
        srv.settimeout(self.timeout_seconds)
        try:
            conn, _ = srv.accept()
        except TimeoutError:
            srv.close()
            return
        srv.close()

        buf = b""
        with conn:
            conn.settimeout(self.timeout_seconds)
            while True:
                try:
                    chunk = conn.recv(4096)
                except TimeoutError:
                    break
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    emitted += 1
                    if self.max_records is not None and emitted >= self.max_records:
                        return


class MqttStreamSource:
    """MQTT topic subscriber that yields JSON-payload records as they arrive.

    Requires ``paho-mqtt`` (``pip install paho-mqtt``).  Subscribes to *topic*
    on *host:port* and collects messages until *max_records* are received.
    Each MQTT message payload must be a UTF-8 encoded JSON object.

    Design note: paho-mqtt is an optional dependency.  If it's not installed an
    ``ImportError`` is raised at *instantiation* time (not import time), so the
    rest of the module loads cleanly without it.
    """

    def __init__(self, host: str, port: int = 1883, topic: str = "tfacd/events/#", max_records: int | None = None, keepalive: int = 60):
        try:
            import paho.mqtt.client as mqtt  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("MqttStreamSource requires paho-mqtt: pip install paho-mqtt") from exc

        self.host = host
        self.port = port
        self.topic = topic
        self.max_records = max_records
        self.keepalive = keepalive
        self._mqtt = mqtt

    def records(self) -> Iterator[dict]:
        import json
        import queue

        q: queue.Queue[dict | None] = queue.Queue()
        emitted = 0
        client = self._mqtt.Client()

        def _on_message(_client, _userdata, msg):
            try:
                q.put(json.loads(msg.payload.decode("utf-8")))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        client.on_message = _on_message
        client.connect(self.host, self.port, self.keepalive)
        client.subscribe(self.topic)
        client.loop_start()

        try:
            while True:
                record = q.get(timeout=60)
                if record is None:
                    break
                yield record
                emitted += 1
                if self.max_records is not None and emitted >= self.max_records:
                    break
        finally:
            client.loop_stop()
            client.disconnect()
