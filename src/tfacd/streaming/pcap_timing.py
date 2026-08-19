"""Read-header-only PCAP timestamp reader - stdlib `struct` only, no scapy/
pyshark/dpkt dependency. Deliberately narrow: it never touches a byte of
packet content, only the 24-byte global header and each 16-byte per-record
header (ts_sec, ts_usec, incl_len, orig_len), seeking past the payload via
incl_len. This carries none of the "silently wrong feature derivation" risk
a full protocol dissector would - it exists purely to drive realistic
inter-arrival pacing for LivePacedSource (live_source.py), not to extract
features. Feature values still come entirely from the paired CSV, which this
module never reads.

Verified directly against real files in this repo (not assumed): parsing
`Attack traffic/Port Scanning attack.pcap` recovers 23,329 packets whose
first/last timestamps match that file's paired CSV's own `frame.time`
column to the microsecond. `Normal traffic/Modbus/Modbus.pcap` has 159,514
packets against 159,502 CSV rows in the paired Modbus.csv - whose own
`frame.time` column is empty for every row - confirming the PCAP is the only
reliable timing source for normal-traffic files, not merely a convenient one.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterator

_GLOBAL_HEADER_SIZE = 24
_RECORD_HEADER_SIZE = 16
_MAGIC_LE = 0xA1B2C3D4
_MAGIC_BE = 0xD4C3B2A1


class PcapFormatError(ValueError):
    """Raised when the file's magic number doesn't match either classic PCAP
    byte order - e.g. a PCAPNG file (different format entirely) or a
    non-PCAP file. Never silently guessed at."""


def iter_packet_timestamps(pcap_path: str | Path) -> Iterator[float]:
    """Yields each packet's capture timestamp (Unix epoch seconds, float) in
    on-disk (== capture) order. Never reads packet payload bytes."""
    with Path(pcap_path).open("rb") as handle:
        header = handle.read(_GLOBAL_HEADER_SIZE)
        if len(header) < _GLOBAL_HEADER_SIZE:
            raise PcapFormatError(f"{pcap_path}: file shorter than a PCAP global header")
        magic = struct.unpack("<I", header[:4])[0]
        if magic == _MAGIC_LE:
            endian = "<"
        elif magic == _MAGIC_BE:
            endian = ">"
        else:
            raise PcapFormatError(f"{pcap_path}: magic number {magic:#x} is not classic PCAP (little/big-endian) - PCAPNG or not a capture file?")

        while True:
            record_header = handle.read(_RECORD_HEADER_SIZE)
            if len(record_header) < _RECORD_HEADER_SIZE:
                return
            ts_sec, ts_usec, incl_len, _orig_len = struct.unpack(endian + "IIII", record_header)
            yield ts_sec + ts_usec / 1_000_000
            handle.seek(incl_len, 1)


def packet_count(pcap_path: str | Path) -> int:
    """Convenience full scan - same cost as iterating and discarding, exposed
    separately since callers that just want a count (e.g. a scenario report)
    shouldn't have to materialize a list."""
    return sum(1 for _ in iter_packet_timestamps(pcap_path))
