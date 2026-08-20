import struct
from pathlib import Path

import pytest

from tfacd.streaming.pcap_timing import PcapFormatError, iter_packet_timestamps, packet_count

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_SMALL_PCAP = REPO_ROOT / "datasets" / "Edge_IIoTset" / "Attack traffic" / "OS Fingerprinting attack.pcap"


def _write_synthetic_pcap(path: Path, records: list[tuple[int, int, bytes]], endian: str = "<") -> None:
    """records: list of (ts_sec, ts_usec, payload_bytes). The magic constant
    is always 0xA1B2C3D4 packed in the given byte order - packing a
    different literal per branch would write the wrong on-disk bytes for a
    real big-endian-written capture (this project's reader distinguishes
    endianness by which byte order the SAME magic constant round-trips
    through, not by swapping which constant gets written)."""
    global_header = struct.pack(endian + "IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)
    with path.open("wb") as handle:
        handle.write(global_header)
        for ts_sec, ts_usec, payload in records:
            handle.write(struct.pack(endian + "IIII", ts_sec, ts_usec, len(payload), len(payload)))
            handle.write(payload)


def test_synthetic_pcap_timestamps_recovered_exactly(tmp_path):
    path = tmp_path / "synthetic.pcap"
    records = [(1700000000, 0, b"\x00" * 10), (1700000001, 500000, b"\x01" * 4), (1700000002, 999999, b"")]
    _write_synthetic_pcap(path, records)

    timestamps = list(iter_packet_timestamps(path))

    assert timestamps == pytest.approx([1700000000.0, 1700000001.5, 1700000002.999999])


def test_synthetic_pcap_big_endian_is_handled(tmp_path):
    path = tmp_path / "synthetic_be.pcap"
    records = [(1700000000, 250000, b"\x00" * 8)]
    _write_synthetic_pcap(path, records, endian=">")

    timestamps = list(iter_packet_timestamps(path))

    assert timestamps == pytest.approx([1700000000.25])


def test_packet_content_is_never_read_only_headers(tmp_path):
    """Seeks past payload bytes rather than reading them - a payload full of
    garbage that would break any real protocol parser must not raise here."""
    path = tmp_path / "garbage_payload.pcap"
    _write_synthetic_pcap(path, [(1700000000, 0, bytes(range(256)) * 4)])

    timestamps = list(iter_packet_timestamps(path))

    assert len(timestamps) == 1


def test_empty_capture_yields_nothing(tmp_path):
    path = tmp_path / "empty.pcap"
    _write_synthetic_pcap(path, [])

    assert list(iter_packet_timestamps(path)) == []


def test_non_pcap_file_raises_pcap_format_error(tmp_path):
    path = tmp_path / "not_a_pcap.bin"
    path.write_bytes(b"this is not a pcap file at all, just plain bytes")

    with pytest.raises(PcapFormatError):
        list(iter_packet_timestamps(path))


def test_truncated_global_header_raises(tmp_path):
    path = tmp_path / "truncated.pcap"
    path.write_bytes(b"\xd4\xc3\xb2\xa1\x02\x00")  # far short of 24 bytes

    with pytest.raises(PcapFormatError):
        list(iter_packet_timestamps(path))


def test_packet_count_matches_number_of_records(tmp_path):
    path = tmp_path / "synthetic.pcap"
    _write_synthetic_pcap(path, [(1700000000, 0, b"a"), (1700000001, 0, b"b"), (1700000002, 0, b"c")])

    assert packet_count(path) == 3


@pytest.mark.skipif(not REAL_SMALL_PCAP.exists(), reason="dataset not present on this machine")
def test_real_small_pcap_file_parses_without_error_and_is_time_ordered():
    timestamps = list(iter_packet_timestamps(REAL_SMALL_PCAP))

    assert len(timestamps) > 0
    assert timestamps == sorted(timestamps)
