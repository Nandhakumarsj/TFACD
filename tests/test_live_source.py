import struct

import pytest

from tfacd.streaming.live_source import LivePacedSource


def _write_pcap(path, timestamps):
    with path.open("wb") as handle:
        handle.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
        for ts in timestamps:
            sec, usec = int(ts), round((ts - int(ts)) * 1_000_000)
            handle.write(struct.pack("<IIII", sec, usec, 0, 0))


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        handle.write("a,b\n")
        for row in rows:
            handle.write(f"{row},{row}\n")


def test_records_are_emitted_in_order_matching_csv_rows(tmp_path):
    csv_path, pcap_path = tmp_path / "data.csv", tmp_path / "data.pcap"
    _write_csv(csv_path, [1, 2, 3])
    _write_pcap(pcap_path, [1_700_000_000.0, 1_700_000_000.1, 1_700_000_000.2])

    source = LivePacedSource(csv_path, pcap_path, speed_multiplier=1000.0)
    records = list(source.records())

    assert [r["a"] for r in records] == ["1", "2", "3"]


def test_sleeps_between_records_scaled_by_speed_multiplier(tmp_path, monkeypatch):
    csv_path, pcap_path = tmp_path / "data.csv", tmp_path / "data.pcap"
    _write_csv(csv_path, [1, 2, 3])
    _write_pcap(pcap_path, [1_700_000_000.0, 1_700_000_002.0, 1_700_000_005.0])  # deltas: 2.0s, 3.0s

    sleeps = []
    monkeypatch.setattr("tfacd.streaming.live_source.time.sleep", lambda s: sleeps.append(s))

    source = LivePacedSource(csv_path, pcap_path, speed_multiplier=2.0)
    list(source.records())

    assert sleeps == pytest.approx([1.0, 1.5])  # 2.0/2.0, 3.0/2.0 - no sleep before the first record


def test_no_sleep_call_for_a_single_record(tmp_path, monkeypatch):
    csv_path, pcap_path = tmp_path / "data.csv", tmp_path / "data.pcap"
    _write_csv(csv_path, [1])
    _write_pcap(pcap_path, [1_700_000_000.0])

    sleeps = []
    monkeypatch.setattr("tfacd.streaming.live_source.time.sleep", lambda s: sleeps.append(s))

    list(LivePacedSource(csv_path, pcap_path).records())

    assert sleeps == []


def test_mismatched_row_and_packet_counts_truncate_to_the_shorter_one(tmp_path, monkeypatch):
    csv_path, pcap_path = tmp_path / "data.csv", tmp_path / "data.pcap"
    _write_csv(csv_path, [1, 2, 3, 4, 5])  # 5 rows
    _write_pcap(pcap_path, [1_700_000_000.0, 1_700_000_000.1, 1_700_000_000.2])  # only 3 packets

    monkeypatch.setattr("tfacd.streaming.live_source.time.sleep", lambda s: None)
    records = list(LivePacedSource(csv_path, pcap_path, speed_multiplier=1000.0).records())

    assert [r["a"] for r in records] == ["1", "2", "3"]  # trailing rows 4,5 have no packet counterpart


def test_negative_or_zero_speed_multiplier_rejected(tmp_path):
    csv_path, pcap_path = tmp_path / "data.csv", tmp_path / "data.pcap"
    _write_csv(csv_path, [1])
    _write_pcap(pcap_path, [1_700_000_000.0])

    with pytest.raises(ValueError):
        LivePacedSource(csv_path, pcap_path, speed_multiplier=0.0)
