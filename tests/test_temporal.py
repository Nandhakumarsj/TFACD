import pandas as pd

from tfacd.data.temporal import (
    add_flow_key,
    build_sequence_index,
    sessionize,
)


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame.time": [
                "2021-01-01 00:00:00.000",
                "2021-01-01 00:00:00.010",
                "2021-01-01 00:00:00.020",
                "2021-01-01 00:01:00.000",
                "2021-01-01 00:01:00.010",
            ],
            "ip.src_host": ["A", "B", "A", "A", "B"],
            "ip.dst_host": ["B", "A", "B", "B", "A"],
            "tcp.srcport": [1000, 80, 1000, 1000, 80],
            "tcp.dstport": [80, 1000, 80, 80, 1000],
            "ip.proto": [6, 6, 6, 6, 6],
            "Attack_type": ["Normal", "Normal", "DDoS_TCP", "DDoS_TCP", "DDoS_TCP"],
        }
    )


def test_flow_key_is_bidirectional():
    keyed, _ = add_flow_key(_sample_frame())
    assert keyed.loc[0, "__flow_key"] == keyed.loc[1, "__flow_key"]


def test_sessionize_splits_long_gap():
    ordered = sessionize(_sample_frame(), inactivity_seconds=30)
    assert ordered["__session_id"].nunique() == 2


def test_windows_do_not_cross_sessions():
    meta, windows = build_sequence_index(
        _sample_frame(), sequence_length=2, stride=1, inactivity_seconds=30
    )
    # First session has 3 rows -> 2 windows. Second session has 2 rows -> 1.
    assert len(meta) == 3
    assert windows.shape == (3, 2)
    assert meta["session_id"].nunique() == 2


def test_sequence_labels_use_final_row():
    meta, _ = build_sequence_index(
        _sample_frame(), sequence_length=2, stride=1, inactivity_seconds=30
    )
    assert set(meta["label"]) == {"Normal", "DDoS_TCP"}
