import numpy as np
import pandas as pd

from tfacd.data.preprocess import _split_temporal_ids_by_session


def test_chronological_session_split_has_no_overlap_and_preserves_time_order():
    rows = []
    for session, minute in enumerate((0, 2, 4, 6)):
        for second in (0, 1, 2):
            rows.append({
                "session_id": f"flow-{session}|session=1",
                "start_time": f"2024-01-01T00:{minute:02d}:{second:02d}",
                "end_time": f"2024-01-01T00:{minute:02d}:{second:02d}",
            })
    meta = pd.DataFrame(rows)
    meta.index = np.arange(len(meta))

    train, val, test = _split_temporal_ids_by_session(meta, 0.25, 0.25, seed=42, chronological=True)
    train_sessions = set(meta.iloc[train]["session_id"])
    val_sessions = set(meta.iloc[val]["session_id"])
    test_sessions = set(meta.iloc[test]["session_id"])

    assert train_sessions.isdisjoint(val_sessions)
    assert train_sessions.isdisjoint(test_sessions)
    assert val_sessions.isdisjoint(test_sessions)
    assert max(meta.iloc[train]["end_time"]) <= min(meta.iloc[val]["start_time"])
    assert max(meta.iloc[val]["end_time"]) <= min(meta.iloc[test]["start_time"])
