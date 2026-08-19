import math

import pandas as pd

from tfacd.streaming.sources import CsvReplaySource, InMemorySource


def _make_csv(tmp_path, n=10):
    frame = pd.DataFrame({"a_number": range(n), "a_label": [f"class-{i % 3}" for i in range(n)]})
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    return path


def test_yields_all_rows_across_multiple_chunks(tmp_path):
    path = _make_csv(tmp_path, n=10)
    source = CsvReplaySource(path, chunk_size=3)
    records = list(source.records())
    assert len(records) == 10
    assert [r["a_number"] for r in records] == [str(i) for i in range(10)]


def test_max_records_stops_early(tmp_path):
    path = _make_csv(tmp_path, n=10)
    source = CsvReplaySource(path, chunk_size=3, max_records=4)
    records = list(source.records())
    assert len(records) == 4


def test_row_indices_filters_to_a_specific_subset(tmp_path):
    path = _make_csv(tmp_path, n=10)
    source = CsvReplaySource(path, chunk_size=3, row_indices=[2, 5, 9])
    records = list(source.records())
    assert [r["a_number"] for r in records] == ["2", "5", "9"]


def test_row_indices_combined_with_max_records(tmp_path):
    path = _make_csv(tmp_path, n=10)
    source = CsvReplaySource(path, chunk_size=3, row_indices=[2, 5, 9], max_records=2)
    records = list(source.records())
    assert [r["a_number"] for r in records] == ["2", "5"]


def test_every_value_arrives_as_string_never_auto_parsed(tmp_path):
    path = _make_csv(tmp_path, n=5)
    source = CsvReplaySource(path, chunk_size=2)
    for record in source.records():
        for value in record.values():
            assert isinstance(value, str) or (isinstance(value, float) and math.isnan(value))


def test_in_memory_source_replays_the_same_records_repeatedly():
    records = [{"a": "1"}, {"a": "2"}, {"a": "3"}]
    source = InMemorySource(records)
    first_pass = list(source.records())
    second_pass = list(source.records())
    assert first_pass == records
    assert second_pass == records
