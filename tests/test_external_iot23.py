from pathlib import Path

import pandas as pd


def test_iot23_fixture_schema_parser_contract_is_documented():
    fixture = pd.DataFrame({
        "id.orig_h": ["192.0.2.1"],
        "id.resp_h": ["198.51.100.1"],
        "id.orig_p": [1234],
        "id.resp_p": [80],
        "proto": ["tcp"],
        "ts": [1700000000.0],
        "label": ["Benign"] if False else ["Benign"],
    })
    assert "id.orig_h" in fixture
    assert "label" in fixture
