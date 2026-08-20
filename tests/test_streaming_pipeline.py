from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from tfacd.common.config import load_config
from tfacd.data.preprocess import heldout_indices
from tfacd.streaming.pipeline import StreamingIDS
from tfacd.streaming.sources import CsvReplaySource

CLASSES = ["Normal", "AttackA", "AttackB"]


class _FakeExtractor:
    """Records carry their own desired logits under "_logits" - full control
    over which class "wins" per test case without needing a real transformer."""

    def __init__(self, feature_dim=3):
        self.metadata = {"feature_dim": feature_dim, "classes": CLASSES}

    def transform(self, records):
        if not records:
            return np.empty((0, self.metadata["feature_dim"]), dtype=np.float32)
        return np.array([r["_logits"] for r in records], dtype=np.float32)


class _IdentityLogitsModel(nn.Module):
    def forward(self, x):
        return x[:, 0, :]  # [n, 1, C] -> [n, C]; treats the injected features as logits directly


class _ListSource:
    def __init__(self, records):
        self._records = records

    def records(self):
        yield from self._records


def make_ids(**kwargs):
    return StreamingIDS(_FakeExtractor(), _IdentityLogitsModel(), CLASSES, device=torch.device("cpu"), **kwargs)


def test_sequence_length_other_than_one_is_refused():
    with pytest.raises(NotImplementedError):
        make_ids(sequence_length=2)


def test_decodes_predicted_class_confidence_and_provenance():
    ids = make_ids(emit_normal=True, batch_size=10)
    records = [{"_logits": [0.0, 5.0, 0.0], "ip.src_host": "10.0.0.1", "ip.dst_host": "10.0.0.2"}]
    alerts, stats = ids.run(_ListSource(records))
    assert len(alerts) == 1
    assert alerts[0].attack_type == "AttackA"
    assert 0.0 < alerts[0].confidence <= 1.0
    assert alerts[0].source_id == "10.0.0.1"
    assert alerts[0].target_asset == "10.0.0.2"
    assert alerts[0].protocol is None
    assert stats.alerts_emitted == 1


def test_source_id_and_target_asset_are_canonicalized():
    """Verified live: raw ip.src_host/ip.dst_host flow unmodified into the LLM
    decision engine's prompt and semantic_risk's scoring - canonicalized once
    here, at construction, so every downstream consumer gets the sanitized value."""
    ids = make_ids(emit_normal=True, batch_size=10)
    records = [{"_logits": [0.0, 5.0, 0.0], "ip.src_host": "10.0.0.1​​", "ip.dst_host": "﻿10.0.0.2"}]
    alerts, _ = ids.run(_ListSource(records))
    assert alerts[0].source_id == "10.0.0.1"  # zero-width spaces stripped
    assert alerts[0].target_asset == "10.0.0.2"  # BOM/zero-width-no-break-space stripped


def test_missing_provenance_fields_stay_none_not_crash():
    ids = make_ids(emit_normal=True, batch_size=10)
    records = [{"_logits": [0.0, 5.0, 0.0]}]  # no ip.src_host/ip.dst_host at all
    alerts, _ = ids.run(_ListSource(records))
    assert alerts[0].source_id is None
    assert alerts[0].target_asset is None


def test_normal_suppressed_by_default():
    ids = make_ids()  # emit_normal defaults False
    records = [{"_logits": [5.0, 0.0, 0.0], "ip.src_host": "a", "ip.dst_host": "b"}]
    alerts, stats = ids.run(_ListSource(records))
    assert alerts == []
    assert stats.alerts_suppressed == 1
    assert stats.records_read == 1


def test_min_confidence_suppresses_low_confidence_alerts():
    ids = make_ids(emit_normal=True, min_confidence=0.99)
    records = [{"_logits": [0.0, 1.0, 0.9], "ip.src_host": "a", "ip.dst_host": "b"}]  # close logits -> low softmax confidence
    alerts, stats = ids.run(_ListSource(records))
    assert alerts == []
    assert stats.alerts_suppressed == 1


def test_batches_split_correctly_across_multiple_flushes():
    ids = make_ids(emit_normal=True, batch_size=2)
    records = [{"_logits": [0.0, 5.0, 0.0], "ip.src_host": str(i), "ip.dst_host": "b"} for i in range(5)]
    alerts, stats = ids.run(_ListSource(records))
    assert stats.records_read == 5
    assert stats.alerts_emitted == 5
    assert len(alerts) == 5


def test_stats_report_nonnegative_timings_and_throughput():
    ids = make_ids(emit_normal=True)
    records = [{"_logits": [0.0, 5.0, 0.0], "ip.src_host": "a", "ip.dst_host": "b"} for _ in range(3)]
    _, stats = ids.run(_ListSource(records))
    assert stats.total_seconds >= 0.0
    assert stats.records_per_second >= 0.0


def test_golden_against_real_heldout_rows():
    """Artifact-gated: skipped if this session's real training artifacts aren't
    present. Confirms the whole streaming pipeline (source -> features ->
    inference) reproduces sane predictions on real held-out rows, not just
    that it runs without crashing."""
    output_dir = Path("artifacts/data")
    checkpoint = Path("artifacts/models/flower_ftil_final.pt")
    if not (output_dir / "prepared.npz").exists() or not checkpoint.exists():
        pytest.skip("real training artifacts not present")

    config = load_config("configs/edge_iiot.yaml")
    if not Path(config["data"]["raw_csv"]).exists():
        pytest.skip(f"raw CSV dataset not present at {config['data']['raw_csv']}")

    config["streaming"] = {
        "model_path": str(checkpoint),
        "require_signature": False,
        "require_certified_status": False,  # this test checks feature/inference correctness, not the certification gate (see test_certification.py)
        "batch_size": 64,
        "emit_normal": True,
    }
    ids = StreamingIDS.from_config(config)

    idx_test = heldout_indices(config)
    idx_test_list = idx_test.tolist()
    sample = sorted(np.random.default_rng(0).choice(idx_test, 200, replace=False).tolist())
    positions = [idx_test_list.index(i) for i in sample]
    true_labels = np.load(output_dir / "prepared.npz")["y_test"][positions]
    metadata = ids.extractor.metadata
    true_class_names = [metadata["classes"][i] for i in true_labels]

    source = CsvReplaySource(config["data"]["raw_csv"], chunk_size=1024, row_indices=sample)
    alerts, stats = ids.run(source)

    assert stats.records_read == 200
    assert len(alerts) == 200  # emit_normal=True, min_confidence=0.0 - one alert per record
    assert stats.records_per_second > 0
    predicted_class_names = [a.attack_type for a in alerts]
    accuracy = sum(p == t for p, t in zip(predicted_class_names, true_class_names)) / len(true_class_names)
    assert accuracy > 0.8  # sanity bound - the real offline macro-F1 is ~0.83 on this checkpoint (evaluate_checkpoint.py is the authoritative number)
