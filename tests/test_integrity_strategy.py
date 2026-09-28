import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, MetricRecord, RecordDict

from tfacd.federated.integrity_strategy import IntegrityAwareStrategy


class FakeMetadata:
    """Stand-in for Flower's authenticated Message.metadata - only
    src_node_id is read by aggregate_train."""

    def __init__(self, src_node_id):
        self.src_node_id = src_node_id


class FakeReply:
    """Stand-in for a Flower reply Message - aggregate_train only calls
    .has_error(), reads .content, and (for identity) .metadata.src_node_id."""

    def __init__(self, content=None, has_error=False, src_node_id=None):
        self.content = content
        self._has_error = has_error
        self.metadata = FakeMetadata(src_node_id)

    def has_error(self):
        return self._has_error


def make_reply(client_id: str, state: dict, num_examples: int = 100, src_node_id: str | None = None) -> FakeReply:
    """src_node_id defaults to client_id so existing tests keep working
    unchanged; pass a different src_node_id to simulate the authenticated
    Flower identity diverging from whatever a client claims in its own
    client-metadata (see test_client_metadata_cannot_spoof_identity)."""
    content = RecordDict(
        {
            "arrays": ArrayRecord({k: torch.from_numpy(v) for k, v in state.items()}),
            "metrics": MetricRecord({"num-examples": num_examples}),
            "client-metadata": ConfigRecord({"client-id": client_id}),
        }
    )
    return FakeReply(content=content, src_node_id=src_node_id if src_node_id is not None else client_id)


def reference_state():
    # Non-zero, like a real (never literally all-zero) model checkpoint - an
    # all-zero reference makes validate_update's norm-ratio check degenerate
    # (any nonzero update looks infinitely large against a zero baseline).
    return {"w": np.full((3, 3), 1.0, dtype=np.float32)}


def build_strategy(tmp_path, **kwargs):
    strategy = IntegrityAwareStrategy(trust_log_path=tmp_path / "trust_log.jsonl", **kwargs)
    strategy._current_arrays = ArrayRecord({k: torch.from_numpy(v) for k, v in reference_state().items()})
    return strategy


def test_error_replies_are_excluded_and_rest_aggregated(tmp_path):
    strategy = build_strategy(tmp_path)
    benign = [{"w": np.full((3, 3), 1.0 + v, dtype=np.float32)} for v in (0.01, -0.01, 0.02, -0.02, 0.0)]
    replies = [make_reply(str(i), s) for i, s in enumerate(benign)]
    replies.append(FakeReply(has_error=True))

    arrays, metrics = strategy.aggregate_train(1, replies)

    assert arrays is not None
    assert int(metrics["ftil_rejected_error"]) == 1
    assert int(metrics["ftil_accepted"]) == 5


def test_structurally_invalid_update_rejected_before_detector(tmp_path):
    strategy = build_strategy(tmp_path, max_update_norm_ratio=10.0)
    benign = [{"w": np.full((3, 3), 1.0 + v, dtype=np.float32)} for v in (0.01, -0.01, 0.02, -0.02)]
    malicious = {"w": np.full((3, 3), 1_000_000.0, dtype=np.float32)}  # model-replacement-style attack
    replies = [make_reply(str(i), s) for i, s in enumerate(benign)]
    replies.append(make_reply("malicious", malicious))

    arrays, metrics = strategy.aggregate_train(1, replies)

    assert int(metrics["ftil_rejected_validation"]) == 1
    assert int(metrics["ftil_accepted"]) == 4
    assert arrays is not None


def test_statistical_outlier_flagged_by_detector_and_excluded_from_aggregate(tmp_path):
    strategy = build_strategy(tmp_path, ema_alpha=1.0, reject_below_trust=0.5)
    rng = np.random.default_rng(0)
    benign = [{"w": 1.0 + rng.normal(0, 0.01, size=(3, 3)).astype(np.float32)} for _ in range(5)]
    outlier = {"w": np.full((3, 3), 4.0, dtype=np.float32)}  # within the norm-ratio bound but a clear outlier
    replies = [make_reply(str(i), s) for i, s in enumerate(benign)]
    replies.append(make_reply("outlier", outlier))

    arrays, metrics = strategy.aggregate_train(1, replies)

    assert int(metrics["ftil_rejected_detector"]) >= 1
    assert int(metrics["ftil_accepted"]) < 6
    assert arrays is not None
    # the outlier should not have dragged the aggregate away from the benign cluster (~1.0)
    aggregated_state = arrays.to_torch_state_dict()
    assert abs(aggregated_state["w"].mean().item() - 1.0) < 0.5


def test_all_replies_rejected_returns_none_arrays(tmp_path):
    strategy = build_strategy(tmp_path, max_update_norm_ratio=10.0)
    malicious = {"w": np.full((3, 3), 1_000_000.0, dtype=np.float32)}
    replies = [make_reply("only-malicious", malicious)]

    arrays, metrics = strategy.aggregate_train(1, replies)

    assert arrays is None
    assert int(metrics["ftil_accepted"]) == 0


def test_zero_dim_integer_buffer_survives_aggregation(tmp_path):
    """Regression: BatchNorm's num_batches_tracked is a 0-d int64 tensor.
    trimmed_mean's np.stack(...).mean(axis=0) hands back a bare numpy scalar
    (not an ndarray) for a 0-d input, which torch.from_numpy rejects unless
    _numpy_to_arrayrecord wraps it back in np.asarray first."""
    strategy = build_strategy(tmp_path)
    strategy._current_arrays = ArrayRecord(
        {"w": torch.from_numpy(np.full((3, 3), 1.0, dtype=np.float32)), "bn.num_batches_tracked": torch.tensor(5, dtype=torch.int64)}
    )
    replies = [
        make_reply(
            str(i),
            {"w": np.full((3, 3), 1.0 + v, dtype=np.float32), "bn.num_batches_tracked": np.array(5 + i, dtype=np.int64)},
        )
        for i, v in enumerate((0.01, -0.01, 0.02))
    ]

    arrays, metrics = strategy.aggregate_train(1, replies)

    assert arrays is not None
    assert int(metrics["ftil_accepted"]) == 3


def test_client_metadata_cannot_spoof_identity(tmp_path):
    """REVIEW FIX (P0/P1): client_id must come from the authenticated
    msg.metadata.src_node_id, never from the client-supplied
    content["client-metadata"]["client-id"]. A client claiming someone else's
    ID in its own metadata must not have its trust history recorded under that
    claimed ID."""
    strategy = build_strategy(tmp_path)
    benign = [{"w": np.full((3, 3), 1.0 + v, dtype=np.float32)} for v in (0.01, -0.01, 0.02)]
    replies = [make_reply(str(i), s) for i, s in enumerate(benign)]
    # This reply's content claims to be "victim", but its authenticated Flower
    # sender identity is "attacker".
    replies.append(make_reply("victim", benign[0], src_node_id="attacker"))

    strategy.aggregate_train(1, replies)

    assert "attacker" in strategy.detector.history
    assert "victim" not in strategy.detector.history


def test_insufficient_security_quorum_skips_round(tmp_path):
    """REVIEW FIX (P0/P1 - security quorum bypass): below min_security_quorum,
    the round must be refused (no aggregate produced) rather than falling
    through to the detector's own "too few clients -> accept all" path."""
    strategy = build_strategy(tmp_path, min_security_quorum=3)
    benign = [{"w": np.full((3, 3), 1.0 + v, dtype=np.float32)} for v in (0.01, -0.01)]
    replies = [make_reply(str(i), s) for i, s in enumerate(benign)]

    arrays, metrics = strategy.aggregate_train(1, replies)

    assert arrays is None
    assert int(metrics["ftil_insufficient_security_quorum"]) == 1
    assert int(metrics["ftil_accepted"]) == 0


def test_trust_evidence_is_persisted(tmp_path):
    strategy = build_strategy(tmp_path)
    benign = [{"w": np.full((3, 3), 1.0 + v, dtype=np.float32)} for v in (0.01, -0.01, 0.02)]
    replies = [make_reply(str(i), s) for i, s in enumerate(benign)]

    strategy.aggregate_train(1, replies)

    log_path = tmp_path / "trust_log.jsonl"
    assert log_path.exists()
    assert '"round": 1' in log_path.read_text(encoding="utf-8")
