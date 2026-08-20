from datetime import datetime, timezone

from tfacd.analytics.threshold_validation import compute_report
from tfacd.analytics.trust_labels import AnalystLabel, AnalystLabelStore
from tfacd.runtime.contracts import AuditEntry, TrustDecision, TrustScores


def make_entry(sequence, trust_level="high", executor_mode="simulate", executed_actions=None):
    decision = TrustDecision(
        incident_id=f"inc-{sequence}",
        accepted=True,
        terminal_stage="capability_enforcement",
        trust_level=trust_level,
        scores=TrustScores(semantic_risk=0.1, context_consistency=0.9, behavioral_trust=0.8, trust_value=0.7),
        executed_actions=executed_actions or [],
        rationale="r",
        executor_mode=executor_mode,
    )
    return AuditEntry(
        sequence=sequence,
        timestamp=datetime.now(timezone.utc),
        incident_id=decision.incident_id,
        agent_id="agent-a",
        entry_hash="h" * 64,
        previous_hash="0" * 64,
        decision=decision,
    )


def write_audit_log(path, entries):
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(entry.model_dump_json() + "\n")


def test_not_ready_below_min_samples(tmp_path):
    audit_log = tmp_path / "audit.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    write_audit_log(audit_log, [make_entry(1), make_entry(2)])

    store = AnalystLabelStore(labels_path)
    store.append(AnalystLabel(audit_sequence=1, label="correct", analyst_id="alice", rationale="r"))
    store.append(AnalystLabel(audit_sequence=2, label="correct", analyst_id="alice", rationale="r"))

    report = compute_report(audit_log, labels_path, min_samples=20)
    assert not report.ready
    assert report.num_labeled_decisions == 2
    assert report.per_trust_level == []


def test_ready_and_per_level_breakdown_once_min_samples_reached(tmp_path):
    audit_log = tmp_path / "audit.jsonl"
    labels_path = tmp_path / "labels.jsonl"

    entries = [make_entry(i, trust_level="high" if i % 2 == 0 else "medium") for i in range(1, 21)]
    write_audit_log(audit_log, entries)

    store = AnalystLabelStore(labels_path)
    for i in range(1, 21):
        label = "correct" if i <= 15 else "false_positive"
        store.append(AnalystLabel(audit_sequence=i, label=label, analyst_id="alice", rationale="r"))

    report = compute_report(audit_log, labels_path, min_samples=20)
    assert report.ready
    assert report.num_labeled_decisions == 20

    by_level = {row.trust_level: row for row in report.per_trust_level}
    assert set(by_level) == {"high", "medium"}
    assert by_level["high"].num_labels + by_level["medium"].num_labels == 20
    assert 0.0 <= by_level["high"].agreement_rate <= 1.0


def test_unmatched_label_is_flagged_not_dropped_silently(tmp_path):
    audit_log = tmp_path / "audit.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    write_audit_log(audit_log, [make_entry(1)])

    store = AnalystLabelStore(labels_path)
    store.append(AnalystLabel(audit_sequence=999, label="correct", analyst_id="alice", rationale="r"))

    report = compute_report(audit_log, labels_path, min_samples=1)
    assert report.unmatched_labels == [999]
    assert report.num_labeled_decisions == 0


def test_production_executor_labels_are_surfaced_regardless_of_readiness(tmp_path):
    audit_log = tmp_path / "audit.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    write_audit_log(
        audit_log,
        [make_entry(1, executor_mode="production", executed_actions=["block_source"])],
    )

    store = AnalystLabelStore(labels_path)
    store.append(AnalystLabel(audit_sequence=1, label="false_positive", analyst_id="alice", rationale="should not have blocked"))

    report = compute_report(audit_log, labels_path, min_samples=20)
    assert not report.ready  # far below min_samples
    assert len(report.production_action_reviews) == 1
    assert report.production_action_reviews[0].executed_actions == ["block_source"]
    assert report.production_action_reviews[0].label == "false_positive"
