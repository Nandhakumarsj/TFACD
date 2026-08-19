from datetime import datetime, timezone
from pathlib import Path

import pytest

from tfacd.analytics.kpi import DEFAULT_AUDIT_LOG, compute_kpis, overall_acceptance_rate, per_agent_summary, top_bottom_agents, trust_level_distribution
from tfacd.runtime.contracts import AuditEntry, TrustDecision, TrustScores

AUDIT_LOG_PATH = Path(__file__).resolve().parents[1] / DEFAULT_AUDIT_LOG


def make_entry(sequence, agent_id, accepted, trust_level=None, trust_value=None, terminal_stage=None):
    scores = TrustScores(semantic_risk=0.1, context_consistency=0.9, behavioral_trust=0.8, trust_value=trust_value) if trust_value is not None else None
    decision = TrustDecision(
        incident_id=f"inc-{sequence}",
        accepted=accepted,
        terminal_stage=terminal_stage or ("capability_enforcement" if scores is not None else "deterministic_controls"),
        trust_level=trust_level,
        scores=scores,
        rationale="r",
    )
    return AuditEntry(
        sequence=sequence,
        timestamp=datetime.now(timezone.utc),
        incident_id=decision.incident_id,
        agent_id=agent_id,
        entry_hash="h" * 64,
        previous_hash="0" * 64,
        decision=decision,
    )


def test_acceptance_rate_and_trust_distribution_on_mixed_fixture():
    entries = [
        make_entry(1, "agent-a", True, "verified", 0.9),
        make_entry(2, "agent-a", True, "high", 0.7),
        make_entry(3, "agent-a", False, "low", 0.2),
        make_entry(4, "agent-b", False),  # hard reject at deterministic_controls - scores None
        make_entry(5, "agent-b", True, "medium", 0.5),
        make_entry(6, "agent-c", True, "verified", 0.95),
        make_entry(7, "agent-c", True, "verified", 0.85),
    ]

    assert overall_acceptance_rate(entries) == pytest.approx(5 / 7)

    dist = trust_level_distribution(entries)
    assert dist == {"low": 1, "medium": 1, "high": 1, "verified": 3, "hard_rejected": 1}


def test_per_agent_summary_skips_null_agent_and_computes_mean_over_scored_only():
    entries = [
        make_entry(1, "agent-a", True, "verified", 0.9),
        make_entry(2, "agent-a", True, "high", 0.7),
        make_entry(3, "agent-a", False, "low", 0.2),
        make_entry(4, "agent-b", False),  # hard reject - excluded from agent-b's mean, counted in num_interactions
        make_entry(5, "agent-b", True, "medium", 0.5),
        make_entry(6, None, True, "verified", 0.99),  # pre-multi-agent-fixture entry - no agent to attribute to
    ]

    summaries = {s.agent_id: s for s in per_agent_summary(entries)}

    assert set(summaries) == {"agent-a", "agent-b"}  # agent_id=None entry dropped

    assert summaries["agent-a"].num_interactions == 3
    assert summaries["agent-a"].num_scored == 3
    assert summaries["agent-a"].acceptance_rate == pytest.approx(2 / 3)
    assert summaries["agent-a"].mean_trust_value == pytest.approx(0.6)

    assert summaries["agent-b"].num_interactions == 2
    assert summaries["agent-b"].num_scored == 1  # only the scored entry counts
    assert summaries["agent-b"].acceptance_rate == pytest.approx(0.5)
    assert summaries["agent-b"].mean_trust_value == pytest.approx(0.5)


def test_top_bottom_agents_skips_low_signal_and_orders_correctly():
    entries = []
    seq = 1
    for agent_id, value in [("agent-1", 0.9), ("agent-2", 0.8), ("agent-3", 0.7), ("agent-4", 0.6), ("agent-5", 0.5)]:
        entries.append(make_entry(seq, agent_id, True, "high", value)); seq += 1
        entries.append(make_entry(seq, agent_id, True, "high", value)); seq += 1
    entries.append(make_entry(seq, "agent-lonely", True, "high", 0.99))  # single scored entry - not enough signal

    summaries = per_agent_summary(entries)
    top, bottom = top_bottom_agents(summaries, n=3, min_scored=2)

    assert [s.agent_id for s in top] == ["agent-1", "agent-2", "agent-3"]
    assert [s.agent_id for s in bottom] == ["agent-5", "agent-4", "agent-3"]
    assert "agent-lonely" not in [s.agent_id for s in top + bottom]


def test_real_audit_log_computes_kpis_without_error():
    report = compute_kpis(AUDIT_LOG_PATH)

    assert report.num_entries > 0
    assert report.per_agent  # non-empty per-agent summary
    assert 0.0 <= report.overall_acceptance_rate <= 1.0
    assert sum(report.trust_level_distribution.values()) == report.num_entries
    assert "agent-risky" in [s.agent_id for s in report.per_agent]
