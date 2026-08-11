from datetime import datetime, timezone
from pathlib import Path

from tfacd.runtime.contracts import AuditEntry, TrustDecision, TrustScores
from tfacd.analytics.reputation import DEFAULT_AUDIT_LOG, CrossAgentReputationEngine, load_entries, rank_agents

AUDIT_LOG_PATH = Path(__file__).resolve().parents[1] / DEFAULT_AUDIT_LOG


def make_entry(sequence, agent_id, accepted, trust_value=None):
    scores = TrustScores(semantic_risk=0.1, context_consistency=0.9, behavioral_trust=0.8, trust_value=trust_value) if trust_value is not None else None
    decision = TrustDecision(
        incident_id=f"inc-{sequence}",
        accepted=accepted,
        terminal_stage="capability_enforcement" if scores is not None else "deterministic_controls",
        trust_level="high" if accepted else "low",
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


def test_clean_agent_outranks_violation_heavy_agent():
    entries = [
        make_entry(1, "agent-clean", accepted=True, trust_value=0.95),
        make_entry(2, "agent-clean", accepted=True, trust_value=0.9),
        make_entry(3, "agent-clean", accepted=True, trust_value=0.92),
        make_entry(4, "agent-messy", accepted=False, trust_value=0.4),
        make_entry(5, "agent-messy", accepted=False),  # hard reject, scores None
        make_entry(6, "agent-messy", accepted=False, trust_value=0.3),
    ]
    engine = CrossAgentReputationEngine()
    engine.update(entries)
    ranking = engine.rank()

    assert [r.agent_id for r in ranking] == ["agent-clean", "agent-messy"]
    assert ranking[0].reputation_score > ranking[1].reputation_score
    assert ranking[1].num_violations == 3


def test_null_agent_id_entries_are_skipped():
    entries = [make_entry(1, None, accepted=True, trust_value=0.9)]
    engine = CrossAgentReputationEngine()
    engine.update(entries)
    assert engine.rank() == []


def test_incremental_update_matches_single_batch_update():
    entries = [
        make_entry(1, "agent-a", accepted=True, trust_value=0.9),
        make_entry(2, "agent-a", accepted=False, trust_value=0.3),
        make_entry(3, "agent-b", accepted=True, trust_value=0.7),
    ]

    batched = CrossAgentReputationEngine()
    batched.update(entries)

    streamed = CrossAgentReputationEngine()
    for entry in entries:
        streamed.update([entry])

    assert streamed.rank() == batched.rank()


def test_hard_and_soft_rejections_both_count_as_violations():
    entries = [
        make_entry(1, "agent-x", accepted=False),  # hard reject, scores None
        make_entry(2, "agent-x", accepted=False, trust_value=0.2),  # soft Stage-3 block
    ]
    engine = CrossAgentReputationEngine()
    engine.update(entries)
    ranking = engine.rank()
    assert ranking[0].num_violations == 2
    assert ranking[0].num_interactions == 2


def test_real_audit_log_ranks_well_behaved_or_improving_above_risky():
    entries = load_entries(AUDIT_LOG_PATH)
    engine = CrossAgentReputationEngine()
    engine.update(entries)
    ranking = engine.rank()
    scores = {r.agent_id: r.reputation_score for r in ranking}

    assert "agent-risky" in scores
    assert "agent-well_behaved" in scores or "agent-improving" in scores
    best_of_the_two = max(scores.get("agent-well_behaved", float("-inf")), scores.get("agent-improving", float("-inf")))
    assert best_of_the_two > scores["agent-risky"]


def test_rank_agents_convenience_reads_default_relative_path():
    # DEFAULT_AUDIT_LOG is relative to repo root; pytest's rootdir-based cwd
    # convention means this only works when run from the repo root, matching
    # how the other trust_boundary/audit tests treat artifact paths.
    ranking = rank_agents(AUDIT_LOG_PATH)
    assert len(ranking) > 0
    assert all(ranking[i].reputation_score >= ranking[i + 1].reputation_score for i in range(len(ranking) - 1))
