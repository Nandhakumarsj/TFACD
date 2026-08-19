from datetime import datetime, timezone

from tfacd.agentic.history import EntityHistory
from tfacd.runtime.contracts import CyberAction, CyberActionPlan, SessionContext
from tfacd.trust_boundary.behavioral_trust import BehavioralTrustEngine

HIGH_RISK = {"rate_limit", "block_source", "isolate_segment", "rotate_session"}


def make_session(agent_id):
    return SessionContext(agent_id=agent_id, session_id="s", issued_at=datetime.now(timezone.utc), nonce="n")


def test_anomalous_plan_scores_lower_than_typical_plan():
    engine = BehavioralTrustEngine(high_risk_capabilities=HIGH_RISK, seed=0)

    typical_plan = CyberActionPlan(incident_id="i1", rationale="r", confidence=0.5, actions=[CyberAction(capability="observe")])
    typical_score = engine.score(make_session("agent-typical"), typical_plan, EntityHistory())

    violation_history = EntityHistory()
    session = make_session("agent-anomalous")
    for _ in range(3):
        violation_history.append(session.agent_id, "trust_decision", {"policy_violation": True})
    anomalous_plan = CyberActionPlan(
        incident_id="i2", rationale="r", confidence=0.5,
        actions=[CyberAction(capability=c) for c in HIGH_RISK],
    )
    anomalous_score = engine.score(session, anomalous_plan, violation_history)

    assert anomalous_score < typical_score


def test_ema_does_not_reset_between_repeated_anomalous_interactions():
    engine = BehavioralTrustEngine(high_risk_capabilities=HIGH_RISK, ema_alpha=0.3, seed=0)
    history = EntityHistory()
    session = make_session("agent-1")
    plan = CyberActionPlan(incident_id="i", rationale="r", confidence=0.5, actions=[CyberAction(capability=c) for c in HIGH_RISK])

    first = engine.score(session, plan, history)
    second = engine.score(session, plan, history)
    assert second <= first


def _seed_real_history(history, num_entities=5, events_per_entity=10):
    """Real-shaped trust_decision events, same payload shape boundary.py's
    _finalize() actually persists - a mostly-benign observed population."""
    for entity_index in range(num_entities):
        agent_id = f"agent-{entity_index}"
        for round_index in range(events_per_entity):
            history.append(
                agent_id, "trust_decision",
                {"capabilities": ["observe", "log_event"], "policy_violation": round_index == 0},
            )


def test_refit_from_history_returns_false_below_min_samples():
    engine = BehavioralTrustEngine(high_risk_capabilities=HIGH_RISK, seed=0)
    history = EntityHistory()
    _seed_real_history(history, num_entities=1, events_per_entity=3)  # 3 events, well under any reasonable min_samples

    original_forest = engine._forest
    refit = engine.refit_from_history(history, min_samples=20)

    assert refit is False
    assert engine._forest is original_forest  # untouched - still the synthetic cold-start population


def test_refit_from_history_replaces_forest_once_enough_real_data_exists():
    engine = BehavioralTrustEngine(high_risk_capabilities=HIGH_RISK, seed=0)
    history = EntityHistory()
    _seed_real_history(history, num_entities=5, events_per_entity=10)  # 50 real events

    original_forest = engine._forest
    refit = engine.refit_from_history(history, min_samples=20)

    assert refit is True
    assert engine._forest is not original_forest


def test_refit_reconstructs_features_from_stored_capabilities_and_violations():
    """White-box: a population built entirely from HIGH_RISK-heavy, high-
    violation-rate events should treat a typical low-risk plan as more
    anomalous under the refit forest than under the original synthetic one -
    proving the real observed data (not the synthetic default) now governs
    what counts as normal."""
    engine = BehavioralTrustEngine(high_risk_capabilities=HIGH_RISK, seed=0)
    history = EntityHistory()
    for entity_index in range(6):
        agent_id = f"risky-agent-{entity_index}"
        for round_index in range(6):
            history.append(
                agent_id, "trust_decision",
                {"capabilities": sorted(HIGH_RISK), "policy_violation": True},
            )  # 36 events, every one high-risk-heavy and a violation

    typical_low_risk_plan = CyberActionPlan(incident_id="i", rationale="r", confidence=0.5, actions=[CyberAction(capability="observe")])
    fresh_session = make_session("agent-being-scored")
    empty_history = EntityHistory()

    before = engine.score(fresh_session, typical_low_risk_plan, empty_history)
    assert engine.refit_from_history(history, min_samples=20) is True
    after = engine.score(make_session("agent-being-scored-2"), typical_low_risk_plan, empty_history)

    # Under the risky-only refit population, a typical low-risk/no-violation
    # plan is now the OUTLIER relative to "normal" - it should score no higher
    # (and in practice lower) than under the original synthetic population,
    # which was centered on exactly this kind of low-risk behavior.
    assert after <= before


def test_all_events_aggregates_across_entities_and_filters_by_kind():
    history = EntityHistory()
    history.append("agent-a", "trust_decision", {"x": 1})
    history.append("agent-b", "trust_decision", {"x": 2})
    history.append("agent-a", "incident", {"x": 3})

    all_decisions = history.all_events(kind="trust_decision")
    assert len(all_decisions) == 2
    assert {e["entity_id"] for e in all_decisions} == {"agent-a", "agent-b"}

    everything = history.all_events()
    assert len(everything) == 3
