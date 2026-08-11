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
