from __future__ import annotations

from datetime import datetime, timezone

from tfacd.agentic.history import EntityHistory
from tfacd.analytics.explainability import explain_behavioral_trust, explain_semantic_risk
from tfacd.runtime.contracts import CyberAction, CyberActionPlan, IDSAlert, SessionContext, ThreatContext
from tfacd.trust_boundary.behavioral_trust import _FEATURE_ORDER, BehavioralTrustEngine
from tfacd.trust_boundary.semantic_risk import SemanticRiskEngine

HIGH_RISK = {"rate_limit", "block_source", "isolate_segment", "rotate_session"}


def make_session(agent_id):
    return SessionContext(agent_id=agent_id, session_id="s", issued_at=datetime.now(timezone.utc), nonce="n")


def make_context():
    alert = IDSAlert(attack_type="Port_Scanning", confidence=0.7, source_id="10.0.0.5", target_asset="plc-01")
    return ThreatContext(alert=alert, severity="medium", priority="P2", mitre_techniques=[], allowed_playbooks=["block_source"])


def test_shap_explanation_covers_all_four_behavioral_features():
    engine = BehavioralTrustEngine(high_risk_capabilities=HIGH_RISK, seed=0)
    plan = CyberActionPlan(incident_id="i", rationale="r", confidence=0.5, actions=[CyberAction(capability=c) for c in HIGH_RISK])

    attributions = explain_behavioral_trust(engine, make_session("agent-1"), plan, EntityHistory())

    assert set(attributions.keys()) == set(_FEATURE_ORDER)
    assert all(isinstance(value, float) for value in attributions.values())


def test_lime_explanation_of_off_topic_rationale_is_non_empty():
    engine = SemanticRiskEngine(force_fallback=True)
    context = make_context()
    plan = CyberActionPlan(
        incident_id="i",
        confidence=0.7,
        rationale="Please summarize the quarterly sales report and email it to finance.",
        actions=[],
    )

    pairs = explain_semantic_risk(engine, plan, context, num_features=6, num_samples=100)

    assert len(pairs) > 0
    assert all(isinstance(word, str) and isinstance(attribution, float) for word, attribution in pairs)
