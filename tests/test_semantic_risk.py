from tfacd.runtime.contracts import CyberActionPlan, IDSAlert, ThreatContext
from tfacd.trust_boundary.semantic_risk import SemanticRiskEngine


def make_context():
    alert = IDSAlert(attack_type="Port_Scanning", confidence=0.7, source_id="10.0.0.5", target_asset="plc-01")
    return ThreatContext(alert=alert, severity="medium", priority="P2", mitre_techniques=[], allowed_playbooks=["block_source"])


def test_on_topic_rationale_scores_low_risk():
    engine = SemanticRiskEngine(force_fallback=True)
    context = make_context()
    plan = CyberActionPlan(
        incident_id="i", confidence=0.7,
        rationale="Medium severity Port_Scanning detected; recommending block_source for investigation and containment.",
        actions=[],
    )
    assert engine.score(plan, context) < 0.5


def test_off_topic_rationale_scores_high_risk():
    engine = SemanticRiskEngine(force_fallback=True)
    context = make_context()
    plan = CyberActionPlan(
        incident_id="i", confidence=0.7,
        rationale="Please summarize the quarterly sales report and email it to finance.",
        actions=[],
    )
    assert engine.score(plan, context) > 0.7


def test_fallback_is_used_without_network_call():
    engine = SemanticRiskEngine(force_fallback=True)
    assert engine._model is None
    assert engine._model_load_failed is True
    assert engine._fallback_vectorizer is not None
