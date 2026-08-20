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


def test_naturally_phrased_on_topic_rationale_not_spuriously_penalized():
    """Regression for a measured, real finding: a genuinely on-topic but
    naturally-phrased (LLM-style) rationale - not the exact template text -
    previously scored Rs~0.485-0.686 under the TF-IDF fallback purely from
    vocabulary/word-choice mismatch, not real topical drift."""
    engine = SemanticRiskEngine(force_fallback=True)
    alert = IDSAlert(attack_type="DDoS_UDP", confidence=0.9, source_id="203.0.113.7", target_asset="gateway-01")
    context = ThreatContext(
        alert=alert, severity="critical", priority="P1", mitre_techniques=[],
        allowed_playbooks=["rate_limit", "block_source", "start_capture", "notify_soc"],
    )
    plan = CyberActionPlan(
        incident_id="i", confidence=0.9,
        rationale=(
            "Based on the detected DDoS_UDP activity originating from source 203.0.113.7 targeting the "
            "gateway-01 asset, immediate rate limiting and source blocking are recommended to contain the "
            "threat, alongside starting a packet capture and notifying the SOC team."
        ),
        actions=[],
    )
    assert engine.score(plan, context) < 0.5


def test_wrong_attack_type_named_is_not_floored():
    """The attack-type keyword floor must not rescue a rationale that names a
    DIFFERENT attack than the one actually detected - only the real match counts."""
    engine = SemanticRiskEngine(force_fallback=True)
    alert = IDSAlert(attack_type="DDoS_UDP", confidence=0.9, source_id="203.0.113.7", target_asset="gateway-01")
    context = ThreatContext(
        alert=alert, severity="critical", priority="P1", mitre_techniques=[],
        allowed_playbooks=["rate_limit", "block_source", "start_capture", "notify_soc"],
    )
    plan = CyberActionPlan(
        incident_id="i", confidence=0.9,
        rationale="Detected Ransomware activity; recommending isolate_segment, block_source, start_capture, notify_soc immediately.",
        actions=[],
    )
    assert engine.score(plan, context) > 0.5
