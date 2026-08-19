from tfacd.runtime.contracts import CyberAction, CyberActionPlan, IDSAlert, ThreatContext
from tfacd.trust_boundary.context_consistency import score


def make_context(target_asset="plc-01", confidence=0.8):
    alert = IDSAlert(attack_type="Port_Scanning", confidence=confidence, target_asset=target_asset)
    return ThreatContext(alert=alert, severity="medium", priority="P2", mitre_techniques=[], allowed_playbooks=["block_source"])


def test_matching_target_and_confidence_scores_high():
    context = make_context()
    plan = CyberActionPlan(
        incident_id="i", rationale="r",
        actions=[CyberAction(capability="block_source", target="plc-01")], confidence=0.8,
    )
    assert score(plan, context) == 1.0


def test_mismatched_target_scores_lower():
    context = make_context()
    plan = CyberActionPlan(
        incident_id="i", rationale="r",
        actions=[CyberAction(capability="block_source", target="some-other-asset")], confidence=0.8,
    )
    assert score(plan, context) < 1.0


def test_mismatched_confidence_scores_lower():
    context = make_context(confidence=0.9)
    plan = CyberActionPlan(
        incident_id="i", rationale="r",
        actions=[CyberAction(capability="block_source", target="plc-01")], confidence=0.1,
    )
    assert score(plan, context) <= 0.6
