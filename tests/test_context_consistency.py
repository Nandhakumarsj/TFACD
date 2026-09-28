from tfacd.runtime.contracts import CyberAction, CyberActionPlan, IDSAlert, ThreatContext
from tfacd.trust_boundary.context_consistency import score


def make_context(target_asset="plc-01", source_id="10.0.0.5", confidence=0.8):
    alert = IDSAlert(
        attack_type="Port_Scanning",
        confidence=confidence,
        source_id=source_id,
        target_asset=target_asset,
    )
    return ThreatContext(alert=alert, severity="medium", priority="P2", mitre_techniques=[], allowed_playbooks=["block_source"])


def test_correct_target_scores_full_marks():
    context = make_context()
    plan = CyberActionPlan(
        incident_id="i", rationale="r",
        actions=[CyberAction(capability="block_source", target="10.0.0.5")], confidence=0.8,
    )
    assert score(plan, context) == 1.0


def test_mismatched_source_target_scores_lower():
    context = make_context()
    plan = CyberActionPlan(
        incident_id="i", rationale="r",
        actions=[CyberAction(capability="block_source", target="plc-01")], confidence=0.8,
    )
    assert score(plan, context) < 1.0


def test_confidence_no_longer_affects_context_consistency_score():
    """REVIEW FIX (Section 20): confidence-gap reward was removed - a plan
    with a correctly-targeted action must score 1.0 regardless of how far its
    confidence is from the detector's, since scoring that gap rewarded simply
    copying the detector's confidence."""
    context = make_context(confidence=0.9)
    plan = CyberActionPlan(
        incident_id="i", rationale="r",
        actions=[CyberAction(capability="block_source", target="10.0.0.5")], confidence=0.1,
    )
    assert score(plan, context) == 1.0


def test_missing_target_on_high_risk_capability_scores_zero():
    """REVIEW FIX (Section 19): target is mandatory for block_source/rate_limit/
    isolate_segment - a missing target must NOT receive the same credit a None
    target gets for capabilities with no expected-target mapping."""
    context = make_context()
    plan = CyberActionPlan(
        incident_id="i", rationale="r",
        actions=[CyberAction(capability="block_source", target=None)], confidence=0.8,
    )
    assert score(plan, context) == 0.0


def test_missing_target_on_capability_without_target_mapping_still_scores_full():
    """A capability with no `_expected_target` mapping (e.g. "observe") never
    required a target in the first place, so a None target there is still a
    full match - only the specific high-risk, target-bearing capabilities are
    made mandatory."""
    context = make_context()
    plan = CyberActionPlan(
        incident_id="i", rationale="r",
        actions=[CyberAction(capability="observe", target=None)], confidence=0.8,
    )
    assert score(plan, context) == 1.0
