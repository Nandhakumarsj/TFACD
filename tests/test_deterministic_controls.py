from tfacd.common.config import load_config
from tfacd.runtime.contracts import CyberAction, CyberActionPlan, IDSAlert, ThreatContext
from tfacd.trust_boundary import deterministic_controls

POLICY = load_config("configs/trust_policy.yaml")


def make_context(allowed_playbooks):
    alert = IDSAlert(attack_type="Port_Scanning", confidence=0.7)
    return ThreatContext(alert=alert, severity="medium", priority="P2", mitre_techniques=[], allowed_playbooks=allowed_playbooks)


def make_plan(capability):
    return CyberActionPlan(
        incident_id="inc-1", rationale="test rationale",
        actions=[CyberAction(capability=capability)], confidence=0.8,
    )


def test_whitelisted_and_authorized_capability_accepted():
    context = make_context(["block_source"])
    result = deterministic_controls.run(make_plan("block_source"), context, POLICY)
    assert result.accepted


def test_capability_outside_whitelist_rejected():
    context = make_context(["nonexistent_capability"])
    result = deterministic_controls.run(make_plan("nonexistent_capability"), context, POLICY)
    assert not result.accepted
    assert any("whitelist" in r for r in result.reasons)


def test_capability_not_authorized_for_context_rejected():
    context = make_context(["observe"])  # block_source is whitelisted but not authorized here
    result = deterministic_controls.run(make_plan("block_source"), context, POLICY)
    assert not result.accepted
    assert any("not authorized" in r for r in result.reasons)


def test_empty_incident_id_rejected():
    context = make_context(["observe"])
    plan = CyberActionPlan(incident_id="  ", rationale="test", actions=[CyberAction(capability="observe")], confidence=0.5)
    result = deterministic_controls.run(plan, context, POLICY)
    assert not result.accepted
    assert any("incident_id" in r for r in result.reasons)
