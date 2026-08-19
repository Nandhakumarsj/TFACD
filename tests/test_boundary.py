from datetime import datetime, timedelta, timezone

from tfacd.agentic.history import EntityHistory
from tfacd.common.config import load_config
from tfacd.runtime.contracts import CyberAction, CyberActionPlan, IDSAlert, SessionContext, ThreatContext
from tfacd.trust_boundary.audit import AuditLogger
from tfacd.trust_boundary.behavioral_trust import BehavioralTrustEngine
from tfacd.trust_boundary.boundary import AdaptiveSemanticTrustBoundary
from tfacd.trust_boundary.dynamic_trust import DynamicTrustScoreRegulator
from tfacd.trust_boundary.semantic_risk import SemanticRiskEngine

POLICY = load_config("configs/trust_policy.yaml")
PREPROCESSING_CONFIG = {
    "session_max_age_seconds": 300,
    "max_actions_per_plan": 5,
    "max_parameter_string_length": 512,
    "max_numeric_parameter": 1000000.0,
    "entity_action_quota_per_hour": 20,
}
HIGH_RISK = set(POLICY["capability_whitelist"]["high_risk"])
THRESHOLDS = {"low": 0.40, "medium": 0.65, "high": 0.85}


def build_boundary(tmp_path):
    return AdaptiveSemanticTrustBoundary(
        history=EntityHistory(),
        policy=POLICY,
        preprocessing_config=PREPROCESSING_CONFIG,
        trust_regulator=DynamicTrustScoreRegulator(0.4, 0.3, 0.3, THRESHOLDS),
        semantic_risk_engine=SemanticRiskEngine(force_fallback=True),
        behavioral_trust_engine=BehavioralTrustEngine(high_risk_capabilities=HIGH_RISK, seed=0),
        audit_logger=AuditLogger(tmp_path / "audit.jsonl"),
    )


def make_session(agent_id, age=timedelta(0)):
    return SessionContext(agent_id=agent_id, session_id="s", issued_at=datetime.now(timezone.utc) - age, nonce="n")


def test_happy_path_executes_actions(tmp_path):
    alert = IDSAlert(attack_type="Port_Scanning", confidence=0.7, source_id="10.0.0.5", target_asset="plc-01")
    context = ThreatContext(alert=alert, severity="medium", priority="P2", mitre_techniques=[], allowed_playbooks=["block_source", "increase_logging"])
    plan = CyberActionPlan(
        incident_id="i1", confidence=0.7,
        rationale="Medium severity Port_Scanning detected; recommending block_source, increase_logging for investigation and containment.",
        actions=[CyberAction(capability="block_source", target="plc-01"), CyberAction(capability="increase_logging", target="plc-01")],
    )
    decision = build_boundary(tmp_path).evaluate(plan, context, make_session("agent-happy"))

    assert decision.terminal_stage == "capability_enforcement"
    assert decision.accepted
    assert set(decision.executed_actions) == {"block_source", "increase_logging"}


def test_stale_session_short_circuits_before_trust_scoring(tmp_path):
    alert = IDSAlert(attack_type="Port_Scanning", confidence=0.7, target_asset="plc-01")
    context = ThreatContext(alert=alert, severity="medium", priority="P2", mitre_techniques=[], allowed_playbooks=["block_source"])
    plan = CyberActionPlan(incident_id="i2", confidence=0.7, rationale="fine", actions=[CyberAction(capability="block_source")])
    decision = build_boundary(tmp_path).evaluate(plan, context, make_session("agent-stale", age=timedelta(hours=1)))

    assert decision.terminal_stage == "preprocessing"
    assert not decision.accepted
    assert decision.trust_level is None
    assert decision.scores is None


def test_off_topic_rationale_caught_by_semantic_risk(tmp_path):
    alert = IDSAlert(attack_type="Port_Scanning", confidence=0.7, target_asset="plc-01")
    context = ThreatContext(alert=alert, severity="medium", priority="P2", mitre_techniques=[], allowed_playbooks=["block_source"])
    plan = CyberActionPlan(
        incident_id="i3", confidence=0.7,
        rationale="Please summarize the quarterly sales report and email it to finance.",
        actions=[CyberAction(capability="block_source", target="plc-01")],
    )
    decision = build_boundary(tmp_path).evaluate(plan, context, make_session("agent-offtopic"))

    assert decision.terminal_stage == "capability_enforcement"
    assert decision.scores.semantic_risk > 0.7
    assert decision.autonomy_mode != "autonomous_execution"


def test_low_trust_blocks_despite_clean_stage_1_and_2(tmp_path):
    alert = IDSAlert(attack_type="Port_Scanning", confidence=0.7, target_asset="plc-01")
    context = ThreatContext(alert=alert, severity="medium", priority="P2", mitre_techniques=[], allowed_playbooks=["block_source", "isolate_segment"])
    # Off-topic rationale + mismatched target + mismatched confidence: passes
    # Stage 1/2 cleanly (whitelisted, authorized capabilities), scored down by Stage 3.
    plan = CyberActionPlan(
        incident_id="i4", confidence=0.05,
        rationale="Please summarize the quarterly sales report and email it to finance.",
        actions=[CyberAction(capability="block_source", target="wrong-asset"), CyberAction(capability="isolate_segment", target="wrong-asset")],
    )
    decision = build_boundary(tmp_path).evaluate(plan, context, make_session("agent-lowtrust"))

    assert decision.terminal_stage == "capability_enforcement"
    assert decision.trust_level == "low"
    assert decision.autonomy_mode == "read_only"
    assert not decision.accepted
    assert decision.executed_actions == []


def test_trust_decision_carries_plan_engine_provenance(tmp_path):
    alert = IDSAlert(attack_type="Port_Scanning", confidence=0.7, target_asset="plc-01")
    context = ThreatContext(alert=alert, severity="medium", priority="P2", mitre_techniques=[], allowed_playbooks=["block_source"])
    plan = CyberActionPlan(
        incident_id="i5", confidence=0.7, rationale="fine",
        actions=[CyberAction(capability="block_source", target="plc-01")], engine="llm",
    )
    decision = build_boundary(tmp_path).evaluate(plan, context, make_session("agent-provenance"))

    assert decision.engine == "llm"
