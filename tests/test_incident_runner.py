from tfacd.agentic.decision_engine import AgenticDecisionEngine
from tfacd.agentic.history import EntityHistory
from tfacd.common.config import load_config
from tfacd.runtime.contracts import IDSAlert
from tfacd.runtime.threat_context import ThreatContextGenerator
from tfacd.streaming.incident_runner import run_incident
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


def build_pipeline(tmp_path):
    history = EntityHistory()
    threat_context_generator = ThreatContextGenerator("configs/threat_context.yaml")
    decision_engine = AgenticDecisionEngine(history=history)
    boundary = AdaptiveSemanticTrustBoundary(
        history=history,
        policy=POLICY,
        preprocessing_config=PREPROCESSING_CONFIG,
        trust_regulator=DynamicTrustScoreRegulator(0.4, 0.3, 0.3, THRESHOLDS),
        semantic_risk_engine=SemanticRiskEngine(force_fallback=True),
        behavioral_trust_engine=BehavioralTrustEngine(high_risk_capabilities=HIGH_RISK, seed=0),
        audit_logger=AuditLogger(tmp_path / "audit.jsonl"),
    )
    return threat_context_generator, decision_engine, boundary


def test_run_incident_produces_a_finalized_decision_through_the_real_pipeline(tmp_path):
    threat_context_generator, decision_engine, boundary = build_pipeline(tmp_path)
    alert = IDSAlert(attack_type="Port_Scanning", confidence=0.8, source_id="10.0.0.5", target_asset="plc-01")

    context, decision = run_incident(
        alert, threat_context_generator=threat_context_generator, decision_engine=decision_engine, boundary=boundary, agent_id="scenario-agent",
    )

    assert context.alert.attack_type == "Port_Scanning"
    assert decision.incident_id  # a real plan/decision was produced, not a stub
    assert decision.terminal_stage in ("preprocessing", "deterministic_controls", "capability_enforcement")


def test_run_incident_uses_a_fresh_nonce_each_call_so_repeated_incidents_are_never_replay_rejected(tmp_path):
    threat_context_generator, decision_engine, boundary = build_pipeline(tmp_path)
    alert = IDSAlert(attack_type="Port_Scanning", confidence=0.8, source_id="10.0.0.5", target_asset="plc-01")

    _, first = run_incident(alert, threat_context_generator=threat_context_generator, decision_engine=decision_engine, boundary=boundary, agent_id="scenario-agent")
    _, second = run_incident(alert, threat_context_generator=threat_context_generator, decision_engine=decision_engine, boundary=boundary, agent_id="scenario-agent")

    assert not any("nonce replay" in reason for result in (first, second) for stage in result.stage_results for reason in stage.reasons)


def test_run_incident_records_to_the_shared_audit_log(tmp_path):
    threat_context_generator, decision_engine, boundary = build_pipeline(tmp_path)
    alert = IDSAlert(attack_type="Port_Scanning", confidence=0.8, source_id="10.0.0.5", target_asset="plc-01")

    run_incident(alert, threat_context_generator=threat_context_generator, decision_engine=decision_engine, boundary=boundary, agent_id="scenario-agent")

    audit_lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(audit_lines) == 1
