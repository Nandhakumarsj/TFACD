from tfacd.agentic.history import EntityHistory
from tfacd.agentic.synthetic import simulate_agent_population
from tfacd.common.config import load_config
from tfacd.runtime.contracts import IDSAlert, ThreatContext
from tfacd.trust_boundary.audit import AuditLogger, verify_chain
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


def test_population_produces_distinct_agents_with_diverging_trust(tmp_path):
    boundary = AdaptiveSemanticTrustBoundary(
        history=EntityHistory(),
        policy=POLICY,
        preprocessing_config=PREPROCESSING_CONFIG,
        trust_regulator=DynamicTrustScoreRegulator(0.4, 0.3, 0.3, {"low": 0.40, "medium": 0.65, "high": 0.85}),
        semantic_risk_engine=SemanticRiskEngine(force_fallback=True),
        behavioral_trust_engine=BehavioralTrustEngine(high_risk_capabilities=HIGH_RISK, seed=0),
        audit_logger=AuditLogger(tmp_path / "audit.jsonl"),
    )
    alert = IDSAlert(attack_type="Port_Scanning", confidence=0.7, target_asset="plc-01")
    context = ThreatContext(alert=alert, severity="medium", priority="P2", mitre_techniques=[], allowed_playbooks=["block_source", "isolate_segment"])

    results = simulate_agent_population(boundary, context, rounds_per_agent=4)

    assert set(results) == {"agent-well_behaved", "agent-borderline", "agent-risky", "agent-improving"}
    for decisions in results.values():
        assert len(decisions) == 4

    well_behaved_final_trust = results["agent-well_behaved"][-1].scores.trust_value
    risky_final_trust = results["agent-risky"][-1].scores.trust_value
    assert well_behaved_final_trust > risky_final_trust

    ok, bad_sequence = verify_chain(tmp_path / "audit.jsonl")
    assert ok
    assert bad_sequence is None


def test_audit_entries_carry_agent_id(tmp_path):
    boundary = AdaptiveSemanticTrustBoundary(
        history=EntityHistory(),
        policy=POLICY,
        preprocessing_config=PREPROCESSING_CONFIG,
        trust_regulator=DynamicTrustScoreRegulator(0.4, 0.3, 0.3, {"low": 0.40, "medium": 0.65, "high": 0.85}),
        semantic_risk_engine=SemanticRiskEngine(force_fallback=True),
        behavioral_trust_engine=BehavioralTrustEngine(high_risk_capabilities=HIGH_RISK, seed=0),
        audit_logger=AuditLogger(tmp_path / "audit.jsonl"),
    )
    alert = IDSAlert(attack_type="Normal", confidence=0.9, target_asset="sensor-01")
    context = ThreatContext(alert=alert, severity="informational", priority="P4", mitre_techniques=[], allowed_playbooks=["observe"])
    simulate_agent_population(boundary, context, rounds_per_agent=1)

    import json

    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    agent_ids = {json.loads(line)["agent_id"] for line in lines}
    assert agent_ids == {"agent-well_behaved", "agent-borderline", "agent-risky", "agent-improving"}
