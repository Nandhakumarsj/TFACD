from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from tfacd.agentic.decision_engine import AgenticDecisionEngine
from tfacd.agentic.history import EntityHistory
from tfacd.agentic.synthetic import simulate_agent_population
from tfacd.common.config import load_config
from tfacd.runtime.contracts import CyberAction, CyberActionPlan, IDSAlert, SessionContext, ThreatContext
from tfacd.runtime.threat_context import ThreatContextGenerator
from tfacd.trust_boundary.audit import AuditLogger
from tfacd.trust_boundary.behavioral_trust import BehavioralTrustEngine
from tfacd.trust_boundary.boundary import AdaptiveSemanticTrustBoundary
from tfacd.trust_boundary.dynamic_trust import DynamicTrustScoreRegulator
from tfacd.trust_boundary.semantic_risk import SemanticRiskEngine

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/edge_iiot.yaml")
args = parser.parse_args()

config = load_config(args.config)
tb_config = config["trust_boundary"]
policy = load_config(config["runtime"]["trust_policy_path"])

history = EntityHistory(persist_path=Path("artifacts/agentic/history.jsonl"))
threat_context_generator = ThreatContextGenerator(config["runtime"]["threat_context_mapping"])
decision_engine = AgenticDecisionEngine(history=history)
boundary = AdaptiveSemanticTrustBoundary(
    history=history,
    policy=policy,
    preprocessing_config=tb_config,
    trust_regulator=DynamicTrustScoreRegulator(
        tb_config["weight_semantic_risk"], tb_config["weight_context_consistency"], tb_config["weight_behavioral_trust"], tb_config["trust_level_thresholds"]
    ),
    semantic_risk_engine=SemanticRiskEngine(model_name=tb_config["sbert_model_name"]),
    behavioral_trust_engine=BehavioralTrustEngine(high_risk_capabilities=set(policy["capability_whitelist"]["high_risk"]), ema_alpha=tb_config["ema_alpha"]),
    audit_logger=AuditLogger(Path("artifacts/trust_boundary/audit_log.jsonl")),
)
session = SessionContext(agent_id="decision_engine_v1", session_id="demo-session", issued_at=datetime.now(timezone.utc), nonce="demo-nonce")


def show(name: str, plan: CyberActionPlan, context: ThreatContext) -> None:
    decision = boundary.evaluate(plan, context, session)
    print(f"\n=== {name} ===")
    print(f"attack_type={context.alert.attack_type} severity={context.severity} priority={context.priority}")
    print(f"rationale: {plan.rationale}")
    print(f"terminal_stage={decision.terminal_stage} accepted={decision.accepted} trust_level={decision.trust_level} autonomy_mode={decision.autonomy_mode}")
    if decision.scores:
        s = decision.scores
        print(f"scores: Rs={s.semantic_risk:.3f} Rc={s.context_consistency:.3f} Rb={s.behavioral_trust:.3f} T={s.trust_value:.3f}")
    else:
        print(f"rejection reasons: {[r for sr in decision.stage_results for r in sr.reasons]}")
    print(f"executed_actions={decision.executed_actions}")


# 1. Clean, low-severity alert - expect the decision engine's own well-formed
# plan to score high and execute the (low-risk) allowed playbooks.
alert1 = IDSAlert(attack_type="Normal", confidence=0.95, source_id="192.168.0.10", target_asset="sensor-04", protocol="MQTT")
context1 = threat_context_generator.enrich(alert1)
show("clean low-severity", decision_engine.decide(alert1, context1), context1)

# 2. Critical alert with a well-formed rationale - expect at least the
# low-risk playbooks to execute, possibly the high-risk ones too if trust is high.
alert2 = IDSAlert(attack_type="DDoS_HTTP_Flood", confidence=0.9, source_id="203.0.113.7", target_asset="gateway-01", protocol="HTTP")
context2 = threat_context_generator.enrich(alert2)
show("critical, well-formed", decision_engine.decide(alert2, context2), context2)

# 3. Deliberately malformed plan (off-topic rationale, mismatched target and
# confidence) bypassing the decision engine - expect a Stage 3 Low-trust block.
alert3 = IDSAlert(attack_type="MITM", confidence=0.6, source_id="198.51.100.9", target_asset="plc-02")
context3 = threat_context_generator.enrich(alert3)
bad_plan = CyberActionPlan(
    incident_id="demo-bad-1", confidence=0.05,
    rationale="Please summarize the quarterly sales report and email it to finance.",
    actions=[CyberAction(capability=p, target="wrong-asset") for p in context3.allowed_playbooks],
)
show("deliberately off-topic plan", bad_plan, context3)

# 4. A synthetic multi-agent population (4 archetypes x 5 rounds each) so the
# audit log has real cross-agent history for Phase II analytics to read.
population_results = simulate_agent_population(boundary, context3, rounds_per_agent=5)
print("\n=== synthetic agent population ===")
for agent_id, decisions in population_results.items():
    trajectory = [f"{d.scores.trust_value:.2f}" if d.scores else "blocked" for d in decisions]
    print(f"{agent_id}: trust trajectory = {trajectory}")

print("\naudit log: artifacts/trust_boundary/audit_log.jsonl")
print("history: artifacts/agentic/history.jsonl")
