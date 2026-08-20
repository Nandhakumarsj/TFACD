"""CLI entry point for Explainable Trust Analytics (analytics/explainability.py,
SHAP/LIME) - previously tested but never invoked outside
tests/test_explainability.py, per an architecture audit.

Unlike the other newly-wired scripts, this one cannot replay a REAL historical
incident: TrustDecision/AuditEntry only retain the executed_actions list and
final scores, not the original CyberActionPlan's full actions/parameters or
the EntityHistory state at scoring time - there is no way to reconstruct an
exact past explanation from the audit log alone. Runs against a clearly-
synthetic, representative example instead (the same "off-topic plan" shape
scripts/run_trust_boundary_demo.py already uses), explaining WHY the trust
boundary would react as it does to a well-formed vs. an off-topic plan.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from tfacd.agentic.history import EntityHistory
from tfacd.analytics.explainability import explain_behavioral_trust, explain_semantic_risk
from tfacd.common.config import load_config
from tfacd.runtime.contracts import CyberAction, CyberActionPlan, IDSAlert, SessionContext, ThreatContext
from tfacd.trust_boundary.behavioral_trust import BehavioralTrustEngine
from tfacd.trust_boundary.semantic_risk import SemanticRiskEngine

config = load_config("configs/edge_iiot.yaml")
tb_config = config["trust_boundary"]
policy = load_config(config["runtime"]["trust_policy_path"])
high_risk = set(policy["capability_whitelist"]["high_risk"])

semantic_engine = SemanticRiskEngine(model_name=tb_config["sbert_model_name"])
behavioral_engine = BehavioralTrustEngine(high_risk_capabilities=high_risk, ema_alpha=tb_config["ema_alpha"])
history = EntityHistory()

alert = IDSAlert(attack_type="Ransomware", confidence=0.96, source_id="10.0.0.17", target_asset="DEVICE_17")
context = ThreatContext(alert=alert, severity="critical", priority="P1", mitre_techniques=[], allowed_playbooks=["isolate_segment", "block_source", "start_capture", "notify_soc"])
session = SessionContext(agent_id="explain-demo", session_id="s", issued_at=datetime.now(timezone.utc), nonce=uuid4().hex)

scenarios = {
    "well-formed, on-topic plan": CyberActionPlan(
        incident_id="explain-1", confidence=0.9,
        rationale="Critical severity Ransomware detected from 10.0.0.17 targeting DEVICE_17; executing isolate_segment, block_source, start_capture, notify_soc per incident response policy.",
        actions=[CyberAction(capability=c, target=alert.target_asset) for c in context.allowed_playbooks],
    ),
    "off-topic plan (bypassing the decision engine)": CyberActionPlan(
        incident_id="explain-2", confidence=0.05,
        rationale="Please summarize the quarterly sales report and email it to finance.",
        actions=[CyberAction(capability=c, target="wrong-asset") for c in context.allowed_playbooks],
    ),
}

for label, plan in scenarios.items():
    print(f"\n=== {label} ===")
    print(f"rationale: {plan.rationale}")

    print("\nSHAP attribution (Behavioral Trust, Rb - which feature pushed the anomaly score up/down):")
    attributions = explain_behavioral_trust(behavioral_engine, session, plan, history)
    for feature, value in sorted(attributions.items(), key=lambda kv: -abs(kv[1])):
        direction = "toward normal" if value > 0 else "toward anomalous"
        print(f"  {feature:<20} {value:+.4f}  ({direction})")

    print("\nLIME attribution (Semantic Risk, Rs - which words in the rationale pushed risk up/down):")
    explanation = explain_semantic_risk(semantic_engine, plan, context)
    for word, weight in explanation:
        direction = "toward high_risk" if weight > 0 else "toward low_risk"
        print(f"  {word:<20} {weight:+.4f}  ({direction})")

print("\nBoth explainers treat their target engine as a black box (real SHAP TreeExplainer on the IsolationForest, real LIME on SemanticRiskEngine.score()) - see analytics/explainability.py's module docstring.")
