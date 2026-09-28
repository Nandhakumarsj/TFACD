"""Capture one end-to-end runtime workflow for the submission.

The decision engine is selected through configuration, so this workflow uses
the gated Ollama primary when the configured service is reachable. The trust
boundary remains independent and uses the deterministic semantic fallback to
keep the trust calculation reproducible without a Sentence-BERT download.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tfacd.agentic.factory import build_decision_engine
from tfacd.agentic.history import EntityHistory
from tfacd.common.config import load_config
from tfacd.runtime.contracts import IDSAlert, SessionContext
from tfacd.runtime.threat_context import ThreatContextGenerator
from tfacd.trust_boundary.audit import AuditLogger
from tfacd.trust_boundary.behavioral_trust import BehavioralTrustEngine
from tfacd.trust_boundary.boundary import AdaptiveSemanticTrustBoundary
from tfacd.trust_boundary.dynamic_trust import DynamicTrustScoreRegulator
from tfacd.trust_boundary.executor_factory import build_executor
from tfacd.trust_boundary.semantic_risk import SemanticRiskEngine


def capture(config_path: str, output_path: str) -> None:
    config = load_config(config_path)
    policy = load_config(config["runtime"]["trust_policy_path"])
    trust_config = config["trust_boundary"]
    history = EntityHistory()
    context_generator = ThreatContextGenerator(config["runtime"]["threat_context_mapping"])
    decision_engine = build_decision_engine(config, history)
    # REVIEW FIX (Section 40 - main runtime path != demonstration path): this
    # used to construct AdaptiveSemanticTrustBoundary without an `executor=`
    # argument at all, silently defaulting to SimulatedExecutor regardless of
    # what configs/*.yaml's trust_boundary.executor.mode actually says - so
    # this script's behavior could diverge from the configured runtime
    # without any error or warning. executor_factory.build_executor(config) is
    # the same canonical construction path scripts/run_streaming_demo.py and
    # scripts/run_attack_scenario.py already use.
    executor = build_executor(config)
    boundary = AdaptiveSemanticTrustBoundary(
        history=history,
        policy=policy,
        preprocessing_config=trust_config,
        executor=executor,
        trust_regulator=DynamicTrustScoreRegulator(
            trust_config["weight_semantic_risk"],
            trust_config["weight_context_consistency"],
            trust_config["weight_behavioral_trust"],
            trust_config["trust_level_thresholds"],
        ),
        semantic_risk_engine=SemanticRiskEngine(
            model_name=trust_config["sbert_model_name"], force_fallback=True,
        ),
        behavioral_trust_engine=BehavioralTrustEngine(
            high_risk_capabilities=set(policy["capability_whitelist"]["high_risk"]),
            ema_alpha=trust_config["ema_alpha"],
        ),
        audit_logger=AuditLogger(Path(output_path).with_name("workflow_audit.jsonl")),
    )

    alert = IDSAlert(
        attack_type="Port_Scanning",
        confidence=0.80,
        source_id="10.0.0.5",
        target_asset="plc-01",
        protocol="Modbus-TCP",
    )
    context = context_generator.enrich(alert)
    plan = decision_engine.decide(alert, context)
    session = SessionContext(
        agent_id="submission-workflow-agent",
        session_id="submission-workflow-session",
        issued_at=datetime.now(timezone.utc),
        nonce=uuid4().hex,
    )
    decision = boundary.evaluate(plan, context, session)

    trace = {
        "workflow": "IDS alert -> Threat Context -> Decision Engine -> Adaptive Semantic Trust Boundary -> execution/audit",
        "configured_engine": config["agentic"]["decision_engine"]["engine"],
        "actual_engine": plan.engine,
        "llm_model": config.get("agentic", {}).get("llm", {}).get("model"),
        "stages": [
            {"module": "IIoT input", "input": "Port_Scanning / Modbus-TCP", "output": alert.model_dump(mode="json")},
            {"module": "Threat Context Enrichment", "input": alert.attack_type, "output": context.model_dump(mode="json")},
            {"module": "Agentic Decision Engine", "input": "ThreatContext", "output": plan.model_dump(mode="json")},
            {"module": "Adaptive Semantic Trust Boundary", "input": "CyberActionPlan + SessionContext", "output": decision.model_dump(mode="json")},
            {"module": "Audit", "input": "final TrustDecision", "output": "workflow_audit.jsonl"},
        ],
        "checks": {
            "context_mapping": context.alert.attack_type == alert.attack_type,
            "playbooks_are_present": bool(context.allowed_playbooks),
            "plan_is_whitelisted": all(action.capability in context.allowed_playbooks for action in plan.actions),
            "decision_finalized": bool(decision.incident_id and decision.terminal_stage),
            "audit_written": Path(output_path).with_name("workflow_audit.jsonl").exists(),
        },
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(destination), "checks": trace["checks"], "decision": trace["stages"][3]["output"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/edge_iiot.yaml")
    parser.add_argument("--output", default="project_submission/work/end_to_end_trace.json")
    arguments = parser.parse_args()
    capture(arguments.config, arguments.output)