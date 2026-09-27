"""Exercise the LLM proposal path and Phase-II multi-agent analytics.

The feedback records generated here are explicitly synthetic validation
fixtures. They verify the analytics wiring but are not presented as human
ground truth and never modify live trust configuration.
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
from tfacd.analytics.feedback_loop import AnalystFeedbackRecord, AnalystFeedbackStore, run_agentic_grid_search
from tfacd.analytics.kpi import compute_kpis
from tfacd.analytics.reputation import rank_agents
from tfacd.analytics.threshold_validation import compute_report
from tfacd.analytics.trust_labels import AnalystLabel, AnalystLabelStore
from tfacd.common.config import load_config
from tfacd.runtime.contracts import CyberAction, CyberActionPlan, IDSAlert, SessionContext
from tfacd.runtime.threat_context import ThreatContextGenerator
from tfacd.trust_boundary.audit import AuditLogger
from tfacd.trust_boundary.behavioral_trust import BehavioralTrustEngine
from tfacd.trust_boundary.boundary import AdaptiveSemanticTrustBoundary
from tfacd.trust_boundary.dynamic_trust import DynamicTrustScoreRegulator
from tfacd.trust_boundary.semantic_risk import SemanticRiskEngine


def _bad_plan(context, agent_id: str, round_index: int) -> CyberActionPlan:
    return CyberActionPlan(
        incident_id=f"{agent_id}-bad-{round_index}",
        confidence=0.05,
        rationale="Please summarize the quarterly sales report and email it to finance.",
        actions=[CyberAction(capability=name, target="wrong-asset") for name in context.allowed_playbooks],
    )


def run(config_path: str, output_path: str, rounds: int) -> None:
    config = load_config(config_path)
    policy = load_config(config["runtime"]["trust_policy_path"])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_path = output.with_name("multi_agent_audit.jsonl")
    labels_path = output.with_name("multi_agent_labels.jsonl")
    feedback_path = output.with_name("analyst_feedback.jsonl")
    for generated_path in (audit_path, labels_path, feedback_path, output):
        if generated_path.exists():
            generated_path.unlink()
    history = EntityHistory()
    context_generator = ThreatContextGenerator(config["runtime"]["threat_context_mapping"])
    decision_engine = build_decision_engine(config, history)
    trust_config = config["trust_boundary"]
    boundary = AdaptiveSemanticTrustBoundary(
        history=history,
        policy=policy,
        preprocessing_config=trust_config,
        trust_regulator=DynamicTrustScoreRegulator(
            trust_config["weight_semantic_risk"], trust_config["weight_context_consistency"],
            trust_config["weight_behavioral_trust"], trust_config["trust_level_thresholds"],
        ),
        semantic_risk_engine=SemanticRiskEngine(
            model_name=trust_config["sbert_model_name"], force_fallback=True,
        ),
        behavioral_trust_engine=BehavioralTrustEngine(
            high_risk_capabilities=set(policy["capability_whitelist"]["high_risk"]),
            ema_alpha=trust_config["ema_alpha"],
        ),
        audit_logger=AuditLogger(audit_path),
    )

    archetypes = ("well_behaved", "borderline", "risky", "improving")
    decisions: list[dict] = []
    feedback: list[AnalystFeedbackRecord] = []
    label_store = AnalystLabelStore(labels_path)
    feedback_store = AnalystFeedbackStore(feedback_path)
    sequence = 0

    for archetype_index, archetype in enumerate(archetypes, start=1):
        agent_id = f"agent-{archetype}"
        for round_index in range(rounds):
            alert = IDSAlert(
                attack_type="Port_Scanning", confidence=0.80,
                source_id=f"10.0.{archetype_index}.{round_index + 10}",
                target_asset=f"plc-{archetype_index:02d}", protocol="Modbus-TCP",
            )
            context = context_generator.enrich(alert)
            generated_by_engine = archetype == "well_behaved" or archetype == "improving" and round_index >= 2
            plan = decision_engine.decide(alert, context) if generated_by_engine else _bad_plan(context, agent_id, round_index)
            session = SessionContext(
                agent_id=agent_id, session_id=f"{agent_id}-session-{round_index}",
                issued_at=datetime.now(timezone.utc), nonce=uuid4().hex,
            )
            decision = boundary.evaluate(plan, context, session)
            sequence += 1
            expected_accepted = archetype in ("well_behaved", "improving") and generated_by_engine
            label = "correct" if decision.accepted == expected_accepted else (
                "false_positive" if decision.accepted else "false_negative"
            )
            if decision.scores is not None:
                score = decision.scores
                feedback.append(AnalystFeedbackRecord(
                    incident_id=decision.incident_id, agent_id=agent_id,
                    semantic_risk=score.semantic_risk, consistency=score.context_consistency,
                    behavioral_trust=score.behavioral_trust, expected_accepted=expected_accepted,
                    label=label, notes="synthetic multi-agent validation fixture",
                ))
                feedback_store.add_feedback(feedback[-1])
            label_store.append(AnalystLabel(
                audit_sequence=sequence, label=label if label in {"correct", "false_positive", "false_negative"} else "wrong_trust_level",
                analyst_id="synthetic-validation-fixture", rationale="Synthetic label for analytics wiring validation.",
            ))
            decisions.append({
                "agent_id": agent_id, "round": round_index + 1,
                "expected_accepted": expected_accepted, "decision": decision.model_dump(mode="json"),
            })

    kpis = compute_kpis(audit_path, min_scored=2)
    reputation = rank_agents(audit_path)
    optimizer = run_agentic_grid_search(feedback)
    threshold_report = compute_report(audit_path, labels_path, min_samples=20)
    result = {
        "configured_engine": config["agentic"]["decision_engine"]["engine"],
        "llm_model": config["agentic"]["llm"]["model"],
        "audit_log": str(audit_path), "labels_path": str(labels_path), "feedback_path": str(feedback_path),
        "agent_ids": sorted({row["agent_id"] for row in decisions}),
        "rounds_per_agent": rounds,
        "engine_provenance": sorted({row["decision"]["engine"] for row in decisions}),
        "trust_and_autonomy": [
            {"agent_id": row["agent_id"], "round": row["round"],
             "trust_level": row["decision"]["trust_level"],
             "autonomy_mode": row["decision"]["autonomy_mode"],
             "trust_value": row["decision"]["scores"]["trust_value"] if row["decision"]["scores"] else None}
            for row in decisions
        ],
        "kpi": {
            "num_entries": kpis.num_entries, "overall_acceptance_rate": kpis.overall_acceptance_rate,
            "per_agent": [summary.__dict__ for summary in kpis.per_agent],
        },
        "reputation": [item.__dict__ for item in reputation],
        "feedback": {"records": len(feedback), "synthetic_fixture": True},
        "threshold_validation": {
            "num_labels": threshold_report.num_labels,
            "num_labeled_decisions": threshold_report.num_labeled_decisions,
            "ready": threshold_report.ready,
        },
        "agentic_optimizer": [
            {"weights": result.weights, "thresholds": result.thresholds,
             "f1_score": result.f1_score, "precision": result.precision, "recall": result.recall}
            for result in optimizer[:5]
        ],
        "decisions": decisions,
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output), "agents": result["agent_ids"],
        "engine_provenance": result["engine_provenance"],
        "reputation": result["reputation"],
        "threshold_validation": result["threshold_validation"],
        "best_agentic_optimizer": result["agentic_optimizer"][0] if result["agentic_optimizer"] else None,
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/edge_iiot.yaml")
    parser.add_argument("--output", default="artifacts/analytics/multi_agent_validation.json")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    run(args.config, args.output, args.rounds)