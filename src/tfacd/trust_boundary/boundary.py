from __future__ import annotations

from typing import Any

from tfacd.agentic.history import EntityHistory
from tfacd.runtime.contracts import CyberActionPlan, SessionContext, StageResult, ThreatContext, TrustDecision, TrustScores
from tfacd.trust_boundary import capability_enforcement, context_consistency, deterministic_controls, output_protection, preprocessing
from tfacd.trust_boundary.audit import AuditLogger
from tfacd.trust_boundary.behavioral_trust import BehavioralTrustEngine
from tfacd.trust_boundary.capability_enforcement import CapabilityExecutor, SimulatedExecutor
from tfacd.trust_boundary.dynamic_trust import DynamicTrustScoreRegulator
from tfacd.trust_boundary.memory_integrity import sanitize_event_payload
from tfacd.trust_boundary.semantic_risk import SemanticRiskEngine


class AdaptiveSemanticTrustBoundary:
    """Orchestrates the full pipeline: Stage 1/2 (short-circuit on rejection) ->
    Stage 3 trust scoring -> capability enforcement -> memory write -> output
    sanitization -> audit logging. Stage 3 only runs once Stage 1/2 pass -
    computing Rs/Rc/Rb for an already-rejected plan would be exactly the kind
    of overclaiming this project's research-honesty conventions warn against.
    """

    def __init__(
        self,
        history: EntityHistory,
        policy: dict[str, Any],
        preprocessing_config: dict[str, Any],
        trust_regulator: DynamicTrustScoreRegulator,
        semantic_risk_engine: SemanticRiskEngine,
        behavioral_trust_engine: BehavioralTrustEngine,
        audit_logger: AuditLogger,
        executor: CapabilityExecutor | None = None,
    ):
        self.history = history
        self.policy = policy
        self.preprocessing_config = preprocessing_config
        self.trust_regulator = trust_regulator
        self.semantic_risk_engine = semantic_risk_engine
        self.behavioral_trust_engine = behavioral_trust_engine
        self.audit_logger = audit_logger
        self.executor = executor or SimulatedExecutor()

    def evaluate(self, plan: CyberActionPlan, context: ThreatContext, session: SessionContext) -> TrustDecision:
        stage_results: list[StageResult] = []

        stage1_result, plan = preprocessing.run(plan, session, self.history, self.preprocessing_config)
        stage_results.append(stage1_result)
        if not stage1_result.accepted:
            return self._finalize(plan, session, stage_results, terminal_stage="preprocessing", accepted=False)

        stage2_result = deterministic_controls.run(plan, context, self.policy)
        stage_results.append(stage2_result)
        if not stage2_result.accepted:
            return self._finalize(plan, session, stage_results, terminal_stage="deterministic_controls", accepted=False)

        semantic_risk = self.semantic_risk_engine.score(plan, context)
        consistency = context_consistency.score(plan, context)
        behavioral_trust = self.behavioral_trust_engine.score(session, plan, self.history)
        scores = self.trust_regulator.evaluate(semantic_risk, consistency, behavioral_trust)
        trust_level = self.trust_regulator.trust_level(scores.trust_value)
        autonomy_mode = self.trust_regulator.autonomy_mode(trust_level)

        executed_actions = capability_enforcement.enforce(plan, autonomy_mode, self.policy, self.executor, context)
        accepted = autonomy_mode != "read_only"

        return self._finalize(
            plan, session, stage_results, terminal_stage="capability_enforcement", accepted=accepted,
            trust_level=trust_level, autonomy_mode=autonomy_mode, scores=scores, executed_actions=executed_actions,
        )

    def _finalize(
        self, plan: CyberActionPlan, session: SessionContext, stage_results: list[StageResult], *, terminal_stage: str, accepted: bool,
        trust_level: str | None = None, autonomy_mode: str | None = None, scores: TrustScores | None = None,
        executed_actions: list[str] | None = None,
    ) -> TrustDecision:
        decision = TrustDecision(
            incident_id=plan.incident_id,
            accepted=accepted,
            terminal_stage=terminal_stage,
            trust_level=trust_level,
            autonomy_mode=autonomy_mode,
            scores=scores,
            stage_results=stage_results,
            executed_actions=executed_actions or [],
            rationale=plan.rationale,
            engine=plan.engine,
        )
        decision = output_protection.sanitize_decision(decision)

        payload = sanitize_event_payload(
            {
                "accepted": accepted,
                "trust_level": trust_level,
                "autonomy_mode": autonomy_mode,
                "trust_value": scores.trust_value if scores else None,
                "capabilities": sorted(a.capability for a in plan.actions),
                "policy_violation": not accepted,
            }
        )
        self.history.append(session.agent_id, "trust_decision", payload)
        self.audit_logger.append(decision, agent_id=session.agent_id)
        return decision
