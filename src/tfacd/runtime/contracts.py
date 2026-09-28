from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class IDSAlert(BaseModel):
    attack_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    source_id: str | None = None
    target_asset: str | None = None
    protocol: str | None = None


class ThreatContext(BaseModel):
    alert: IDSAlert
    severity: Literal["informational", "low", "medium", "high", "critical"]
    priority: Literal["P1", "P2", "P3", "P4"]
    mitre_techniques: list[str]
    allowed_playbooks: list[str]


class CyberAction(BaseModel):
    capability: str
    target: str | None = None
    parameters: dict[str, str | int | float | bool] = {}


class CyberActionPlan(BaseModel):
    incident_id: str
    rationale: str
    actions: list[CyberAction]
    confidence: float = Field(ge=0.0, le=1.0)
    # Which decision engine produced this plan ("template", "llm", or "fallback:<reason>")
    # - audit provenance, not a security control. See agentic/base.py::DecisionEngine.
    engine: str = "template"


class SessionContext(BaseModel):
    """Identifies the agent/session submitting a plan for trust evaluation.

    Never derive this from IDSAlert.source_id - that identifies the attacker,
    a different entity from the agent whose behavior the trust boundary scores.
    """

    agent_id: str
    session_id: str
    tenant_id: str = "default"
    issued_at: datetime
    nonce: str


class StageResult(BaseModel):
    stage: str
    accepted: bool
    reasons: list[str] = []


class TrustScores(BaseModel):
    semantic_risk: float = Field(ge=0.0, le=1.0)
    context_consistency: float = Field(ge=0.0, le=1.0)
    behavioral_trust: float = Field(ge=0.0, le=1.0)
    trust_value: float = Field(ge=0.0, le=1.0)


class TrustDecision(BaseModel):
    incident_id: str
    # REVIEW FIX (Section 18 - overloaded `accepted` semantics): `accepted`
    # means ONLY "the trust boundary approved this plan for some level of
    # autonomy" (i.e. autonomy_mode != "read_only"). It says nothing about
    # whether any action actually ran, or ran successfully - that is what
    # `execution_status` below is for. Analytics/consumers that need to know
    # whether an action was actually carried out must read `execution_status`,
    # not infer it from `accepted`.
    accepted: bool
    terminal_stage: str
    trust_level: Literal["low", "medium", "high", "verified"] | None = None
    autonomy_mode: Literal["read_only", "recommendation", "restricted_action", "autonomous_execution"] | None = None
    scores: TrustScores | None = None
    stage_results: list[StageResult] = []
    executed_actions: list[str] = []
    rationale: str
    engine: str = "template"
    # Which CapabilityExecutor ran executed_actions - "simulate" (SimulatedExecutor,
    # today's only executor) or "production" (ProductionExecutor, a real
    # deployment-specific backend). None when capability_enforcement never ran
    # (an earlier stage already rejected the plan), same convention as
    # trust_level/autonomy_mode/scores above. Provenance, not itself a security
    # control - same posture as `engine`.
    executor_mode: Literal["simulate", "production"] | None = None
    # REVIEW FIX (Section 18): separates "was this plan trust-approved"
    # (`accepted`) from "what actually happened when we tried to carry it
    # out". Values:
    #   not_attempted        - capability_enforcement never ran (an earlier
    #                          stage rejected the plan) or autonomy_mode was
    #                          "read_only"/"recommendation" so nothing was
    #                          eligible to run.
    #   simulated            - every eligible action ran, via SimulatedExecutor
    #                          (no real-world effect).
    #   executed             - every eligible action ran, via ProductionExecutor
    #                          (real-world effect).
    #   failed               - at least one action was eligible to run but the
    #                          executor reported failure for ALL of them.
    #   partially_executed   - the executor reported failure for SOME but not
    #                          all eligible actions.
    execution_status: Literal[
        "not_attempted", "simulated", "executed", "failed", "partially_executed"
    ] = "not_attempted"


class AuditEntry(BaseModel):
    sequence: int
    timestamp: datetime
    incident_id: str
    # Who submitted the plan - log/provenance metadata, not part of the
    # decision's own content. Lets Phase II analytics group entries per agent.
    agent_id: str | None = None
    entry_hash: str
    previous_hash: str
    decision: TrustDecision
