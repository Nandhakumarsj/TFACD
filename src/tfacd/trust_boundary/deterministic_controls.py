from __future__ import annotations

from typing import Any

from tfacd.runtime.contracts import CyberActionPlan, StageResult, ThreatContext


def run(plan: CyberActionPlan, context: ThreatContext, policy: dict[str, Any]) -> StageResult:
    reasons: list[str] = []

    if not plan.incident_id.strip():
        reasons.append("incident_id is empty")
    if not plan.rationale.strip():
        reasons.append("rationale is empty")

    whitelist = policy["capability_whitelist"]
    known_capabilities = set(whitelist["low_risk"]) | set(whitelist["high_risk"])
    for action in plan.actions:
        if action.capability not in known_capabilities:
            reasons.append(f"capability '{action.capability}' not in tool whitelist")
        elif action.capability not in context.allowed_playbooks:
            reasons.append(f"capability '{action.capability}' not authorized for this threat context (allowed: {context.allowed_playbooks})")

    return StageResult(stage="deterministic_controls", accepted=not reasons, reasons=reasons)
