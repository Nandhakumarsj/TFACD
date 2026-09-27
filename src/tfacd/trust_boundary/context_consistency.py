from __future__ import annotations

from tfacd.runtime.contracts import CyberActionPlan, ThreatContext


def _expected_target(capability: str, context: ThreatContext) -> str | None:
    if capability in {"block_source", "rate_limit"}:
        return context.alert.source_id
    if capability == "isolate_segment":
        return context.alert.target_asset
    return None


def score(plan: CyberActionPlan, context: ThreatContext) -> float:
    """Rc in [0,1]: how well the plan's claims match the threat context it's responding to."""
    if plan.actions:
        target_matches = sum(
            1
            for a in plan.actions
            if a.target is None or a.target == _expected_target(a.capability, context)
        ) / len(plan.actions)
    else:
        target_matches = 0.0

    confidence_gap = abs(plan.confidence - context.alert.confidence)
    confidence_alignment = max(0.0, 1.0 - confidence_gap)

    return (target_matches + confidence_alignment) / 2
