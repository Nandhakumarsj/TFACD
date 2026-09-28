from __future__ import annotations

from tfacd.runtime.contracts import CyberActionPlan, ThreatContext

# REVIEW FIX (Section 19): these capabilities act on a specific
# source/segment - a missing target is not "no claim to check", it's a
# high-risk action with no stated target at all, and must not receive full
# target-consistency credit the way a None target does for other capabilities.
_TARGET_REQUIRED_CAPABILITIES = {"block_source", "rate_limit", "isolate_segment"}


def _expected_target(capability: str, context: ThreatContext) -> str | None:
    if capability in {"block_source", "rate_limit"}:
        return context.alert.source_id
    if capability == "isolate_segment":
        return context.alert.target_asset
    return None


def _target_is_consistent(capability: str, target: str | None, context: ThreatContext) -> bool:
    expected = _expected_target(capability, context)
    if capability in _TARGET_REQUIRED_CAPABILITIES:
        # REVIEW FIX (Section 19): target is mandatory for these - None can
        # never count as a match, however permissive that would be for a
        # capability with no `_expected_target` mapping.
        return target is not None and target == expected
    return target is None or target == expected


def score(plan: CyberActionPlan, context: ThreatContext) -> float:
    """Rc in [0,1]: how well the plan's claims match the threat context it's
    responding to - i.e. did it target the right entity for each capability.

    REVIEW FIX (Section 20): this used to also reward `plan.confidence` being
    close to `context.alert.confidence` ("confidence_alignment"), averaged in
    at 50% weight. That directly contradicts the LLM decision-engine prompt's
    explicit instruction to report its OWN calibrated confidence rather than
    copy the detector's - the two requirements fought each other, and an
    agent could raise its context-consistency score for free by simply
    parroting the detector confidence back. Removed entirely rather than
    patched: a future replacement (e.g. confidence <-> evidence consistency)
    should be a fresh, deliberately-scoped signal rather than a repaired
    version of this one. Context-consistency is now purely about whether the
    plan's actions target the right entity for the threat.
    """
    if not plan.actions:
        return 0.0
    return sum(1 for a in plan.actions if _target_is_consistent(a.capability, a.target, context)) / len(plan.actions)
