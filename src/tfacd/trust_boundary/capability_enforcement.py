from __future__ import annotations

import logging
from typing import Any, Protocol

from tfacd.runtime.contracts import CyberAction, CyberActionPlan, ThreatContext

logger = logging.getLogger(__name__)


class CapabilityExecutor(Protocol):
    def execute(self, action: CyberAction) -> bool: ...


class SimulatedExecutor:
    """No real firewall/SOC integration exists in this project. Logs what would
    happen and returns success - a real executor is a deployment-specific
    extension point, not something this repo pretends to have.
    """

    mode = "simulate"

    def execute(self, action: CyberAction) -> bool:
        logger.info("would execute: capability=%s target=%s params=%s", action.capability, action.target, action.parameters)
        return True


def enforce(plan: CyberActionPlan, autonomy_mode: str, policy: dict[str, Any], executor: CapabilityExecutor, context: ThreatContext) -> list[str]:
    """Function interception: gates execution by autonomy mode and re-checks the
    whitelist immediately before invoking the executor, as defense-in-depth
    against drift between Stage 2's plan-validation time and this call-site time.

    `context` is required (not optional) so this re-check can never silently
    degrade to whitelist-only: Stage 2 (deterministic_controls.py) enforces TWO
    independent things - the static low_risk/high_risk whitelist AND
    context.allowed_playbooks (the threat-context-specific authorization for
    THIS incident's severity). Before this parameter existed, only the first
    was re-verified here, so the docstring's "re-checks the whitelist" claim
    was accurate only for the static whitelist, not the full set of checks
    Stage 2 performs - verified live during an architecture audit.
    """
    if autonomy_mode in ("read_only", "recommendation"):
        return []

    whitelist = policy["capability_whitelist"]
    low_risk = set(whitelist["low_risk"])
    known = low_risk | set(whitelist["high_risk"])
    allowed_for_context = set(context.allowed_playbooks)

    executed: list[str] = []
    for action in plan.actions:
        if action.capability not in known:
            continue
        if action.capability not in allowed_for_context:
            continue
        if autonomy_mode == "restricted_action" and action.capability not in low_risk:
            continue
        if executor.execute(action):
            executed.append(action.capability)
    return executed
