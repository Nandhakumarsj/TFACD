from __future__ import annotations

import logging
from typing import Any, Protocol

from tfacd.runtime.contracts import CyberAction, CyberActionPlan

logger = logging.getLogger(__name__)


class CapabilityExecutor(Protocol):
    def execute(self, action: CyberAction) -> bool: ...


class SimulatedExecutor:
    """No real firewall/SOC integration exists in this project. Logs what would
    happen and returns success - a real executor is a deployment-specific
    extension point, not something this repo pretends to have.
    """

    def execute(self, action: CyberAction) -> bool:
        logger.info("would execute: capability=%s target=%s params=%s", action.capability, action.target, action.parameters)
        return True


def enforce(plan: CyberActionPlan, autonomy_mode: str, policy: dict[str, Any], executor: CapabilityExecutor) -> list[str]:
    """Function interception: gates execution by autonomy mode and re-checks the
    whitelist immediately before invoking the executor, as defense-in-depth
    against drift between Stage 2's plan-validation time and this call-site time.
    """
    if autonomy_mode in ("read_only", "recommendation"):
        return []

    whitelist = policy["capability_whitelist"]
    low_risk = set(whitelist["low_risk"])
    known = low_risk | set(whitelist["high_risk"])

    executed: list[str] = []
    for action in plan.actions:
        if action.capability not in known:
            continue
        if autonomy_mode == "restricted_action" and action.capability not in low_risk:
            continue
        if executor.execute(action):
            executed.append(action.capability)
    return executed
