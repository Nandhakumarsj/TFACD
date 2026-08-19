from __future__ import annotations

from typing import Protocol

from tfacd.runtime.contracts import CyberActionPlan, IDSAlert, ThreatContext


class DecisionEngine(Protocol):
    def decide(self, alert: IDSAlert, context: ThreatContext) -> CyberActionPlan: ...
