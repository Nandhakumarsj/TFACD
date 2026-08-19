"""One alert through the full runtime pipeline: Threat Context Generator ->
Agentic Decision Engine -> Adaptive Semantic Trust Boundary -> (capability
enforcement/audit as part of boundary.evaluate()). Extracted out of
scripts/run_streaming_demo.py's per-incident loop body so
scripts/run_attack_scenario.py can reuse the exact same wiring instead of a
second, copy-pasted implementation that could drift from it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from tfacd.agentic.base import DecisionEngine
from tfacd.runtime.contracts import IDSAlert, SessionContext, ThreatContext, TrustDecision
from tfacd.runtime.threat_context import ThreatContextGenerator
from tfacd.trust_boundary.boundary import AdaptiveSemanticTrustBoundary


def run_incident(
    alert: IDSAlert,
    *,
    threat_context_generator: ThreatContextGenerator,
    decision_engine: DecisionEngine,
    boundary: AdaptiveSemanticTrustBoundary,
    agent_id: str,
) -> tuple[ThreatContext, TrustDecision]:
    context = threat_context_generator.enrich(alert)
    plan = decision_engine.decide(alert, context)
    # A fresh session per incident, not one shared session for the whole run:
    # session_max_age_seconds would otherwise expire mid-replay, and a new
    # nonce per plan is semantically correct for a live IDS-driven agent
    # anyway. uuid4(), not id(alert) - object identity isn't a real
    # uniqueness guarantee, and history is persisted across script runs, so a
    # collision-prone nonce risks a false-replay-rejection.
    session = SessionContext(
        agent_id=agent_id, session_id=f"session-{alert.attack_type}", issued_at=datetime.now(timezone.utc), nonce=uuid4().hex,
    )
    decision = boundary.evaluate(plan, context, session)
    return context, decision
