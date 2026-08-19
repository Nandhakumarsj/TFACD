from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from tfacd.runtime.contracts import CyberAction, CyberActionPlan, SessionContext, ThreatContext, TrustDecision
from tfacd.trust_boundary.boundary import AdaptiveSemanticTrustBoundary

# Four archetypes spanning the trust spectrum, so Phase II analytics (reputation
# ranking, forecasting, drift) have more than one agent's history to compare -
# without this, "Cross-Agent Reputation Engine" has nothing to rank against.
_ARCHETYPES = ("well_behaved", "borderline", "risky", "improving")


def _plan_for(archetype: str, round_index: int, context: ThreatContext, incident_id: str) -> CyberActionPlan:
    on_topic_rationale = (
        f"{context.severity.capitalize()} severity {context.alert.attack_type} detected; "
        f"recommending {', '.join(context.allowed_playbooks)} per policy."
    )
    off_topic_rationale = "Please summarize the quarterly sales report and email it to finance."

    if archetype == "well_behaved":
        rationale, confidence, target = on_topic_rationale, context.alert.confidence, context.alert.target_asset
    elif archetype == "borderline":
        rationale = on_topic_rationale if round_index % 2 == 0 else off_topic_rationale
        confidence, target = context.alert.confidence, context.alert.target_asset
    elif archetype == "risky":
        rationale, confidence, target = off_topic_rationale, 0.05, "wrong-asset"
    else:  # improving: starts risky, converges to on-topic over rounds
        rationale = off_topic_rationale if round_index < 2 else on_topic_rationale
        confidence = min(context.alert.confidence, 0.2 + 0.2 * round_index)
        target = "wrong-asset" if round_index < 2 else context.alert.target_asset

    actions = [CyberAction(capability=p, target=target) for p in context.allowed_playbooks]
    return CyberActionPlan(incident_id=f"{incident_id}-{archetype}-{round_index}", rationale=rationale, actions=actions, confidence=confidence)


def simulate_agent_population(
    boundary: AdaptiveSemanticTrustBoundary,
    context: ThreatContext,
    rounds_per_agent: int = 5,
) -> dict[str, list[TrustDecision]]:
    """Runs each archetype through `rounds_per_agent` interactions against the
    same boundary, building a real multi-agent audit log. Returns each agent's
    decisions for inspection (tests, the demo script's printed summary)."""
    results: dict[str, list[TrustDecision]] = {}
    for archetype in _ARCHETYPES:
        agent_id = f"agent-{archetype}"
        decisions = []
        for round_index in range(rounds_per_agent):
            # A genuinely unique nonce per round (uuid4, not derived from the fixed
            # archetype/round_index) - the caller's history is typically persisted
            # across script runs, and a deterministic nonce would collide with a
            # PRIOR run's still-fresh nonce (verified live in
            # scripts/run_trust_boundary_demo.py - the same failure mode).
            session = SessionContext(
                agent_id=agent_id, session_id=f"session-{archetype}-{round_index}",
                issued_at=datetime.now(timezone.utc), nonce=uuid4().hex,
            )
            plan = _plan_for(archetype, round_index, context, incident_id=f"synthetic-{archetype}")
            decisions.append(boundary.evaluate(plan, context, session))
        results[agent_id] = decisions
    return results
