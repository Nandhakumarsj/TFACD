from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

from tfacd.agentic.history import EntityHistory
from tfacd.runtime.contracts import CyberAction, CyberActionPlan, IDSAlert, ThreatContext

# Capabilities that act ON THE ATTACKER (block their traffic / throttle their
# rate) need CyberAction.target = the attacker's address (alert.source_id).
# Every other capability (isolate_segment protects the asset's segment,
# rotate_session/low-risk actions are asset- or session-scoped, not attacker-
# scoped) keeps target = alert.target_asset, today's existing behavior.
# Concrete consequence of getting this wrong: SimulatedExecutor never cared
# what `target` held (it only ever logged it), but trust_boundary/
# production_executor.py's real block_source backend genuinely needs the
# attacker's IP to build a correct firewall rule - before this, every
# capability (including block_source) was handed target_asset (the PROTECTED
# device), which would have blocked the wrong address.
_SOURCE_TARGETED_CAPABILITIES = frozenset({"block_source", "rate_limit"})


def _action_target(capability: str, alert: IDSAlert) -> str | None:
    if capability in _SOURCE_TARGETED_CAPABILITIES:
        return alert.source_id
    return alert.target_asset


# Filled with {attack_type}/{source_id}/{target_asset}/{playbooks} at decision time.
# semantic_risk.py scores plan.rationale against the same templates, keyed by severity.
RATIONALE_TEMPLATES = {
    "critical": "Critical severity {attack_type} detected from {source_id} targeting {target_asset}; "
    "executing {playbooks} per incident response policy to contain the threat immediately.",
    "high": "High severity {attack_type} detected from {source_id}; executing {playbooks} to contain and investigate.",
    "medium": "Medium severity {attack_type} detected; recommending {playbooks} for investigation and containment.",
    "low": "Low severity {attack_type} observed; recommending {playbooks} for monitoring.",
    "informational": "Informational event {attack_type} observed; logging for visibility, no containment action required.",
}


class AgenticDecisionEngine:
    """Turns a ThreatContext into a CyberActionPlan: correlate, reason, recommend.

    Incident correlation keys the shared history store by IDSAlert.source_id (the
    attacker) - a different namespace from the trust boundary's SessionContext.agent_id
    keying (the agent submitting the plan for trust evaluation), even though both
    read/write the same EntityHistory instance if one is shared.
    """

    def __init__(self, history: EntityHistory | None = None, repeat_window_minutes: int = 30):
        self.history = history or EntityHistory()
        self.repeat_window = timedelta(minutes=repeat_window_minutes)

    def decide(self, alert: IDSAlert, context: ThreatContext) -> CyberActionPlan:
        source = alert.source_id or "unknown-source"
        prior_incidents = self.history.recent(source, kind="incident", within=self.repeat_window)
        repeat_activity = len(prior_incidents) > 0

        playbooks = list(context.allowed_playbooks)
        actions = [CyberAction(capability=playbook, target=_action_target(playbook, alert), parameters={}) for playbook in playbooks]

        rationale = RATIONALE_TEMPLATES[context.severity].format(
            attack_type=alert.attack_type,
            source_id=source,
            target_asset=alert.target_asset or "unknown-asset",
            playbooks=", ".join(playbooks) or "no playbooks",
        )
        if repeat_activity:
            window_minutes = int(self.repeat_window.total_seconds() // 60)
            rationale += f" Repeat activity from {source} observed in the last {window_minutes} minutes."

        confidence = min(1.0, alert.confidence + 0.05) if repeat_activity else alert.confidence
        incident_id = f"{source}-{context.priority}-{uuid4().hex[:8]}"

        plan = CyberActionPlan(incident_id=incident_id, rationale=rationale, actions=actions, confidence=confidence)
        self.history.append(
            source,
            "incident",
            {"attack_type": alert.attack_type, "target_asset": alert.target_asset, "severity": context.severity, "incident_id": incident_id},
        )
        return plan
