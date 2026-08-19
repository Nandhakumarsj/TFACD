"""KPI aggregation over the trust-boundary audit trail for the Phase II
security dashboard - pure read/compute functions only, no HTML here (that
lives in scripts/generate_security_dashboard.py) so this stays unit-testable
without a rendering harness. TrustDecision.scores is None for a hard Stage-1/2
rejection (preprocessing/deterministic_controls) - no trust score was ever
computed for those, so they're counted as their own "hard_rejected" bucket
rather than folded into a trust level or given a fabricated placeholder score.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tfacd.runtime.contracts import AuditEntry

DEFAULT_AUDIT_LOG = "artifacts/trust_boundary/audit_log.jsonl"
_TRUST_LEVELS = ("low", "medium", "high", "verified")


@dataclass
class AgentKPISummary:
    agent_id: str
    num_interactions: int
    num_scored: int  # subset of num_interactions with decision.scores is not None
    acceptance_rate: float
    mean_trust_value: float | None  # None when num_scored == 0 - nothing to average


@dataclass
class KPIReport:
    num_entries: int
    overall_acceptance_rate: float
    trust_level_distribution: dict[str, int]  # low/medium/high/verified + hard_rejected
    per_agent: list[AgentKPISummary]
    top_agents: list[AgentKPISummary]
    bottom_agents: list[AgentKPISummary]


def load_entries(path: str | Path = DEFAULT_AUDIT_LOG) -> list[AuditEntry]:
    """Full re-read of the hash-chained log - does not verify the chain itself
    (see trust_boundary.audit.verify_chain for that); mirrors analytics.reputation's
    loader, kept separate so kpi.py has no cross-module dependency within analytics/."""
    entries = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(AuditEntry.model_validate_json(line))
    return entries


def overall_acceptance_rate(entries: list[AuditEntry]) -> float:
    if not entries:
        return 0.0
    return sum(1 for e in entries if e.decision.accepted) / len(entries)


def trust_level_distribution(entries: list[AuditEntry]) -> dict[str, int]:
    counts = {level: 0 for level in _TRUST_LEVELS}
    counts["hard_rejected"] = 0
    for entry in entries:
        if entry.decision.scores is None:
            counts["hard_rejected"] += 1
        elif entry.decision.trust_level in counts:
            counts[entry.decision.trust_level] += 1
    return counts


def per_agent_summary(entries: list[AuditEntry]) -> list[AgentKPISummary]:
    """One AgentKPISummary per distinct agent_id. Entries with agent_id=None
    (pre-multi-agent-fixture history) are skipped - same precedent as
    analytics.reputation: there is no agent identity to attribute a summary to."""
    by_agent: dict[str, list[AuditEntry]] = {}
    for entry in entries:
        if entry.agent_id is None:
            continue
        by_agent.setdefault(entry.agent_id, []).append(entry)

    summaries = []
    for agent_id, agent_entries in by_agent.items():
        scored_values = [e.decision.scores.trust_value for e in agent_entries if e.decision.scores is not None]
        accepted = sum(1 for e in agent_entries if e.decision.accepted)
        summaries.append(
            AgentKPISummary(
                agent_id=agent_id,
                num_interactions=len(agent_entries),
                num_scored=len(scored_values),
                acceptance_rate=accepted / len(agent_entries),
                mean_trust_value=sum(scored_values) / len(scored_values) if scored_values else None,
            )
        )
    return summaries


def top_bottom_agents(
    summaries: list[AgentKPISummary], n: int = 3, min_scored: int = 2
) -> tuple[list[AgentKPISummary], list[AgentKPISummary]]:
    """Ranks agents by mean_trust_value, skipping agents with fewer than
    `min_scored` scored entries (not enough signal for a stable mean). Returns
    (top n best-first, bottom n worst-first); the two can overlap when fewer
    than 2n agents are eligible - that's a correct reflection of a small
    population, not deduplicated away."""
    eligible = sorted(
        (s for s in summaries if s.num_scored >= min_scored),
        key=lambda s: s.mean_trust_value,
        reverse=True,
    )
    top = eligible[:n]
    bottom = list(reversed(eligible))[:n]
    return top, bottom


def compute_kpis(path: str | Path = DEFAULT_AUDIT_LOG, top_n: int = 3, min_scored: int = 2) -> KPIReport:
    """One-shot convenience for scripts/CLIs: full-log read + every KPI above."""
    entries = load_entries(path)
    per_agent = per_agent_summary(entries)
    top, bottom = top_bottom_agents(per_agent, n=top_n, min_scored=min_scored)
    return KPIReport(
        num_entries=len(entries),
        overall_acceptance_rate=overall_acceptance_rate(entries),
        trust_level_distribution=trust_level_distribution(entries),
        per_agent=per_agent,
        top_agents=top,
        bottom_agents=bottom,
    )
