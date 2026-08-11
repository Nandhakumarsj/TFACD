"""Cross-agent reputation from the trust-boundary audit trail: a recency-
weighted mean of TrustScores.trust_value (T) per agent, minus a flat penalty
per policy violation - TrustDecision.accepted is the violation signal (not
scores presence alone), since a hard Stage-1/2 rejection (scores=None) and a
soft Stage-3 Low-trust block (scores present, accepted=False) are both policy
violations for reputation purposes. Entries with agent_id=None predate the
multi-agent fixture and carry no agent to attribute reputation to, so they're
skipped.

"Streaming Trust Signals": update() mutates running per-agent state from a
batch of AuditEntry, so it CAN be called repeatedly on new log tail slices
without re-reading history already scored. load_entries()/rank_agents() below
still do one full file re-read per call for now - there's no tailing/offset-
tracking log reader built yet, only the incremental-update state that would
make one worthwhile later.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from tfacd.runtime.contracts import AuditEntry

DEFAULT_AUDIT_LOG = "artifacts/trust_boundary/audit_log.jsonl"
_RECENT_WINDOW = 5  # mean of the last N *scored* decisions per agent
_VIOLATION_PENALTY = 0.15  # subtracted per not-accepted decision in an agent's history


@dataclass
class _AgentState:
    """Per-agent running state - the unit update() mutates incrementally."""

    recent_trust: deque[float]
    num_interactions: int = 0
    num_violations: int = 0


@dataclass
class AgentReputation:
    agent_id: str
    reputation_score: float  # not clamped to [0,1] like TrustScores - violations can push it negative
    num_interactions: int
    num_violations: int


class CrossAgentReputationEngine:
    """Fits no model and holds no population - just accumulates per-agent
    counters/recent-trust windows, mirroring the dict[str, float] running-state
    idiom BehavioralTrustEngine uses for its trust EMA.
    """

    def __init__(self, recent_window: int = _RECENT_WINDOW, violation_penalty: float = _VIOLATION_PENALTY):
        self.recent_window = recent_window
        self.violation_penalty = violation_penalty
        self._agents: dict[str, _AgentState] = {}

    def update(self, entries: list[AuditEntry]) -> None:
        # Assumes entries arrive in chronological (sequence) order, as they do
        # both when read top-to-bottom from the log and when appended live -
        # the recent-trust deque's maxlen otherwise wouldn't mean "last N".
        for entry in entries:
            if entry.agent_id is None:
                continue
            state = self._agents.setdefault(entry.agent_id, _AgentState(recent_trust=deque(maxlen=self.recent_window)))
            state.num_interactions += 1
            decision = entry.decision
            if not decision.accepted:
                state.num_violations += 1
            if decision.scores is not None:
                state.recent_trust.append(decision.scores.trust_value)

    def rank(self) -> list[AgentReputation]:
        """Best to worst by reputation_score."""
        results = []
        for agent_id, state in self._agents.items():
            recent_avg = sum(state.recent_trust) / len(state.recent_trust) if state.recent_trust else 0.0
            score = recent_avg - self.violation_penalty * state.num_violations
            results.append(AgentReputation(agent_id, score, state.num_interactions, state.num_violations))
        return sorted(results, key=lambda r: r.reputation_score, reverse=True)


def load_entries(path: str | Path = DEFAULT_AUDIT_LOG) -> list[AuditEntry]:
    """Full re-read of the hash-chained log - does not itself call verify_chain;
    callers who need tamper-evidence guarantees should call that separately."""
    entries = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(AuditEntry.model_validate_json(line))
    return entries


def rank_agents(
    path: str | Path = DEFAULT_AUDIT_LOG,
    recent_window: int = _RECENT_WINDOW,
    violation_penalty: float = _VIOLATION_PENALTY,
) -> list[AgentReputation]:
    """One-shot convenience for scripts/CLIs: full-log read + rank."""
    engine = CrossAgentReputationEngine(recent_window=recent_window, violation_penalty=violation_penalty)
    engine.update(load_entries(path))
    return engine.rank()
