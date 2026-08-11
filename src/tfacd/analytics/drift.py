from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

# Page-Hinkley test (Page 1954 / Hinkley 1971): a lightweight, well-known
# cumulative-sum change detector for a shift in a stream's mean. One-sided by
# construction below - it flags an *increase* in the running mean beyond a
# `delta` tolerance, not decreases; to also catch decreases, run a second
# detector fed the negated stream. This ONE detector class is instantiated
# TWICE further down for two unrelated data sources - see the module-level
# note above each instantiation for what each one can and cannot claim.

_SCORE_FIELDS = ("semantic_risk", "context_consistency", "behavioral_trust", "trust_value")


@dataclass
class PageHinkleyResult:
    drift_detected: bool
    cumulative_sum: float
    running_minimum: float


@dataclass
class PageHinkleyDetector:
    """Stateful cumulative-sum detector - fit incrementally via .update(), one value at a time."""

    delta: float = 0.005  # tolerated magnitude of drift per step before it accumulates
    lam: float = 1.0  # alarm threshold on (cumulative_sum - running_minimum); tuned for [0,1]-bounded scores, not raw/unbounded signals
    _count: int = field(default=0, init=False)
    _mean: float = field(default=0.0, init=False)
    _cumulative_sum: float = field(default=0.0, init=False)
    _running_minimum: float = field(default=0.0, init=False)

    def update(self, value: float) -> PageHinkleyResult:
        self._count += 1
        self._mean += (value - self._mean) / self._count
        self._cumulative_sum += value - self._mean - self.delta
        self._running_minimum = min(self._running_minimum, self._cumulative_sum)
        drift = (self._cumulative_sum - self._running_minimum) > self.lam
        return PageHinkleyResult(drift, self._cumulative_sum, self._running_minimum)

    def reset(self) -> None:
        self._count = 0
        self._mean = 0.0
        self._cumulative_sum = 0.0
        self._running_minimum = 0.0


def detect_drift_points(values: Sequence[float], delta: float = 0.005, lam: float = 1.0) -> list[int]:
    """Run a fresh detector over `values`; return indices where an alarm fired.

    Resets the detector after each alarm (standard Page-Hinkley practice) so a
    stream with more than one shift can register more than one drift point.
    """
    detector = PageHinkleyDetector(delta=delta, lam=lam)
    drift_indices: list[int] = []
    for i, value in enumerate(values):
        if detector.update(value).drift_detected:
            drift_indices.append(i)
            detector.reset()
    return drift_indices


def audit_log_drift(path: str | Path, delta: float = 0.005, lam: float = 1.0) -> dict[str, dict[str, list[int]]]:
    """Instantiation 1 (agentic-side): per-agent_id Page-Hinkley over each of
    Rs/Rc/Rb/T from the Trust Boundary audit log. This is a generic "has this
    agent's score pattern shifted" signal only - the audit log has never seen a
    gradual_scaling-style attack (that attack lives only in the FTIL/federated
    pipeline, see audit_log_drift's sibling below), so no attack-detection claim
    is made here, only distributional drift.
    """
    entries_by_agent: dict[str, dict[str, list[float]]] = defaultdict(lambda: {f: [] for f in _SCORE_FIELDS})
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        scores = entry["decision"].get("scores")
        if scores is None:  # rejected at preprocessing/deterministic_controls - no trust score was ever computed
            continue
        agent_id = entry.get("agent_id") or "unknown"
        for f in _SCORE_FIELDS:
            entries_by_agent[agent_id][f].append(scores[f])
    return {agent_id: {f: detect_drift_points(values, delta=delta, lam=lam) for f, values in fields.items()} for agent_id, fields in entries_by_agent.items()}


def ftil_trust_drift(path: str | Path, delta: float = 0.005, lam: float = 1.0) -> dict[str, list[int]]:
    """Instantiation 2 (FTIL-side): per-client_id Page-Hinkley over trust_scores
    across federated rounds, from a separate data source (artifacts/models/ftil_trust_log.jsonl).

    This is the ONLY one of the two instantiations that gets to claim relevance to
    catching a gradual_scaling-style attack (src/tfacd/integrity/attacks.py) -
    that attack grows a malicious client's update deviation a little more each
    round specifically to stay under a single-round threshold, and a cumulative-
    deviation test like Page-Hinkley is well-suited to noticing that kind of slow
    escalation across rounds. This is a structural argument, not a benchmarked
    result: no experiment here measures detection lag/precision against a live
    gradual_scaling run. A client absent from a round (rejected before trust
    scoring) simply contributes no sample for that round, rather than a
    fabricated placeholder value.
    """
    scores_by_client: dict[str, list[float]] = defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        for client_id, score in json.loads(line).get("trust_scores", {}).items():
            scores_by_client[client_id].append(float(score))
    return {client_id: detect_drift_points(values, delta=delta, lam=lam) for client_id, values in scores_by_client.items()}
