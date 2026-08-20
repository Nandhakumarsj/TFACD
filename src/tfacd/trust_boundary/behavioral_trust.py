from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
from sklearn.ensemble import IsolationForest

from tfacd.agentic.history import EntityHistory
from tfacd.runtime.contracts import CyberActionPlan, SessionContext

_POPULATION_SIZE = 40
# Feature order must match between the synthetic population and every scored
# vector - IsolationForest.fit()/.decision_function() are silently wrong if the
# column order drifts between the two, same discipline as vectorize.py's
# sorted-key flattening in the FTIL layer.
_FEATURE_ORDER = ("high_risk_fraction", "action_count", "violation_rate", "recent_event_count")


def _feature_vector(session: SessionContext, plan: CyberActionPlan, history: EntityHistory, high_risk_capabilities: set[str]) -> np.ndarray:
    capabilities = sorted(a.capability for a in plan.actions)
    high_risk_fraction = sum(1 for c in capabilities if c in high_risk_capabilities) / len(capabilities) if capabilities else 0.0
    recent = history.recent(session.agent_id, kind="trust_decision", within=timedelta(hours=1))
    violation_rate = sum(1 for e in recent if e["payload"].get("policy_violation")) / len(recent) if recent else 0.0
    return np.array([high_risk_fraction, len(capabilities), violation_rate, len(recent)], dtype=np.float64)


class BehavioralTrustEngine:
    """Rb in [0,1]: how consistent this plan/entity's behavior is with an
    expected-benign population (synthetic, cold-start - no real fleet of agents
    to learn "normal" from yet), smoothed across interactions via the same
    dict[str, float] + EMA idiom PCAClusterEMAFilter uses in the FTIL layer.
    """

    def __init__(self, high_risk_capabilities: set[str], ema_alpha: float = 0.3, seed: int = 0):
        self.high_risk_capabilities = high_risk_capabilities
        self.ema_alpha = ema_alpha
        self.trust_ema: dict[str, float] = {}
        rng = np.random.default_rng(seed)
        population = np.column_stack(
            [
                np.clip(rng.normal(0.1, 0.1, _POPULATION_SIZE), 0.0, 1.0),  # high_risk_fraction
                np.clip(rng.normal(1.5, 1.0, _POPULATION_SIZE), 0, None),  # action_count
                np.clip(rng.normal(0.02, 0.05, _POPULATION_SIZE), 0.0, 1.0),  # violation_rate
                np.clip(rng.normal(3, 2, _POPULATION_SIZE), 0, None),  # recent_event_count
            ]
        )
        self._forest = IsolationForest(n_estimators=100, random_state=seed, contamination="auto").fit(population)

    def score(self, session: SessionContext, plan: CyberActionPlan, history: EntityHistory) -> float:
        features = _feature_vector(session, plan, history, self.high_risk_capabilities)
        raw = float(self._forest.decision_function(features.reshape(1, -1))[0])  # higher = more normal
        round_score = 1.0 / (1.0 + np.exp(-4.0 * raw))  # squash to (0,1)

        previous = self.trust_ema.get(session.agent_id, 1.0)
        updated = self.ema_alpha * round_score + (1.0 - self.ema_alpha) * previous
        self.trust_ema[session.agent_id] = updated
        return updated

    def refit_from_history(self, history: EntityHistory, min_samples: int = 20) -> bool:
        """Replaces the synthetic cold-start population with one reconstructed
        from REAL observed trust_decision events, once enough real data exists.
        Unsupervised - does not need "was this decision wrong" ground truth
        (which doesn't exist anywhere in this repo, see feedback_loop.py's own
        docstring); it only needs "what does typical observed behavior actually
        look like now", which trust history already answers. Returns False
        (no-op, population unchanged) if fewer than min_samples real
        trust_decision events exist across ALL entities - refitting an anomaly
        detector on a handful of points would replace a documented-synthetic
        population with an undocumented-noisy one, not a real improvement.

        Deliberately not called anywhere in the live pipeline (boundary.py never
        calls this) - exposed as an explicit, operator-invoked capability via
        scripts/refit_behavioral_trust.py, the same "tested library capability,
        not something a script silently triggers mid-run" posture as
        run_threshold_optimizer.py for the FL-side detector.

        Reconstructs each event's feature vector EXACTLY, not approximately:
        high_risk_fraction/action_count come directly from that event's own
        stored "capabilities" list (persisted by boundary.py's _finalize());
        violation_rate/recent_event_count are recomputed the same way
        _feature_vector() computes them live - a rolling 1-hour window of that
        SAME entity's OTHER events strictly before this one - by replaying
        history chronologically per entity, not read as a shortcut aggregate.
        """
        by_entity: dict[str, list[dict]] = {}
        for event in history.all_events(kind="trust_decision"):
            by_entity.setdefault(event["entity_id"], []).append(event)

        rows: list[list[float]] = []
        for events in by_entity.values():
            events = sorted(events, key=lambda e: e["timestamp"])
            for index, event in enumerate(events):
                capabilities = event["payload"].get("capabilities") or []
                high_risk_fraction = sum(1 for c in capabilities if c in self.high_risk_capabilities) / len(capabilities) if capabilities else 0.0
                event_time = datetime.fromisoformat(event["timestamp"])
                window_start = event_time - timedelta(hours=1)
                prior = [e for e in events[:index] if datetime.fromisoformat(e["timestamp"]) >= window_start]
                violation_rate = sum(1 for e in prior if e["payload"].get("policy_violation")) / len(prior) if prior else 0.0
                rows.append([high_risk_fraction, float(len(capabilities)), violation_rate, float(len(prior))])

        if len(rows) < min_samples:
            return False

        population = np.array(rows, dtype=np.float64)
        self._forest = IsolationForest(n_estimators=100, random_state=0, contamination="auto").fit(population)
        return True
