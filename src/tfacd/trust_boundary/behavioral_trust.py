from __future__ import annotations

from datetime import timedelta

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
