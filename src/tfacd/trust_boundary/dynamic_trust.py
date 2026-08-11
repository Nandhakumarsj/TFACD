from __future__ import annotations

from tfacd.runtime.contracts import TrustScores

_AUTONOMY_MODES = {
    "low": "read_only",
    "medium": "recommendation",
    "high": "restricted_action",
    "verified": "autonomous_execution",
}


class DynamicTrustScoreRegulator:
    """Combines Rs/Rc/Rb into T = ws*(1-Rs) + wc*Rc + wb*Rb, buckets T into a
    Trust Level via configured thresholds, and maps Trust Level to an autonomy
    mode. Weight-sum validation mirrors aggregation.weighted_average's
    non-negative/positive-sum check in the FTIL layer.
    """

    def __init__(self, weight_semantic_risk: float, weight_context_consistency: float, weight_behavioral_trust: float, thresholds: dict[str, float]):
        total = weight_semantic_risk + weight_context_consistency + weight_behavioral_trust
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"trust weights must sum to 1.0, got {total}")
        self.ws = weight_semantic_risk
        self.wc = weight_context_consistency
        self.wb = weight_behavioral_trust
        self.thresholds = thresholds

    def evaluate(self, semantic_risk: float, context_consistency: float, behavioral_trust: float) -> TrustScores:
        trust_value = self.ws * (1.0 - semantic_risk) + self.wc * context_consistency + self.wb * behavioral_trust
        return TrustScores(
            semantic_risk=semantic_risk,
            context_consistency=context_consistency,
            behavioral_trust=behavioral_trust,
            trust_value=trust_value,
        )

    def trust_level(self, trust_value: float) -> str:
        if trust_value < self.thresholds["low"]:
            return "low"
        if trust_value < self.thresholds["medium"]:
            return "medium"
        if trust_value < self.thresholds["high"]:
            return "high"
        return "verified"

    def autonomy_mode(self, trust_level: str) -> str:
        return _AUTONOMY_MODES[trust_level]
