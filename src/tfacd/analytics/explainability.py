from __future__ import annotations

import logging

import numpy as np

try:
    import shap as _shap  # optional – only needed by explain_behavioral_trust
    _HAS_SHAP = True
except ModuleNotFoundError:
    _shap = None  # type: ignore[assignment]
    _HAS_SHAP = False

try:
    from lime.lime_text import LimeTextExplainer as _LimeTextExplainer  # optional
    _HAS_LIME = True
except ModuleNotFoundError:
    _LimeTextExplainer = None  # type: ignore[assignment]
    _HAS_LIME = False

from tfacd.agentic.history import EntityHistory
from tfacd.runtime.contracts import CyberActionPlan, SessionContext, ThreatContext
from tfacd.trust_boundary.behavioral_trust import _FEATURE_ORDER, BehavioralTrustEngine, _feature_vector
from tfacd.trust_boundary.semantic_risk import SemanticRiskEngine

logger = logging.getLogger(__name__)

# Flat functions, not a stateful class - both explainers borrow an already-fitted
# engine (BehavioralTrustEngine._forest, SemanticRiskEngine's live similarity path)
# for a single prediction and hold no state of their own between calls.


def explain_behavioral_trust(engine: BehavioralTrustEngine, session: SessionContext, plan: CyberActionPlan, history: EntityHistory) -> dict[str, float]:
    """SHAP attribution of each Rb feature for one scored plan - which of the four
    features pushed the IsolationForest's anomaly score up or down. Reuses
    engine._forest as fit at construction; never refits, and ignores/does not
    touch engine.trust_ema (that EMA smoothing happens after this raw score).
    """
    vector = _feature_vector(session, plan, history, engine.high_risk_capabilities)
    if not _HAS_SHAP:
        # Fallback: uniform heuristic attribution (feature vector values normalised)
        logger.warning("shap not installed – returning uniform feature-vector attributions")
        total = float(np.abs(vector).sum()) or 1.0
        return {name: float(v / total) for name, v in zip(_FEATURE_ORDER, vector)}
    shap_values = _shap.TreeExplainer(engine._forest).shap_values(vector.reshape(1, -1))
    attributions = np.asarray(shap_values).reshape(-1)
    return {name: float(value) for name, value in zip(_FEATURE_ORDER, attributions)}


def explain_semantic_risk(
    engine: SemanticRiskEngine, plan: CyberActionPlan, context: ThreatContext, num_features: int = 8, num_samples: int = 500
) -> list[tuple[str, float]]:
    """LIME attribution of which words in plan.rationale push Rs up/down.

    Treats SemanticRiskEngine.score as a black box (real SBERT or TF-IDF fallback,
    whichever this engine instance is actually running) via the standard LIME
    pseudo-classifier trick: wrap the continuous Rs in a 2-column
    [1-Rs, Rs] array so LimeTextExplainer's classification-shaped API can probe
    it by perturbing plan.rationale. incident_id/actions/confidence are held
    fixed across perturbations; only the rationale text varies.
    """

    def classifier_fn(texts: list[str]) -> np.ndarray:
        risk = np.array(
            [
                engine.score(
                    CyberActionPlan(incident_id=plan.incident_id, rationale=text, confidence=plan.confidence, actions=plan.actions),
                    context,
                )
                for text in texts
            ]
        )
        return np.column_stack([1.0 - risk, risk])

    if not _HAS_LIME:
        # Fallback: return top-n words scored by plain Rs difference
        logger.warning("lime not installed – returning heuristic word-level attributions")
        words = list(dict.fromkeys(plan.rationale.split()))[:num_features]
        base_risk = engine.score(plan, context)
        result = []
        for w in words:
            masked = plan.rationale.replace(w, "")
            masked_plan = CyberActionPlan(incident_id=plan.incident_id, rationale=masked, confidence=plan.confidence, actions=plan.actions)
            delta = base_risk - engine.score(masked_plan, context)
            result.append((w, float(delta)))
        return sorted(result, key=lambda x: abs(x[1]), reverse=True)[:num_features]
    explainer = _LimeTextExplainer(class_names=["low_risk", "high_risk"])
    explanation = explainer.explain_instance(plan.rationale, classifier_fn, labels=(1,), num_features=num_features, num_samples=num_samples)
    return explanation.as_list(label=1)
