from __future__ import annotations

import numpy as np
import shap
from lime.lime_text import LimeTextExplainer

from tfacd.agentic.history import EntityHistory
from tfacd.runtime.contracts import CyberActionPlan, SessionContext, ThreatContext
from tfacd.trust_boundary.behavioral_trust import _FEATURE_ORDER, BehavioralTrustEngine, _feature_vector
from tfacd.trust_boundary.semantic_risk import SemanticRiskEngine

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
    shap_values = shap.TreeExplainer(engine._forest).shap_values(vector.reshape(1, -1))
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

    explainer = LimeTextExplainer(class_names=["low_risk", "high_risk"])
    explanation = explainer.explain_instance(plan.rationale, classifier_fn, labels=(1,), num_features=num_features, num_samples=num_samples)
    return explanation.as_list(label=1)
