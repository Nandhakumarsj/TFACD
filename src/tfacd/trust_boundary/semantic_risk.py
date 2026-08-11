from __future__ import annotations

import numpy as np

from tfacd.agentic.decision_engine import RATIONALE_TEMPLATES
from tfacd.runtime.contracts import CyberActionPlan, ThreatContext


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class SemanticRiskEngine:
    """Rs in [0,1]: how semantically off-topic a plan's rationale is relative to
    the expected rationale for its threat context's severity - catches a decision
    engine producing a hallucinated/off-task rationale for the actual incident.

    Real Sentence-BERT (all-MiniLM-L6-v2, CPU - GPU stays dedicated to IDS
    training/inference) by default, falling back to TF-IDF cosine similarity if
    the model can't be loaded (no network, blocked, etc.). The load failure is
    cached so a no-network environment doesn't retry - and hang - on every call.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", force_fallback: bool = False):
        self.model_name = model_name
        self._model = None
        self._model_load_failed = force_fallback
        self._fallback_vectorizer = None
        if force_fallback:
            self._build_fallback()

    def _build_fallback(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Fit once over the full template corpus, not per pairwise comparison -
        # a 2-document fit makes IDF weighting meaningless. Words in a plan's
        # rationale outside this small fixed vocabulary are dropped by transform();
        # this is a known approximation, adequate to catch gross topic mismatches.
        self._fallback_vectorizer = TfidfVectorizer().fit(list(RATIONALE_TEMPLATES.values()))

    def _get_model(self):
        if self._model is not None or self._model_load_failed:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        except Exception:
            self._model_load_failed = True
            self._build_fallback()
        return self._model

    def score(self, plan: CyberActionPlan, context: ThreatContext) -> float:
        template = RATIONALE_TEMPLATES[context.severity].format(
            attack_type=context.alert.attack_type,
            source_id=context.alert.source_id or "unknown-source",
            target_asset=context.alert.target_asset or "unknown-asset",
            playbooks=", ".join(context.allowed_playbooks) or "no playbooks",
        )
        model = self._get_model()
        if model is not None:
            embeddings = model.encode([plan.rationale, template])
            similarity = _cosine(embeddings[0], embeddings[1])
        else:
            vectors = self._fallback_vectorizer.transform([plan.rationale, template]).toarray()
            similarity = _cosine(vectors[0], vectors[1])
        return 1.0 - max(0.0, min(1.0, similarity))
