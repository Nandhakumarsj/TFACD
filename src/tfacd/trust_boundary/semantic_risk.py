from __future__ import annotations

from pathlib import Path

import numpy as np

from tfacd.agentic.decision_engine import RATIONALE_TEMPLATES
from tfacd.runtime.contracts import CyberActionPlan, ThreatContext

_DEFAULT_THREAT_CONTEXT_PATH = "configs/threat_context.yaml"

# A real LLM-authored rationale describes an action in prose ("blocking the
# source", "rate limiting the traffic"), never the raw snake_case capability
# identifier ("block_source", "rate_limit") - TfidfVectorizer tokenizes these
# as completely unrelated features, so vocabulary rendered only from
# threat_context.yaml (which uses the identifiers) still leaves the fallback
# scoring naturally-phrased action descriptions as off-vocabulary. This bridges
# the gap for the fixed, known set of capabilities in trust_policy.yaml.
_CAPABILITY_PHRASES = {
    "observe": "observing the activity",
    "log_event": "logging the event",
    "increase_logging": "increasing logging and monitoring",
    "create_ticket": "creating a ticket for follow-up",
    "notify_soc": "notifying the security operations center SOC team",
    "start_capture": "starting a packet capture",
    "rate_limit": "applying rate limiting",
    "block_source": "blocking the source",
    "isolate_segment": "isolating the network segment",
    "rotate_session": "rotating session credentials",
}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _rendered_vocabulary_examples(threat_context_path: str | Path) -> list[str]:
    """Expands the TF-IDF fallback's vocabulary beyond the 5 raw templates'
    literal placeholder syntax (e.g. "{attack_type}"), which never contains a
    real attack-type/playbook word - fits on real RENDERED examples for every
    class in threat_context.yaml instead, so words like "DDoS_UDP" or
    "isolate_segment" actually exist in vocabulary. Measured live: this closes
    a spurious ~0.485 Rs penalty a naturally-phrased (LLM-style), genuinely
    on-topic rationale previously took purely from vocabulary mismatch, not
    real topical drift (verified during an architecture audit). Returns []
    (degrade, don't crash) if the file can't be read - this is already a
    fallback-of-a-fallback path (no SBERT AND no threat-context file), and a
    missing optional enrichment source shouldn't take down scoring entirely.
    """
    try:
        import yaml

        mapping = yaml.safe_load(Path(threat_context_path).read_text(encoding="utf-8"))
    except Exception:
        return []
    examples = []
    for attack_type, entry in mapping.items():
        template = RATIONALE_TEMPLATES.get(entry.get("severity"))
        if template is None:
            continue
        playbooks = entry.get("allowed_playbooks") or []
        examples.append(
            template.format(
                attack_type=attack_type, source_id="10.0.0.1", target_asset="asset-01",
                playbooks=", ".join(playbooks) or "no playbooks",
            )
        )
        # A second, natural-language-phrased example for the same scenario -
        # see _CAPABILITY_PHRASES above for why the identifier-only rendering
        # above isn't sufficient on its own.
        phrases = [_CAPABILITY_PHRASES.get(p, p) for p in playbooks]
        if phrases:
            examples.append(
                f"Detected {attack_type} activity from the source targeting the asset; "
                f"recommending {', '.join(phrases)} to contain the threat."
            )
    return examples


class SemanticRiskEngine:
    """Rs in [0,1]: how semantically off-topic a plan's rationale is relative to
    the expected rationale for its threat context's severity - catches a decision
    engine producing a hallucinated/off-task rationale for the actual incident.

    Real Sentence-BERT (all-MiniLM-L6-v2, CPU - GPU stays dedicated to IDS
    training/inference) by default, falling back to TF-IDF cosine similarity if
    the model can't be loaded (no network, blocked, etc.). The load failure is
    cached so a no-network environment doesn't retry - and hang - on every call.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", force_fallback: bool = False, threat_context_path: str | Path = _DEFAULT_THREAT_CONTEXT_PATH):
        self.model_name = model_name
        self.threat_context_path = threat_context_path
        self._model = None
        self._model_load_failed = force_fallback
        self._fallback_vectorizer = None
        if force_fallback:
            self._build_fallback()

    def _build_fallback(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Fit once over the full corpus, not per pairwise comparison - a
        # 2-document fit makes IDF weighting meaningless. Words in a plan's
        # rationale outside this vocabulary are dropped by transform(); this is
        # a known approximation, adequate to catch gross topic mismatches.
        corpus = list(RATIONALE_TEMPLATES.values()) + _rendered_vocabulary_examples(self.threat_context_path)
        self._fallback_vectorizer = TfidfVectorizer().fit(corpus)

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
            # TF-IDF cosine similarity is a bag-of-words comparison - it cannot
            # recognize that a differently-phrased rationale is still on-topic
            # the way SBERT's real semantic similarity can (measured live: even
            # after expanding the fit vocabulary above, a genuinely on-topic,
            # naturally-phrased rationale still scored similarity ~0.31-0.51
            # against the template purely from word-choice differences, not
            # topical drift). Rather than chase paraphrase-invariance through
            # ever more corpus tuning - a losing battle for a bag-of-words
            # model - floor the score on the one distinctive, rare signal
            # TF-IDF CAN reliably detect: does the rationale name the actual
            # attack type? A floor, not an override, so this never LOWERS a
            # score TF-IDF already rated as more similar on its own.
            attack_type_words = context.alert.attack_type.lower().replace("_", " ")
            if attack_type_words in plan.rationale.lower().replace("_", " "):
                similarity = max(similarity, 0.6)
        return 1.0 - max(0.0, min(1.0, similarity))
