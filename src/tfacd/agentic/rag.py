"""Security Retrieval-Augmented Generation (RAG) Engine.

Indexes MITRE ATT&CK for ICS techniques, IIoT threat guidelines, and playbook
remediation knowledge to enrich the LLM decision engine's context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tfacd.runtime.contracts import IDSAlert

logger = logging.getLogger(__name__)

# Core Knowledge Corpus for Industrial Cyber Defense (ICS/IIoT)
DEFAULT_KNOWLEDGE_CORPUS = [
    {
        "id": "MITRE_T0814",
        "title": "Denial of Service - Wireless / Network Flooding (T0814)",
        "content": "DoS attacks in IIoT flood network channels or Modbus/MQTT brokers with illegitimate packets, causing high latency or dropped control signals. Remediation: rate-limiting, blocking source IP, and applying network microsegmentation.",
    },
    {
        "id": "MITRE_T0855",
        "title": "Unauthorized Command Message / Command Injection (T0855)",
        "content": "Adversaries send unauthorized control commands to PLCs or IIoT gateways (e.g., Modbus write single coil / register). Remediation: isolate affected gateway, revoke unauthorized session nonces, enforce strict action whitelisting.",
    },
    {
        "id": "MITRE_T0866",
        "title": "Exploitation of Remote Services / Backdoor (T0866)",
        "content": "Exploiting exposed services (SSH, web management interface, OPC-UA) to establish persistence or pivot inside OT network. Remediation: block source IP, terminate active sessions, trigger credential rotation.",
    },
    {
        "id": "MITRE_T0831",
        "title": "Manipulation of Control / False Data Injection (T0831)",
        "content": "Attackers tamper with sensor telemetry or process variables (FDI attack) to spoof operational status. Remediation: isolate telemetry source, enable deep packet inspection, alert SOC for manual process verification.",
    },
    {
        "id": "IIOT_PLAYBOOK_GUIDELINES",
        "title": "IIoT Incident Response Principles",
        "content": "High severity events require immediate containment (block_source or isolate_host). Medium severity events recommend targeted observation and rate limits (apply_qos_limit, alert_soc). Low/informational events require audit logging only.",
    },
]


@dataclass
class KnowledgeSnippet:
    doc_id: str
    title: str
    content: str
    similarity_score: float


class SecurityKnowledgeBase:
    """TF-IDF indexed security knowledge base."""

    def __init__(self, documents: Sequence[dict] | None = None):
        self.documents = list(documents or DEFAULT_KNOWLEDGE_CORPUS)
        self.corpus = [f"{d['title']} {d['content']}" for d in self.documents]
        self.vectorizer = TfidfVectorizer().fit(self.corpus)
        self.doc_vectors = self.vectorizer.transform(self.corpus)

    def retrieve(self, query: str, top_k: int = 2) -> list[KnowledgeSnippet]:
        if not query.strip():
            return []
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.doc_vectors)[0]
        ranked_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            score = float(similarities[idx])
            if score > 0.05:
                doc = self.documents[idx]
                results.append(
                    KnowledgeSnippet(
                        doc_id=doc["id"],
                        title=doc["title"],
                        content=doc["content"],
                        similarity_score=score,
                    )
                )
        return results


class RAGEngine:
    """High-level RAG engine retrieving contextual snippets for IDS alerts."""

    def __init__(self, kb: SecurityKnowledgeBase | None = None):
        self.kb = kb or SecurityKnowledgeBase()

    def get_context_for_alert(self, alert: IDSAlert, top_k: int = 2) -> str:
        query = f"{alert.attack_type} attack from {alert.source_id or 'unknown'} targeting {alert.target_asset or 'unknown'}"
        snippets = self.kb.retrieve(query, top_k=top_k)
        if not snippets:
            return ""
        formatted = ["### Retrieved Security Knowledge & Playbook Rules:"]
        for s in snippets:
            formatted.append(f"- **[{s.doc_id}] {s.title}**: {s.content}")
        return "\n".join(formatted)
