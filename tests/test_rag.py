from tfacd.agentic.rag import RAGEngine, SecurityKnowledgeBase
from tfacd.runtime.contracts import IDSAlert


def test_security_knowledge_base_retrieval():
    kb = SecurityKnowledgeBase()
    snippets = kb.retrieve("DoS attack flooding network", top_k=2)
    assert len(snippets) > 0
    assert any("DoS" in s.content or "Denial" in s.title for s in snippets)


def test_rag_engine_alert_context():
    rag = RAGEngine()
    alert = IDSAlert(attack_type="DDoS", confidence=0.92, source_id="192.168.1.100", target_asset="10.0.0.1")
    context = rag.get_context_for_alert(alert)
    assert "Retrieved Security Knowledge" in context
    assert len(context) > 0
