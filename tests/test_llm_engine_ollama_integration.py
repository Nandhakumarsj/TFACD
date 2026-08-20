"""Service-gated integration test: exercises LLMDecisionEngine against a real
local Ollama server, and self-skips when one isn't reachable so `pytest -q`
stays green on machines that never ran scripts/run_llm_engine_benchmark.py.

Same shape as test_streaming_pipeline.py's artifact-gated golden test. Asserts
only invariants the system must hold regardless of what the model says - never
that a local quantized model must produce a particular plan.
"""

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from tfacd.agentic.history import EntityHistory
from tfacd.common.config import load_config
from tfacd.runtime.contracts import IDSAlert, ThreatContext
from tfacd.runtime.threat_context import ThreatContextGenerator

BASE_URL = "http://localhost:11434"


def _ollama_model_available(model: str) -> bool:
    try:
        with urllib.request.urlopen(f"{BASE_URL}/api/tags", timeout=3) as response:
            tags = json.loads(response.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        return False
    return any(entry.get("name") == model for entry in tags.get("models", []))


def _engine_for(model: str, history: EntityHistory):
    from langchain_ollama import ChatOllama

    from tfacd.agentic.llm_engine import LLMDecisionEngine

    chat_model = ChatOllama(model=model, base_url=BASE_URL, temperature=0.0, num_ctx=4096)
    return LLMDecisionEngine(chat_model, history=history, max_attempts=2)


def test_real_ollama_decision_respects_the_allowed_playbook_ceiling():
    config = load_config("configs/edge_iiot.yaml")
    model = config["agentic"]["llm"]["model"]
    if not _ollama_model_available(model):
        pytest.skip(f"Ollama model {model!r} not reachable at {BASE_URL}")

    generator = ThreatContextGenerator(config["runtime"]["threat_context_mapping"])
    alert = IDSAlert(attack_type="DDoS_UDP", confidence=0.93, source_id="203.0.113.7", target_asset="gateway-01")
    context = generator.enrich(alert)

    engine = _engine_for(model, EntityHistory())
    plan = engine.decide(alert, context)

    # Invariants that must hold whatever the model produced - including via the
    # fallback path, which is a legitimate outcome, not a test failure.
    assert plan.engine == "llm" or plan.engine.startswith("fallback:")
    assert plan.actions, "a plan must propose at least one action"
    assert all(a.capability in context.allowed_playbooks for a in plan.actions)
    assert plan.rationale.strip()
    assert 0.0 <= plan.confidence <= 1.0
    assert plan.incident_id.startswith(f"{alert.source_id}-{context.priority}-")


def test_real_ollama_plan_passes_the_full_trust_boundary():
    """The point of the whole design: an LLM-authored plan must survive the same
    unmodified ASTB the deterministic engine's plans go through."""
    config = load_config("configs/edge_iiot.yaml")
    model = config["agentic"]["llm"]["model"]
    if not _ollama_model_available(model):
        pytest.skip(f"Ollama model {model!r} not reachable at {BASE_URL}")

    from datetime import datetime, timezone

    from tfacd.runtime.contracts import SessionContext
    from tfacd.trust_boundary.audit import AuditLogger
    from tfacd.trust_boundary.behavioral_trust import BehavioralTrustEngine
    from tfacd.trust_boundary.boundary import AdaptiveSemanticTrustBoundary
    from tfacd.trust_boundary.dynamic_trust import DynamicTrustScoreRegulator
    from tfacd.trust_boundary.semantic_risk import SemanticRiskEngine

    policy = load_config(config["runtime"]["trust_policy_path"])
    tb_config = config["trust_boundary"]
    history = EntityHistory()

    generator = ThreatContextGenerator(config["runtime"]["threat_context_mapping"])
    alert = IDSAlert(attack_type="Ransomware", confidence=0.96, source_id="198.51.100.4", target_asset="hmi-03")
    context = generator.enrich(alert)
    plan = _engine_for(model, history).decide(alert, context)

    boundary = AdaptiveSemanticTrustBoundary(
        history=history, policy=policy, preprocessing_config=tb_config,
        trust_regulator=DynamicTrustScoreRegulator(
            tb_config["weight_semantic_risk"], tb_config["weight_context_consistency"],
            tb_config["weight_behavioral_trust"], tb_config["trust_level_thresholds"],
        ),
        semantic_risk_engine=SemanticRiskEngine(force_fallback=True),
        behavioral_trust_engine=BehavioralTrustEngine(high_risk_capabilities=set(policy["capability_whitelist"]["high_risk"]), seed=0),
        audit_logger=AuditLogger(Path("artifacts/agentic/ollama_integration_audit.jsonl")),
    )
    session = SessionContext(agent_id="ollama_integration_test", session_id="s", issued_at=datetime.now(timezone.utc), nonce="n")
    decision = boundary.evaluate(plan, context, session)

    assert decision.engine == plan.engine  # provenance survives into the audited decision
    assert set(decision.executed_actions) <= set(context.allowed_playbooks)
