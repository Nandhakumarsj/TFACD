from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

from tfacd.agentic.decision_engine import AgenticDecisionEngine
from tfacd.agentic.graph import build_decision_graph, initial_state
from tfacd.agentic.history import EntityHistory
from tfacd.runtime.contracts import CyberActionPlan, IDSAlert, ThreatContext


class LLMDecisionEngine:
    """DecisionEngine-conforming wrapper around the LangGraph reason/validate/retry
    loop (agentic/graph.py). Falls back to the deterministic AgenticDecisionEngine -
    reused, not reimplemented - whenever the graph can't produce a schema-valid,
    whitelist-respecting plan within max_attempts, or the chat model itself errors
    (e.g. Ollama unreachable). The returned plan's `engine` field records which path
    actually produced it ("llm", "llm:fallback_model", or "fallback:<reason>") for
    audit provenance - see runtime/contracts.py.

    An optional second (fallback) model is a MODEL-level fallback, distinct from
    fallback_engine (the ENGINE-level fallback: llm -> deterministic template). If
    fallback_chat_model is given, a primary-model failure is retried against it
    using the exact same build_decision_graph() reason/validate/retry logic -
    reused, not duplicated - before dropping to the deterministic engine.

    The fallback path never reimplements the deterministic engine's whitelist safety:
    it calls AgenticDecisionEngine.decide(), which only ever proposes capabilities
    already in context.allowed_playbooks - so whatever the LLM tried to propose is
    irrelevant once fallback triggers.
    """

    def __init__(
        self,
        chat_model,
        history: EntityHistory | None = None,
        fallback_engine: AgenticDecisionEngine | None = None,
        fallback_chat_model=None,
        max_attempts: int = 2,
        repeat_window_minutes: int = 30,
        structured_output_method: str = "function_calling",
    ):
        self.history = history or EntityHistory()
        self.fallback_engine = fallback_engine or AgenticDecisionEngine(history=self.history)
        self.max_attempts = max_attempts
        self.repeat_window = timedelta(minutes=repeat_window_minutes)
        self.repeat_window_minutes = repeat_window_minutes
        self._graph = build_decision_graph(chat_model, max_attempts=max_attempts, structured_output_method=structured_output_method)
        self._fallback_model_graph = (
            build_decision_graph(fallback_chat_model, max_attempts=max_attempts, structured_output_method=structured_output_method)
            if fallback_chat_model is not None
            else None
        )

    def decide(self, alert: IDSAlert, context: ThreatContext) -> CyberActionPlan:
        source = alert.source_id or "unknown-source"
        prior_incidents = self.history.recent(source, kind="incident", within=self.repeat_window)
        repeat_activity = len(prior_incidents) > 0

        state = initial_state(
            alert, context, repeat_activity=repeat_activity, repeat_window_minutes=self.repeat_window_minutes, max_attempts=self.max_attempts,
        )

        plan, reason = self._invoke_graph(self._graph, state)
        engine_label = "llm"
        if plan is None and self._fallback_model_graph is not None:
            fallback_plan, fallback_reason = self._invoke_graph(self._fallback_model_graph, state)
            if fallback_plan is not None:
                plan, engine_label = fallback_plan, "llm:fallback_model"
            else:
                reason = f"primary:{reason}; fallback_model:{fallback_reason}"

        if plan is None:
            return self._fallback(alert, context, reason=reason)

        incident_id = f"{source}-{context.priority}-{uuid4().hex[:8]}"
        plan = plan.model_copy(update={"incident_id": incident_id, "engine": engine_label})
        self.history.append(
            source, "incident",
            {"attack_type": alert.attack_type, "target_asset": alert.target_asset, "severity": context.severity, "incident_id": incident_id},
        )
        return plan

    def _invoke_graph(self, graph, state) -> tuple[CyberActionPlan | None, str | None]:
        """Runs one compiled reason/validate/retry graph to completion. Returns
        (plan, error_reason) - error_reason is None iff plan is not None."""
        try:
            result = graph.invoke(state)
        except Exception as exc:
            return None, f"error:{exc}"
        plan = result.get("plan")
        if plan is None:
            errors = "; ".join(result.get("validation_errors", [])) or "exhausted retries"
            return None, f"exhausted:{errors}"
        return plan, None

    def _fallback(self, alert: IDSAlert, context: ThreatContext, *, reason: str) -> CyberActionPlan:
        plan = self.fallback_engine.decide(alert, context)
        return plan.model_copy(update={"engine": f"fallback:{reason}"})
