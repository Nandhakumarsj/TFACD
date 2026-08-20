"""LangGraph reason -> validate -> retry loop for the LLM-backed decision engine.

This graph never falls back to the deterministic engine on its own - if `attempt`
reaches `max_attempts` without a valid plan, the final state's `plan` stays None.
`LLMDecisionEngine.decide()` is the only place that reacts to that by invoking the
real `AgenticDecisionEngine` as fallback, keeping "reuse, don't reimplement" literal.

`validate()` here is a cheap fail-fast pre-filter (schema + capability-subset-of-
allowed-list), not a second trust boundary - the real Adaptive Semantic Trust
Boundary (trust_boundary/) still evaluates whatever plan comes out of this graph.
Its only job is to avoid spending a full ASTB evaluation (SBERT + IsolationForest)
on a plan that's structurally broken, and to give the LLM concrete, correctable
feedback on retry instead of a single silent rejection.
"""

from __future__ import annotations

import json
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from tfacd.runtime.contracts import CyberAction, CyberActionPlan, IDSAlert, ThreatContext

SYSTEM_PROMPT = """You are the reasoning component of an industrial IoT cyber-defense \
agent. You are given a structured threat context produced by a certified federated \
intrusion detection model. Your job is to propose a Cyber Action Plan: which of the \
explicitly allowed playbooks to invoke, a target, and a rationale grounded in the \
evidence you were given.

Hard rules, enforced independently downstream regardless of what you output:
- You may only propose capabilities from the "allowed_playbooks" list you are given. \
Proposing anything else will be rejected.
- You do not have to propose every allowed playbook - propose a reasoned subset if a \
smaller response is more appropriate to the evidence. An action you don't propose can \
never execute later, even if the system ends up trusting you highly, so omit an action \
only when you have a real reason to.
- Write a rationale that references the specific evidence you were given (attack type, \
confidence, severity, repeat activity if present) - a generic or off-topic rationale is \
scored independently and will reduce this plan's trust.
- Set confidence to your own calibrated confidence in this specific plan, not a copy of \
the detector's confidence.
- Set each action's target correctly: for an action that acts on the attacker (e.g. \
block_source, rate_limit), target must be source_id - the attacker's address. For an \
action that protects the asset instead (e.g. isolate_segment), target must be \
target_asset. Getting this backwards means the wrong address gets acted on.

Respond only with the requested structured output."""

# Appended only for structured_output_method="json_mode" - Ollama models that
# don't support tool-calling (verified live: gemma3:4b returns HTTP 400 "does
# not support tools" from with_structured_output()'s default "function_calling"
# method) get their schema for free via tool/function metadata; json_mode
# models need the shape spelled out in text instead, since `format: "json"`
# alone only constrains "valid JSON", not "JSON matching this schema".
_JSON_MODE_SCHEMA_SUFFIX = "\n\nRespond with ONLY a JSON object matching this schema, no other text:\n{schema}"


def build_system_prompt(structured_output_method: str = "function_calling") -> str:
    """Shared by _reason_node here and benchmark.py::measure_tokens_per_second -
    both need the exact same prompt shape a real decision would use, not two
    copies that can drift apart."""
    if structured_output_method == "json_mode":
        return SYSTEM_PROMPT + _JSON_MODE_SCHEMA_SUFFIX.format(schema=json.dumps(LLMProposedPlan.model_json_schema()))
    return SYSTEM_PROMPT


class LLMProposedAction(BaseModel):
    capability: str
    target: str | None = None
    # None is accepted here but NOT in contracts.py's CyberAction.parameters -
    # a local model routinely emits {"target": null} for a parameter it chose not
    # to set, and rejecting the whole plan over that would burn a retry on a
    # non-issue. Null-valued entries are dropped when converting to CyberAction
    # (see _validate_node), so the canonical contract stays strict.
    parameters: dict[str, str | int | float | bool | None] = {}


class LLMProposedPlan(BaseModel):
    rationale: str
    actions: list[LLMProposedAction]
    confidence: float = Field(ge=0.0, le=1.0)


class DecisionState(TypedDict):
    alert: IDSAlert
    context: ThreatContext
    repeat_activity: bool
    repeat_window_minutes: int
    attempt: int
    max_attempts: int
    proposed: LLMProposedPlan | None
    validation_errors: list[str]
    plan: CyberActionPlan | None


def initial_state(
    alert: IDSAlert, context: ThreatContext, *, repeat_activity: bool, repeat_window_minutes: int, max_attempts: int
) -> DecisionState:
    return DecisionState(
        alert=alert, context=context, repeat_activity=repeat_activity, repeat_window_minutes=repeat_window_minutes,
        attempt=0, max_attempts=max_attempts, proposed=None, validation_errors=[], plan=None,
    )


def build_human_prompt(state: DecisionState) -> str:
    alert, context = state["alert"], state["context"]
    lines = [
        f"attack_type: {alert.attack_type}",
        f"detector_confidence: {alert.confidence:.3f}",
        f"source_id: {alert.source_id or 'unknown'}",
        f"target_asset: {alert.target_asset or 'unknown'}",
        f"severity: {context.severity}",
        f"priority: {context.priority}",
        f"mitre_techniques: {context.mitre_techniques or 'none listed'}",
        f"allowed_playbooks: {context.allowed_playbooks}",
    ]
    if state["repeat_activity"]:
        lines.append(f"repeat_activity: this source was seen again within the last {state['repeat_window_minutes']} minutes")
    if state["validation_errors"]:
        lines.append(
            "Your previous proposal was rejected for: " + "; ".join(state["validation_errors"]) + ". Correct these and propose again."
        )
    return "\n".join(lines)


def _reason_node(chat_model, structured_output_method: str = "function_calling"):
    if structured_output_method == "json_mode":
        # Caller (factory.py/benchmark.py) is responsible for constructing
        # chat_model with format="json" set - that's a ChatOllama constructor
        # arg, not something this function can set on an already-built model.
        structured_model = chat_model.with_structured_output(LLMProposedPlan, method="json_mode")
    else:
        structured_model = chat_model.with_structured_output(LLMProposedPlan)
    system_prompt = build_system_prompt(structured_output_method)

    def reason(state: DecisionState) -> dict[str, Any]:
        messages = [SystemMessage(system_prompt), HumanMessage(build_human_prompt(state))]
        try:
            result = structured_model.invoke(messages)
            proposed = LLMProposedPlan.model_validate(result) if isinstance(result, dict) else result
            return {"proposed": proposed, "attempt": state["attempt"] + 1, "validation_errors": []}
        except Exception as exc:
            # The chat model/output-parser can fail in many library-specific ways
            # (connection error, malformed JSON, schema mismatch) - all of them are
            # a retryable validation failure here, never an uncaught crash.
            return {"proposed": None, "attempt": state["attempt"] + 1, "validation_errors": [f"LLM call failed: {exc}"]}

    return reason


def _validate_node(state: DecisionState) -> dict[str, Any]:
    proposed = state["proposed"]
    if proposed is None:
        return {"validation_errors": state["validation_errors"] or ["no proposal produced"]}

    errors: list[str] = []
    allowed = set(state["context"].allowed_playbooks)
    disallowed = {a.capability for a in proposed.actions} - allowed
    if disallowed:
        errors.append(f"capabilities not in allowed_playbooks: {sorted(disallowed)} (allowed: {sorted(allowed)})")
    if not proposed.actions:
        errors.append("plan has no actions - propose at least one of the allowed playbooks")
    if not proposed.rationale.strip():
        errors.append("rationale is empty")

    if errors:
        return {"validation_errors": errors}

    plan = CyberActionPlan(
        incident_id="pending",  # LLMDecisionEngine.decide() assigns the real incident_id, matching AgenticDecisionEngine's convention
        rationale=proposed.rationale,
        actions=[
            CyberAction(capability=a.capability, target=a.target, parameters={k: v for k, v in a.parameters.items() if v is not None})
            for a in proposed.actions
        ],
        confidence=proposed.confidence,
        engine="llm",
    )
    return {"plan": plan, "validation_errors": []}


def _route(state: DecisionState) -> str:
    if state["plan"] is not None:
        return "end"
    if state["attempt"] < state["max_attempts"]:
        return "retry"
    return "end"


def build_decision_graph(chat_model, max_attempts: int = 2, structured_output_method: str = "function_calling"):
    graph = StateGraph(DecisionState)
    graph.add_node("reason", _reason_node(chat_model, structured_output_method))
    graph.add_node("validate", _validate_node)
    graph.add_edge(START, "reason")
    graph.add_edge("reason", "validate")
    graph.add_conditional_edges("validate", _route, {"retry": "reason", "end": END})
    return graph.compile()
