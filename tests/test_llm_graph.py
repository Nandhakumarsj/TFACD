from tfacd.agentic.graph import LLMProposedAction, LLMProposedPlan, build_decision_graph, initial_state
from tfacd.runtime.contracts import IDSAlert, ThreatContext


def make_context(playbooks=None):
    alert = IDSAlert(attack_type="Port_Scanning", confidence=0.7, source_id="10.0.0.5", target_asset="plc-01")
    context = ThreatContext(
        alert=alert, severity="medium", priority="P2", mitre_techniques=[],
        allowed_playbooks=playbooks or ["block_source", "increase_logging", "create_ticket"],
    )
    return alert, context


class _FakeStructuredRunnable:
    """Faithful fake of what `chat_model.with_structured_output(Schema).invoke(messages)`
    returns: a list of pre-scripted responses (LLMProposedPlan instances or exceptions
    to raise), consumed one per call. Records every call's messages for retry-feedback
    assertions."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.invocations: list[list] = []

    def invoke(self, messages):
        self.invocations.append(messages)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _FakeChatModel:
    def __init__(self, responses):
        self.runnable = _FakeStructuredRunnable(responses)
        self.with_structured_output_calls: list[dict] = []

    def with_structured_output(self, schema, **kwargs):
        self.with_structured_output_calls.append(kwargs)
        return self.runnable


def run_graph(chat_model, alert, context, max_attempts=2):
    graph = build_decision_graph(chat_model, max_attempts=max_attempts)
    state = initial_state(alert, context, repeat_activity=False, repeat_window_minutes=30, max_attempts=max_attempts)
    return graph.invoke(state)


def test_valid_first_attempt_produces_plan():
    alert, context = make_context()
    proposal = LLMProposedPlan(
        rationale="port scanning observed, blocking source", actions=[LLMProposedAction(capability="block_source", target="plc-01")], confidence=0.8
    )
    chat_model = _FakeChatModel([proposal])

    result = run_graph(chat_model, alert, context)

    assert result["plan"] is not None
    assert result["plan"].engine == "llm"
    assert [a.capability for a in result["plan"].actions] == ["block_source"]
    assert result["attempt"] == 1


def test_invalid_then_valid_on_retry_carries_error_feedback():
    alert, context = make_context()
    bad = LLMProposedPlan(rationale="x", actions=[LLMProposedAction(capability="shutdown_plant")], confidence=0.5)
    good = LLMProposedPlan(rationale="corrected", actions=[LLMProposedAction(capability="block_source")], confidence=0.5)
    chat_model = _FakeChatModel([bad, good])

    result = run_graph(chat_model, alert, context, max_attempts=2)

    assert result["plan"] is not None
    assert result["attempt"] == 2
    second_call_messages = chat_model.runnable.invocations[1]
    assert any("shutdown_plant" in str(m.content) for m in second_call_messages)


def test_always_invalid_exhausts_attempts_without_plan():
    alert, context = make_context()
    bad = LLMProposedPlan(rationale="x", actions=[LLMProposedAction(capability="shutdown_plant")], confidence=0.5)
    chat_model = _FakeChatModel([bad, bad])

    result = run_graph(chat_model, alert, context, max_attempts=2)

    assert result["plan"] is None
    assert result["attempt"] == 2
    assert chat_model.runnable._responses == []  # both scripted attempts were actually consumed


def test_subset_of_allowed_playbooks_is_accepted():
    alert, context = make_context(playbooks=["block_source", "increase_logging", "create_ticket", "notify_soc"])
    proposal = LLMProposedPlan(rationale="single most relevant action for this evidence", actions=[LLMProposedAction(capability="block_source")], confidence=0.9)
    chat_model = _FakeChatModel([proposal])

    result = run_graph(chat_model, alert, context)

    assert result["plan"] is not None
    assert [a.capability for a in result["plan"].actions] == ["block_source"]


def test_empty_actions_rejected():
    alert, context = make_context()
    empty_plan = LLMProposedPlan(rationale="nothing to do", actions=[], confidence=0.1)
    chat_model = _FakeChatModel([empty_plan, empty_plan])

    result = run_graph(chat_model, alert, context, max_attempts=2)

    assert result["plan"] is None


def test_null_valued_parameters_are_dropped_not_rejected():
    """A local model routinely emits {"target": null} for a parameter it chose
    not to set - found live with qwen3:4b, which failed every plan this way."""
    alert, context = make_context()
    proposal = LLMProposedPlan(
        rationale="blocking the source of this scan",
        actions=[LLMProposedAction(capability="block_source", target="plc-01", parameters={"target": None, "duration": 300})],
        confidence=0.8,
    )
    chat_model = _FakeChatModel([proposal])

    result = run_graph(chat_model, alert, context)

    assert result["plan"] is not None
    assert result["plan"].actions[0].parameters == {"duration": 300}


def test_json_mode_requests_json_mode_and_includes_schema_in_prompt():
    """Verified live: gemma3:4b returns HTTP 400 'does not support tools' under
    the default function_calling method - json_mode is the only way to use it."""
    alert, context = make_context()
    proposal = LLMProposedPlan(rationale="reasoned response for a json_mode model", actions=[LLMProposedAction(capability="block_source")], confidence=0.7)
    chat_model = _FakeChatModel([proposal])
    graph = build_decision_graph(chat_model, max_attempts=1, structured_output_method="json_mode")
    state = initial_state(alert, context, repeat_activity=False, repeat_window_minutes=30, max_attempts=1)

    result = graph.invoke(state)

    assert result["plan"] is not None
    assert chat_model.with_structured_output_calls == [{"method": "json_mode"}]
    # The system message must carry the schema for a json_mode model, since it
    # gets no tool-calling metadata to infer the shape from.
    system_message = chat_model.runnable.invocations[0][0]
    assert "schema" in system_message.content.lower()
    assert "rationale" in system_message.content  # a real field name from LLMProposedPlan's schema


def test_function_calling_is_the_default_method():
    alert, context = make_context()
    proposal = LLMProposedPlan(rationale="default method response", actions=[LLMProposedAction(capability="block_source")], confidence=0.7)
    chat_model = _FakeChatModel([proposal])

    run_graph(chat_model, alert, context, max_attempts=1)

    assert chat_model.with_structured_output_calls == [{}]


def test_invoke_exception_handled_without_crashing():
    alert, context = make_context()
    good = LLMProposedPlan(rationale="recovered after a transient failure", actions=[LLMProposedAction(capability="block_source")], confidence=0.5)
    chat_model = _FakeChatModel([RuntimeError("ollama unreachable"), good])

    result = run_graph(chat_model, alert, context, max_attempts=2)

    assert result["plan"] is not None
    assert result["attempt"] == 2
