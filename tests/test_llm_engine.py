from tfacd.agentic.decision_engine import AgenticDecisionEngine
from tfacd.agentic.graph import LLMProposedAction, LLMProposedPlan
from tfacd.agentic.history import EntityHistory
from tfacd.agentic.llm_engine import LLMDecisionEngine
from tfacd.runtime.contracts import IDSAlert, ThreatContext


def make_context(playbooks=None):
    alert = IDSAlert(attack_type="Port_Scanning", confidence=0.7, source_id="10.0.0.5", target_asset="plc-01")
    context = ThreatContext(
        alert=alert, severity="medium", priority="P2", mitre_techniques=[],
        allowed_playbooks=playbooks or ["block_source", "increase_logging", "create_ticket"],
    )
    return alert, context


class _FakeStructuredRunnable:
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

    def with_structured_output(self, schema):
        return self.runnable


def test_success_path_stamps_llm_engine_and_respects_allowed_playbooks():
    alert, context = make_context()
    proposal = LLMProposedPlan(rationale="port scanning observed, blocking source", actions=[LLMProposedAction(capability="block_source", target="plc-01")], confidence=0.8)
    engine = LLMDecisionEngine(_FakeChatModel([proposal]), history=EntityHistory())

    plan = engine.decide(alert, context)

    assert plan.engine == "llm"
    assert set(a.capability for a in plan.actions) <= set(context.allowed_playbooks)
    assert plan.incident_id.startswith(f"{alert.source_id}-{context.priority}-")


def test_exhausted_retries_falls_back_and_matches_deterministic_engine_output():
    alert, context = make_context()
    bad = LLMProposedPlan(rationale="x", actions=[LLMProposedAction(capability="shutdown_plant")], confidence=0.5)
    llm_history = EntityHistory()
    engine = LLMDecisionEngine(_FakeChatModel([bad, bad]), history=llm_history, max_attempts=2)

    plan = engine.decide(alert, context)

    reference_engine = AgenticDecisionEngine(history=EntityHistory())
    reference_plan = reference_engine.decide(alert, context)

    assert plan.engine.startswith("fallback:")
    assert [a.capability for a in plan.actions] == [a.capability for a in reference_plan.actions]
    assert plan.rationale == reference_plan.rationale
    assert plan.confidence == reference_plan.confidence


def test_chat_model_exception_falls_back_with_error_reason():
    alert, context = make_context()
    engine = LLMDecisionEngine(_FakeChatModel([RuntimeError("ollama unreachable"), RuntimeError("ollama unreachable")]), history=EntityHistory(), max_attempts=2)

    plan = engine.decide(alert, context)

    assert plan.engine.startswith("fallback:exhausted:") or plan.engine.startswith("fallback:error:")


def test_repeat_activity_reaches_the_prompt():
    alert, context = make_context()
    proposal = LLMProposedPlan(rationale="first", actions=[LLMProposedAction(capability="block_source")], confidence=0.5)
    proposal2 = LLMProposedPlan(rationale="second", actions=[LLMProposedAction(capability="block_source")], confidence=0.5)
    chat_model = _FakeChatModel([proposal, proposal2])
    engine = LLMDecisionEngine(chat_model, history=EntityHistory())

    engine.decide(alert, context)
    engine.decide(alert, context)

    second_call_messages = chat_model.runnable.invocations[1]
    assert any("repeat_activity" in str(m.content) for m in second_call_messages)


def test_successful_decision_appends_incident_history_event():
    alert, context = make_context()
    proposal = LLMProposedPlan(rationale="port scanning observed", actions=[LLMProposedAction(capability="block_source")], confidence=0.8)
    history = EntityHistory()
    engine = LLMDecisionEngine(_FakeChatModel([proposal]), history=history)

    engine.decide(alert, context)

    events = history.recent(alert.source_id, kind="incident")
    assert len(events) == 1
    assert events[0]["payload"]["attack_type"] == "Port_Scanning"


def test_hard_ceiling_never_crossed_even_when_llm_always_proposes_disallowed_capability():
    alert, context = make_context(playbooks=["block_source"])
    bad = LLMProposedPlan(rationale="x", actions=[LLMProposedAction(capability="shutdown_plant")], confidence=0.5)
    engine = LLMDecisionEngine(_FakeChatModel([bad, bad]), history=EntityHistory(), max_attempts=2)

    plan = engine.decide(alert, context)

    assert all(a.capability in context.allowed_playbooks for a in plan.actions)
    assert "shutdown_plant" not in [a.capability for a in plan.actions]


def test_primary_exhausted_retries_to_fallback_model_which_succeeds():
    alert, context = make_context()
    bad = LLMProposedPlan(rationale="x", actions=[LLMProposedAction(capability="shutdown_plant")], confidence=0.5)
    good = LLMProposedPlan(rationale="fallback model succeeded", actions=[LLMProposedAction(capability="block_source")], confidence=0.7)

    engine = LLMDecisionEngine(
        _FakeChatModel([bad, bad]),
        fallback_chat_model=_FakeChatModel([good]),
        history=EntityHistory(),
        max_attempts=2,
    )

    plan = engine.decide(alert, context)

    assert plan.engine == "llm:fallback_model"
    assert [a.capability for a in plan.actions] == ["block_source"]


def test_both_primary_and_fallback_model_fail_drops_to_deterministic_engine():
    alert, context = make_context()
    bad = LLMProposedPlan(rationale="x", actions=[LLMProposedAction(capability="shutdown_plant")], confidence=0.5)

    engine = LLMDecisionEngine(
        _FakeChatModel([bad, bad]),
        fallback_chat_model=_FakeChatModel([bad, bad]),
        history=EntityHistory(),
        max_attempts=2,
    )

    plan = engine.decide(alert, context)

    assert plan.engine.startswith("fallback:")
    assert "primary:" in plan.engine
    assert "fallback_model:" in plan.engine


def test_no_fallback_model_configured_behaves_exactly_as_before():
    alert, context = make_context()
    proposal = LLMProposedPlan(rationale="ok", actions=[LLMProposedAction(capability="block_source")], confidence=0.8)
    engine = LLMDecisionEngine(_FakeChatModel([proposal]), history=EntityHistory())

    assert engine._fallback_model_graph is None
    plan = engine.decide(alert, context)
    assert plan.engine == "llm"
