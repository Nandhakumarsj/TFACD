from tfacd.agentic.decision_engine import AgenticDecisionEngine
from tfacd.agentic.history import EntityHistory
from tfacd.runtime.contracts import IDSAlert, ThreatContext


def make_context(severity="medium", priority="P2", playbooks=None):
    alert = IDSAlert(attack_type="Port_Scanning", confidence=0.7, source_id="10.0.0.5", target_asset="plc-01")
    return ThreatContext(
        alert=alert, severity=severity, priority=priority,
        mitre_techniques=[], allowed_playbooks=playbooks or ["block_source", "increase_logging"],
    )


def test_first_incident_produces_plan_with_matching_actions():
    engine = AgenticDecisionEngine(history=EntityHistory())
    context = make_context()
    plan = engine.decide(context.alert, context)

    assert [a.capability for a in plan.actions] == context.allowed_playbooks
    assert "Port_Scanning" in plan.rationale
    assert "Repeat activity" not in plan.rationale
    assert plan.confidence == context.alert.confidence


def test_repeat_activity_from_same_source_is_correlated():
    history = EntityHistory()
    engine = AgenticDecisionEngine(history=history)
    context = make_context()

    first_plan = engine.decide(context.alert, context)
    second_plan = engine.decide(context.alert, context)

    assert "Repeat activity" in second_plan.rationale
    assert second_plan.confidence > context.alert.confidence
    assert first_plan.incident_id != second_plan.incident_id


def test_plan_defaults_to_template_engine():
    engine = AgenticDecisionEngine(history=EntityHistory())
    context = make_context()
    plan = engine.decide(context.alert, context)

    assert plan.engine == "template"
