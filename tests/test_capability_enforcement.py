from tfacd.common.config import load_config
from tfacd.runtime.contracts import CyberAction, CyberActionPlan, IDSAlert, ThreatContext
from tfacd.trust_boundary.capability_enforcement import SimulatedExecutor, enforce

POLICY = load_config("configs/trust_policy.yaml")


class RecordingExecutor:
    def __init__(self):
        self.calls = []

    def execute(self, action):
        self.calls.append(action.capability)
        return True


def make_plan():
    return CyberActionPlan(
        incident_id="i", rationale="r", confidence=0.8,
        actions=[CyberAction(capability="observe"), CyberAction(capability="block_source")],
    )


def make_context(allowed_playbooks=("observe", "block_source")):
    alert = IDSAlert(attack_type="Port_Scanning", confidence=0.7)
    return ThreatContext(alert=alert, severity="medium", priority="P2", mitre_techniques=[], allowed_playbooks=list(allowed_playbooks))


def test_read_only_and_recommendation_execute_nothing():
    for mode in ("read_only", "recommendation"):
        executor = RecordingExecutor()
        executed = enforce(make_plan(), mode, POLICY, executor, make_context())
        assert executed == []
        assert executor.calls == []


def test_restricted_action_only_executes_low_risk():
    executor = RecordingExecutor()
    executed = enforce(make_plan(), "restricted_action", POLICY, executor, make_context())
    assert executed == ["observe"]


def test_autonomous_execution_executes_everything_whitelisted():
    executor = RecordingExecutor()
    executed = enforce(make_plan(), "autonomous_execution", POLICY, executor, make_context())
    assert set(executed) == {"observe", "block_source"}


def test_unwhitelisted_capability_skipped_even_when_autonomous():
    plan = CyberActionPlan(incident_id="i", rationale="r", confidence=0.8, actions=[CyberAction(capability="delete_all_data")])
    executor = RecordingExecutor()
    executed = enforce(plan, "autonomous_execution", POLICY, executor, make_context())
    assert executed == []


def test_whitelisted_but_not_context_allowed_capability_skipped():
    """The defense-in-depth re-check this task added: block_source is a real
    whitelisted capability, but if THIS incident's context never authorized it
    (e.g. a low-severity context that only allows "observe"), it must still be
    skipped even at autonomous_execution - closing the gap where only the
    static whitelist, not context.allowed_playbooks, was re-verified here."""
    executor = RecordingExecutor()
    context = make_context(allowed_playbooks=("observe",))  # block_source deliberately not authorized for this context
    executed = enforce(make_plan(), "autonomous_execution", POLICY, executor, context)
    assert executed == ["observe"]
    assert "block_source" not in executor.calls


def test_simulated_executor_returns_success():
    result = SimulatedExecutor().execute(CyberAction(capability="observe"))
    assert result is True
