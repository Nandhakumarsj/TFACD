from tfacd.common.config import load_config
from tfacd.runtime.contracts import CyberAction, CyberActionPlan
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


def test_read_only_and_recommendation_execute_nothing():
    for mode in ("read_only", "recommendation"):
        executor = RecordingExecutor()
        executed = enforce(make_plan(), mode, POLICY, executor)
        assert executed == []
        assert executor.calls == []


def test_restricted_action_only_executes_low_risk():
    executor = RecordingExecutor()
    executed = enforce(make_plan(), "restricted_action", POLICY, executor)
    assert executed == ["observe"]


def test_autonomous_execution_executes_everything_whitelisted():
    executor = RecordingExecutor()
    executed = enforce(make_plan(), "autonomous_execution", POLICY, executor)
    assert set(executed) == {"observe", "block_source"}


def test_unwhitelisted_capability_skipped_even_when_autonomous():
    plan = CyberActionPlan(incident_id="i", rationale="r", confidence=0.8, actions=[CyberAction(capability="delete_all_data")])
    executor = RecordingExecutor()
    executed = enforce(plan, "autonomous_execution", POLICY, executor)
    assert executed == []


def test_simulated_executor_returns_success():
    result = SimulatedExecutor().execute(CyberAction(capability="observe"))
    assert result is True
