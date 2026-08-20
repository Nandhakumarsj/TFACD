import json
import subprocess

from tfacd.runtime.contracts import CyberAction
from tfacd.trust_boundary.production_executor import ProductionExecutor


class FakeRunner:
    """Stubbed subprocess runner - records every call, never touches the real
    OS. Returns success (returncode=0) unless a capability is in fail_on."""

    def __init__(self, fail_on=frozenset()):
        self.calls: list[list[str]] = []
        self.fail_on = fail_on

    def __call__(self, args, **kwargs):
        self.calls.append(args)
        returncode = 1 if any(token in self.fail_on for token in args) else 0
        return subprocess.CompletedProcess(args=args, returncode=returncode)


def make_executor(system, **kwargs):
    runner = FakeRunner()
    executor = ProductionExecutor(run_subprocess=runner, system=system, **kwargs)
    return executor, runner


def test_windows_block_source_builds_netsh_command_and_records_it(tmp_path):
    executor, runner = make_executor("Windows", action_log_path=tmp_path / "actions.jsonl")
    action = CyberAction(capability="block_source", target="203.0.113.5")

    result = executor.execute(action)

    assert result is True
    assert runner.calls == [["netsh", "advfirewall", "firewall", "add", "rule", "name=tfacd_block_source_203.0.113.5", "dir=in", "action=block", "remoteip=203.0.113.5"]]
    record = json.loads((tmp_path / "actions.jsonl").read_text().splitlines()[0])
    assert record["capability"] == "block_source"
    assert record["backend"] == "windows_netsh"


def test_windows_rate_limit_has_no_backend_and_falls_back_to_simulated():
    executor, runner = make_executor("Windows")
    action = CyberAction(capability="rate_limit", target="203.0.113.5")

    result = executor.execute(action)

    assert result is True  # SimulatedExecutor always "succeeds"
    assert runner.calls == []  # no real command was ever built


def test_linux_block_source_uses_nftables_when_available(tmp_path, monkeypatch):
    monkeypatch.setattr("tfacd.trust_boundary.production_executor.shutil.which", lambda name: "/usr/sbin/nft")
    executor, runner = make_executor("Linux", action_log_path=tmp_path / "actions.jsonl")
    action = CyberAction(capability="block_source", target="203.0.113.5")

    result = executor.execute(action)

    assert result is True
    assert runner.calls[0][:4] == ["nft", "add", "table", "inet"]
    assert runner.calls[-1] == ["nft", "add", "rule", "inet", "tfacd", "input", "ip", "saddr", "203.0.113.5", "drop"]


def test_linux_rate_limit_adds_accept_then_drop_rule(monkeypatch):
    monkeypatch.setattr("tfacd.trust_boundary.production_executor.shutil.which", lambda name: "/usr/sbin/nft")
    executor, runner = make_executor("Linux")
    action = CyberAction(capability="rate_limit", target="203.0.113.5", parameters={"requests_per_second": 25})

    executor.execute(action)

    accept_call = next(c for c in runner.calls if "accept" in c)
    assert "25/second" in accept_call
    assert runner.calls[-1] == ["nft", "add", "rule", "inet", "tfacd", "input", "ip", "saddr", "203.0.113.5", "drop"]


def test_linux_falls_back_to_iptables_when_nft_missing(monkeypatch):
    monkeypatch.setattr("tfacd.trust_boundary.production_executor.shutil.which", lambda name: None)
    executor, runner = make_executor("Linux")
    action = CyberAction(capability="block_source", target="203.0.113.5")

    result = executor.execute(action)

    assert result is True
    assert runner.calls == [["iptables", "-I", "INPUT", "-s", "203.0.113.5", "-j", "DROP"]]


def test_protected_target_refused_before_any_command_is_built():
    executor, runner = make_executor("Linux", protected_targets=["203.0.113.0/24"])
    action = CyberAction(capability="block_source", target="203.0.113.5")

    result = executor.execute(action)

    assert result is False
    assert runner.calls == []


def test_non_ip_target_refused_cleanly_not_guessed():
    """isolate_segment's target is an asset NAME (e.g. "plc-01") in this
    project's data model, not a network address - see decision_engine.py.
    No asset inventory exists to translate it, so this must refuse, not
    fabricate a mapping."""
    executor, runner = make_executor("Linux")
    action = CyberAction(capability="isolate_segment", target="plc-01")

    result = executor.execute(action)

    assert result is False
    assert runner.calls == []


def test_missing_target_refused():
    executor, runner = make_executor("Linux")
    action = CyberAction(capability="block_source", target=None)

    assert executor.execute(action) is False
    assert runner.calls == []


def test_rotate_session_writes_real_record_no_subprocess_call(tmp_path):
    executor, runner = make_executor("Linux", session_rotation_log_path=tmp_path / "rotations.jsonl")
    action = CyberAction(capability="rotate_session", target="plc-01")

    result = executor.execute(action)

    assert result is True
    assert runner.calls == []
    record = json.loads((tmp_path / "rotations.jsonl").read_text().splitlines()[0])
    assert record["capability"] == "rotate_session"
    assert record["target"] == "plc-01"


def test_low_risk_capability_writes_real_record_no_subprocess_call(tmp_path):
    executor, runner = make_executor("Linux", action_log_path=tmp_path / "actions.jsonl")
    action = CyberAction(capability="observe", target="plc-01")

    result = executor.execute(action)

    assert result is True
    assert runner.calls == []
    record = json.loads((tmp_path / "actions.jsonl").read_text().splitlines()[0])
    assert record["capability"] == "observe"
    assert record["backend"] == "log_only"


def test_unknown_capability_falls_back_to_simulated():
    executor, runner = make_executor("Linux")
    action = CyberAction(capability="totally_unknown_capability", target="plc-01")

    result = executor.execute(action)

    assert result is True
    assert runner.calls == []


def test_mode_class_attribute_is_production():
    assert ProductionExecutor.mode == "production"
