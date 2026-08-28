from tfacd.runtime.contracts import CyberAction
from tfacd.trust_boundary.capability_enforcement import build_executor_from_config
from tfacd.trust_boundary.executors import CommandExecutor, PluggableCapabilityExecutor, WebhookExecutor, sanitize_cmd_arg


def test_sanitize_cmd_arg():
    assert sanitize_cmd_arg("192.168.1.100") == "192.168.1.100"
    assert sanitize_cmd_arg("gateway-01") == "gateway-01"
    try:
        sanitize_cmd_arg("192.168.1.1; rm -rf /")
        assert False, "Should have raised ValueError for shell metacharacters"
    except ValueError:
        pass


def test_command_executor_dry_run():
    executor = CommandExecutor(dry_run=True)
    action = CyberAction(capability="block_source", target="192.168.1.50")
    assert executor.execute(action) is True


def test_webhook_executor_dry_run():
    executor = WebhookExecutor(webhook_url="http://localhost:8080/test", dry_run=True)
    action = CyberAction(capability="alert_soc", target="plc-node-1")
    assert executor.execute(action) is True


def test_pluggable_capability_executor():
    cmd_exec = CommandExecutor(dry_run=True)
    webhook_exec = WebhookExecutor(webhook_url="http://localhost:8080/test", dry_run=True)
    pluggable = PluggableCapabilityExecutor(
        routes={"block_source": cmd_exec, "alert_soc": webhook_exec}
    )
    assert pluggable.execute(CyberAction(capability="block_source", target="10.0.0.1")) is True
    assert pluggable.execute(CyberAction(capability="alert_soc", target="10.0.0.2")) is True
    assert pluggable.execute(CyberAction(capability="other_action", target="10.0.0.3")) is True


def test_build_executor_from_config():
    config = {
        "capability_execution": {
            "driver": "command",
            "dry_run": True,
        }
    }
    executor = build_executor_from_config(config)
    assert isinstance(executor, CommandExecutor)
