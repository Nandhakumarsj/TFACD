"""Real and pluggable capability executors for active cyber defense containment.

Extends the trust boundary's CapabilityExecutor protocol with real drivers:
- CommandExecutor: Safely executes OS firewall/containment commands (iptables, netsh, custom scripts).
- WebhookExecutor: Dispatches JSON action telemetry to SIEM/SOAR HTTP webhooks.
- PluggableCapabilityExecutor: Routes specific capabilities to registered execution drivers with fallback.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import urllib.request
from typing import Any, Callable

from tfacd.runtime.contracts import CyberAction

logger = logging.getLogger(__name__)

# Sanitization: target/parameters must not contain shell injection characters
_SAFE_ARG_PATTERN = re.compile(r"^[a-zA-Z0-9_.:/\-]+$")


def sanitize_cmd_arg(value: str) -> str:
    """Ensures a command line argument contains no shell metacharacters."""
    val = value.strip()
    if not _SAFE_ARG_PATTERN.match(val):
        raise ValueError(f"unsafe argument for command execution: {value!r}")
    return val


class CommandExecutor:
    """Executes real system-level commands (e.g. firewall blocking, QoS shaping).

    Uses parameterized template strings per capability with strict argument sanitization.
    """

    DEFAULT_TEMPLATES = {
        "block_source": "netsh advfirewall firewall add rule name=\"Block_{target}\" dir=in action=block remoteip={target}",
        "isolate_host": "netsh advfirewall firewall add rule name=\"Isolate_{target}\" dir=in action=block remoteip={target}",
        "log_audit_event": "echo [AUDIT] Executed log for {target}",
    }

    def __init__(
        self,
        command_templates: dict[str, str] | None = None,
        timeout_seconds: float = 5.0,
        dry_run: bool = False,
    ):
        self.templates = command_templates or self.DEFAULT_TEMPLATES
        self.timeout = timeout_seconds
        self.dry_run = dry_run

    def execute(self, action: CyberAction) -> bool:
        template = self.templates.get(action.capability)
        if not template:
            logger.warning("No command template for capability '%s'", action.capability)
            return False

        target_str = sanitize_cmd_arg(action.target or "unknown-target")
        params = {"target": target_str}
        for k, v in action.parameters.items():
            if isinstance(v, str):
                params[k] = sanitize_cmd_arg(v)
            else:
                params[k] = str(v)

        try:
            cmd = template.format(**params)
        except KeyError as exc:
            logger.error("Missing template parameter for %s: %s", action.capability, exc)
            return False

        if self.dry_run:
            logger.info("[DRY-RUN CommandExecutor] would run: %s", cmd)
            return True

        logger.info("Executing command capability %s: %s", action.capability, cmd)
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
            if proc.returncode == 0:
                logger.info("Command execution succeeded: %s", proc.stdout.strip())
                return True
            logger.error("Command execution failed (code %d): %s", proc.returncode, proc.stderr.strip())
            return False
        except Exception as exc:
            logger.error("Command execution error for %s: %s", action.capability, exc)
            return False


class WebhookExecutor:
    """Dispatches HTTP POST JSON webhooks to SOC / SIEM endpoints."""

    def __init__(
        self,
        webhook_url: str,
        auth_header: str | None = None,
        timeout_seconds: float = 5.0,
        dry_run: bool = False,
    ):
        self.webhook_url = webhook_url
        self.auth_header = auth_header
        self.timeout = timeout_seconds
        self.dry_run = dry_run

    def execute(self, action: CyberAction) -> bool:
        payload = {
            "capability": action.capability,
            "target": action.target,
            "parameters": action.parameters,
            "source": "TFACD_AdaptiveSemanticTrustBoundary",
        }

        if self.dry_run:
            logger.info("[DRY-RUN WebhookExecutor] would POST to %s: %s", self.webhook_url, payload)
            return True

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            if self.auth_header:
                req.add_header("Authorization", self.auth_header)

            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                success = 200 <= resp.status < 300
                logger.info("Webhook POST to %s status=%d", self.webhook_url, resp.status)
                return success
        except Exception as exc:
            logger.error("Webhook dispatch error for %s to %s: %s", action.capability, self.webhook_url, exc)
            return False


class PluggableCapabilityExecutor:
    """Multiplexer routing specific capability strings to dedicated driver implementations,
    with fallback to a default executor (e.g. SimulatedExecutor).
    """

    def __init__(
        self,
        routes: dict[str, Any] | None = None,
        default_executor: Any | None = None,
    ):
        from tfacd.trust_boundary.capability_enforcement import SimulatedExecutor
        self.routes = routes or {}
        self.default_executor = default_executor or SimulatedExecutor()

    def register_route(self, capability: str, executor: Any) -> None:
        self.routes[capability] = executor

    def execute(self, action: CyberAction) -> bool:
        executor = self.routes.get(action.capability, self.default_executor)
        return executor.execute(action)
