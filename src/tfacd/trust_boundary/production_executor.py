"""Real (non-simulated) CapabilityExecutor backend - the deployment-specific
extension point SimulatedExecutor's own docstring anticipated
(capability_enforcement.py). Config-switchable per trust_boundary.executor.mode
("simulate", today's default and behavior, or "production") - see
executor_factory.py for how a caller picks between the two; this module never
gets constructed unless "production" is explicitly configured.

Detects the OS once (platform.system()) and dispatches each capability to a
small per-capability backend. Linux (nftables, iptables fallback) is the
primary, most-complete real backend - it's this project's actual IIoT
edge-deployment target. Windows (netsh advfirewall) exists so this can be
live-fire tested on a dev workstation without a Linux box on hand; it has one
documented gap (rate_limit has no reliable built-in CLI equivalent) rather
than a fake implementation.

Every trust DECISION still flows through the existing, unmodified hash-chained
audit log (trust_boundary/audit.py) - this module adds no competing audit
mechanism for that. What it does add is its own real-side-effect log
(action_log_path) for capabilities that have nothing else to write to
(rotate_session, the six low-risk capabilities), and the actual OS commands
for the network-acting ones.

Honest limitation, not silently worked around: isolate_segment's target is
alert.target_asset (an asset NAME like "plc-01" in this project's data model -
see agentic/decision_engine.py), not a network address. This project has no
asset-name-to-IP/CIDR inventory anywhere, so a real firewall rule cannot be
built from a bare asset name. _validate_network_target refuses cleanly (return
False, loud log) rather than fabricate a mapping - isolate_segment only
executes for real when action.target already IS a parseable IP/CIDR.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from tfacd.runtime.contracts import CyberAction

logger = logging.getLogger(__name__)

DEFAULT_ACTION_LOG_PATH = "artifacts/trust_boundary/production_action_log.jsonl"
DEFAULT_SESSION_ROTATION_LOG_PATH = "artifacts/trust_boundary/session_rotation_requests.jsonl"
DEFAULT_PROTECTED_TARGETS = ["127.0.0.1/32", "169.254.0.0/16", "::1/128"]

_NETWORK_CAPABILITIES = {"block_source", "isolate_segment", "rate_limit"}
_LOG_ONLY_CAPABILITIES = {"observe", "log_event", "increase_logging", "create_ticket", "notify_soc", "start_capture"}
_NFT_TABLE = "tfacd"


def _is_valid_network_target(target: str) -> bool:
    try:
        ipaddress.ip_network(target, strict=False)
        return True
    except ValueError:
        return False


class ProductionExecutor:
    """Real CapabilityExecutor. See module docstring for scope and the
    isolate_segment/asset-name limitation."""

    mode = "production"

    def __init__(
        self,
        protected_targets: list[str] | None = None,
        action_log_path: str | Path = DEFAULT_ACTION_LOG_PATH,
        session_rotation_log_path: str | Path = DEFAULT_SESSION_ROTATION_LOG_PATH,
        run_subprocess: Callable[..., "subprocess.CompletedProcess"] = subprocess.run,
        system: str | None = None,
    ):
        # Hard rail, independent of and in addition to the trust boundary's
        # own whitelist/autonomy-mode checks - checked before any command is
        # built for a network-acting capability. The default (loopback +
        # link-local) is a sane minimum, not a complete safety net: pass your
        # real management/gateway addresses explicitly for your deployment.
        self.protected_targets = protected_targets if protected_targets is not None else list(DEFAULT_PROTECTED_TARGETS)
        self.action_log_path = Path(action_log_path)
        self.session_rotation_log_path = Path(session_rotation_log_path)
        self._run = run_subprocess
        self.system = system or platform.system()  # "Windows" | "Linux" | ...
        self._simulated_fallback = None  # lazily constructed only if actually needed

    def execute(self, action: CyberAction) -> bool:
        if action.capability in _NETWORK_CAPABILITIES:
            return self._network_action(action)
        if action.capability == "rotate_session":
            return self._rotate_session(action)
        if action.capability in _LOG_ONLY_CAPABILITIES:
            return self._log_only(action)
        logger.warning("ProductionExecutor has no backend for capability=%s - falling back to simulated logging", action.capability)
        return self._fallback_to_simulated(action)

    # ---- safety rail ----

    def _is_protected(self, target: str) -> bool:
        candidate = ipaddress.ip_network(target, strict=False)
        for protected in self.protected_targets:
            try:
                if candidate.overlaps(ipaddress.ip_network(protected, strict=False)):
                    return True
            except ValueError:
                continue
        return False

    # ---- network-acting capabilities (block_source, isolate_segment, rate_limit) ----

    def _network_action(self, action: CyberAction) -> bool:
        target = action.target
        if not target:
            logger.warning("ProductionExecutor: capability=%s has no target - nothing to act on", action.capability)
            return False
        if not _is_valid_network_target(target):
            logger.warning(
                "ProductionExecutor refusing capability=%s target=%r: not a parseable IP/CIDR (this project has no "
                "asset-name-to-address inventory - see module docstring) - refusing rather than guessing", action.capability, target,
            )
            return False
        if self._is_protected(target):
            logger.warning("ProductionExecutor refusing capability=%s target=%s: target is in protected_targets", action.capability, target)
            return False

        if self.system == "Windows":
            return self._windows_network_action(action, target)
        if self.system == "Linux":
            return self._linux_network_action(action, target)

        logger.warning("ProductionExecutor: no real backend for system=%s - falling back to simulated logging", self.system)
        return self._fallback_to_simulated(action)

    # ---- Windows backend: netsh advfirewall ----

    def _windows_rule_name(self, action: CyberAction, target: str) -> str:
        return f"tfacd_{action.capability}_{target}".replace("/", "_").replace(":", "_")

    def _windows_network_action(self, action: CyberAction, target: str) -> bool:
        if action.capability == "rate_limit":
            # No reliable built-in Windows CLI equivalent to a Linux tc/nft
            # rate limiter - documented gap, not a fake implementation.
            logger.warning("ProductionExecutor: rate_limit has no Windows backend - falling back to simulated logging")
            return self._fallback_to_simulated(action)

        rule_name = self._windows_rule_name(action, target)
        args = ["netsh", "advfirewall", "firewall", "add", "rule", f"name={rule_name}", "dir=in", "action=block", f"remoteip={target}"]
        result = self._run(args, capture_output=True, text=True, timeout=15)
        self._record_action(action, target, backend="windows_netsh", command=args, returncode=result.returncode)
        return result.returncode == 0

    def remove_windows_block_rule(self, action: CyberAction, target: str) -> bool:
        """Not part of CapabilityExecutor - a live-fire test/operator utility
        to remove a rule added by _windows_network_action, e.g. after
        verifying the mechanism against an RFC 5737 test address."""
        rule_name = self._windows_rule_name(action, target)
        args = ["netsh", "advfirewall", "firewall", "delete", "rule", f"name={rule_name}"]
        result = self._run(args, capture_output=True, text=True, timeout=15)
        return result.returncode == 0

    # ---- Linux backend: nftables (self-contained tfacd table/chain), iptables fallback ----

    def _ensure_nft_base(self) -> None:
        # nft's "add" is idempotent (unlike "create") - safe to run every time.
        self._run(["nft", "add", "table", "inet", _NFT_TABLE], capture_output=True, text=True, timeout=15)
        self._run(
            ["nft", "add", "chain", "inet", _NFT_TABLE, "input", "{", "type", "filter", "hook", "input", "priority", "0", ";", "}"],
            capture_output=True, text=True, timeout=15,
        )

    def _linux_network_action(self, action: CyberAction, target: str) -> bool:
        if shutil.which("nft") is None:
            return self._linux_iptables_fallback(action, target)

        self._ensure_nft_base()
        if action.capability == "rate_limit":
            rate = action.parameters.get("requests_per_second", 10)
            accept_args = ["nft", "add", "rule", "inet", _NFT_TABLE, "input", "ip", "saddr", target, "limit", "rate", f"{int(rate)}/second", "accept"]
            self._run(accept_args, capture_output=True, text=True, timeout=15)
            args = ["nft", "add", "rule", "inet", _NFT_TABLE, "input", "ip", "saddr", target, "drop"]
        else:
            args = ["nft", "add", "rule", "inet", _NFT_TABLE, "input", "ip", "saddr", target, "drop"]

        result = self._run(args, capture_output=True, text=True, timeout=15)
        self._record_action(action, target, backend="linux_nftables", command=args, returncode=result.returncode)
        return result.returncode == 0

    def _linux_iptables_fallback(self, action: CyberAction, target: str) -> bool:
        if action.capability == "rate_limit":
            self._run(
                ["iptables", "-I", "INPUT", "-s", target, "-m", "limit", "--limit", "10/second", "-j", "ACCEPT"],
                capture_output=True, text=True, timeout=15,
            )
        args = ["iptables", "-I", "INPUT", "-s", target, "-j", "DROP"]
        result = self._run(args, capture_output=True, text=True, timeout=15)
        self._record_action(action, target, backend="linux_iptables", command=args, returncode=result.returncode)
        return result.returncode == 0

    # ---- rotate_session: real, but application-level, not OS-level ----

    def _rotate_session(self, action: CyberAction) -> bool:
        """No OS/network action - and, checked directly against this
        project's actual SessionContext usage, no live session-issuance
        service exists anywhere to rotate synchronously against either
        (every script here mints a fresh SessionContext with a new nonce per
        call already). What IS real: a persisted, timestamped
        rotation-request record, so a real deployment's session-issuance
        layer has something genuine to consume - not a fabricated "rotated"
        claim with nothing behind it.
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "capability": action.capability,
            "target": action.target,
            "parameters": action.parameters,
        }
        self.session_rotation_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.session_rotation_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
        return True

    # ---- low-risk capabilities: real persisted log entry, no OS/network action ----

    def _log_only(self, action: CyberAction) -> bool:
        self._record_action(action, action.target, backend="log_only", command=None, returncode=0)
        return True

    def _record_action(self, action: CyberAction, target: str | None, *, backend: str, command, returncode: int) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "capability": action.capability,
            "target": target,
            "backend": backend,
            "command": command,
            "returncode": returncode,
        }
        self.action_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.action_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
        logger.info("ProductionExecutor executed: %s", record)

    def _fallback_to_simulated(self, action: CyberAction) -> bool:
        from tfacd.trust_boundary.capability_enforcement import SimulatedExecutor

        if self._simulated_fallback is None:
            self._simulated_fallback = SimulatedExecutor()
        logger.warning(
            "ProductionExecutor falling back to SIMULATED execution for capability=%s target=%s - no real backend available",
            action.capability, action.target,
        )
        return self._simulated_fallback.execute(action)
